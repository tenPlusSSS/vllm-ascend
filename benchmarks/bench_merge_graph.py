# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare PCP candidate argsort/topk in captured NPU graphs.

The default cases mirror the torch paths at vllm-ascend commit 4ee3591.
--pcp-triton also loads the B/C production implementations from the installed
branch. Rank-major input packets are prepared in advance. Timings exclude
HCCL and model execution. CPU mode checks the reference operations only.
"""

import argparse
import json
import random
import statistics
from functools import partial
from pathlib import Path

import torch

PATTERNS = ("random", "ties", "near_ties", "few_valid", "all_invalid")
METHODS = ("argsort", "topk_sorted", "topk_unsorted")


def pack_topk_candidates(indices, scores, key_start, shard_len):
    """Mirror the production packet, preserving FP32 score bits."""
    valid = (indices >= 0) & (indices < shard_len[:, None]) & ~torch.isnan(scores) & (scores > -float("inf"))
    ids = torch.where(valid, indices + key_start[:, None], -1).to(torch.int32)
    ids, order = torch.sort(ids, dim=-1, stable=True)
    values = scores.masked_fill(~valid, -float("inf")).gather(1, order).contiguous()
    return torch.stack((ids, values.view(torch.int32)), dim=-1)


def unpack(packet, world, slots, k):
    gathered = packet.reshape(world, slots, k, 2)
    return gathered.transpose(0, 1).reshape(slots, world * k, 2)


def extract_scores(packet, world, slots, k):
    return unpack(packet, world, slots, k)[:, :, 1].contiguous().view(torch.float32)


def select(scores, k, method):
    if method == "argsort":
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return torch.topk(scores, k, dim=1, largest=True, sorted=method == "topk_sorted").indices


def merge(data, world, slots, k, method):
    """Production post-AllGather path; only the selection method varies."""
    gathered = unpack(data["packet"], world, slots, k)
    scores = gathered[:, :, 1].contiguous().view(torch.float32)
    order = select(scores, k, method)
    merged = gathered[:, :, 0].gather(1, order).unsqueeze(1)
    tokens = data["output_source"].numel()
    padding = merged.new_full((1, 1, k), -1)
    combined = torch.cat((data["native_ids"][:tokens], merged, padding))
    return combined.index_select(0, data["output_source"])


def pack_selected(data):
    indices = data["native_ids"].index_select(0, data["shared_row"]).squeeze(1)
    scores = data["native_scores"].index_select(0, data["shared_row"]).squeeze(1).float()
    return pack_topk_candidates(indices, scores, data["key_start"], data["shard_len"])


def make_inputs(world, slots, k, pattern, seed):
    generator = torch.Generator().manual_seed(seed)
    shard_len = max(32768, k)
    shape = (world, slots, k)
    ids = torch.stack([torch.randperm(shard_len, generator=generator)[:k] for _ in range(world * slots)])
    ids = ids.reshape(shape).to(torch.int32)
    count = world * slots * k
    permutation = torch.randperm(count, generator=generator).reshape(shape)
    scores = (permutation.float() - count / 2) / 1024
    if pattern == "ties":
        scores = (permutation % 4).float()
        # Include both signed zeros without changing their mathematical score.
        scores[..., 0] = -0.0
        scores[..., 1] = 0.0
    elif pattern == "near_ties":
        # Adjacent FP32 values: epsilon-based tie breaking would corrupt these.
        scores = (permutation.to(torch.int32) + 0x3F800000).contiguous().view(torch.float32)
    elif pattern == "few_valid":
        ids[..., max(1, k // (2 * world)) :] = -1
    elif pattern == "all_invalid":
        ids.fill_(-1)
    if pattern in ("ties", "few_valid", "all_invalid"):
        scores[..., -1] = float("nan")
        scores[..., -2] = -float("inf")
        # Exercise IDs outside the shard as well as negative placeholders.
        ids[..., -3] = shard_len
        if pattern == "ties":
            scores[..., -4] = float("inf")
    starts = torch.arange(world, dtype=torch.int64)[:, None].expand(world, slots).reshape(-1) * shard_len
    lengths = torch.full((world * slots,), shard_len, dtype=torch.int32)
    packet = pack_topk_candidates(ids.reshape(-1, k), scores.reshape(-1, k), starts, lengths)
    # Like the native QLI buffers, an owner or helper can land in a different
    # row on the next replay. The rank-major AllGather packets stay fixed-size.
    offset = slots if seed % 2 else 0
    native_ids = torch.cat((ids[0], torch.full_like(ids[0], -1))).roll(offset, 0).unsqueeze(1)
    native_scores = torch.cat((scores[0], torch.full_like(scores[0], -float("inf")))).roll(offset, 0).unsqueeze(1)
    score_order = packet[..., 1].contiguous().view(torch.float32).argsort(dim=1, descending=True, stable=True)
    return {
        "packet": packet,
        "sorted_packet": packet.gather(1, score_order[..., None].expand(-1, -1, 2)),
        "scores": extract_scores(packet, world, slots, k),
        "native_ids": native_ids,
        "native_scores": native_scores,
        "shared_row": offset + torch.arange(slots),
        "key_start": starts[:slots],
        "shard_len": lengths[:slots],
        "output_source": slots + torch.arange(slots),
    }


def make_cases(data, world, slots, k):
    cases = {
        "pack": partial(pack_selected, data),
        "unpack_scores": partial(extract_scores, data["packet"], world, slots, k),
    }
    for method in METHODS:
        cases[f"select.{method}"] = partial(select, data["scores"], k, method)
        cases[f"merge.{method}"] = partial(merge, data, world, slots, k, method)
    return cases


def add_triton_cases(cases, data, world, slots):
    from vllm_ascend.attention.context_parallel.pcp_indexer import merge_candidates_torch
    from vllm_ascend.ops.triton.pcp_indexer_candidates import merge_sorted_candidates, pack_sorted_candidates

    cases["pack.BC"] = partial(
        pack_sorted_candidates,
        data["native_ids"],
        data["native_scores"],
        data["shared_row"],
        data["key_start"],
        data["shard_len"],
    )
    args = (data["sorted_packet"], data["native_ids"], data["output_source"], slots, slots, world)
    cases["merge.B"] = partial(merge_candidates_torch, *args)
    cases["merge.C"] = partial(merge_sorted_candidates, *args)


def check_output(name, output, cpu, world, slots, k):
    actual = output.detach().cpu()
    if name == "pack.BC":
        torch.testing.assert_close(actual, cpu["sorted_packet"][:slots], rtol=0, atol=0)
        return {"packet_bits_exact": True}
    if name == "pack":
        torch.testing.assert_close(actual, cpu["packet"][:slots], rtol=0, atol=0)
        return {"packet_bits_exact": True}
    if name == "unpack_scores":
        torch.testing.assert_close(actual, cpu["scores"], rtol=0, atol=0)
        return {"scores_exact": True}
    gathered = unpack(cpu["packet"], world, slots, k)
    candidate_ids = gathered[:, :, 0]
    scores = cpu["scores"]
    if name.startswith("select."):
        assert actual.shape == (slots, k)
        assert ((actual >= 0) & (actual < world * k)).all()
        assert all(row.unique().numel() == k for row in actual)
        actual = candidate_ids.gather(1, actual.long())
    else:
        assert actual.shape == (slots, 1, k)
        actual = actual.squeeze(1)
    overlaps, exact = [], []
    for slot in range(slots):
        # Independent host reference: score descending, then global token ID.
        entries = list(zip(candidate_ids[slot].tolist(), scores[slot].tolist()))
        expected = sorted(entries, key=lambda item: (-item[1], item[0]))[:k]
        expected_ids = [item[0] for item in expected]
        lookup = dict(entries)
        actual_ids = actual[slot].tolist()
        assert all(token_id in lookup for token_id in actual_ids)
        valid = [token_id for token_id in actual_ids if token_id >= 0]
        expected_valid = {token_id for token_id in expected_ids if token_id >= 0}
        assert len(valid) == len(expected_valid), "Missing valid candidates or invalid sentinel leak"
        assert len(valid) == len(set(valid)), "Duplicate valid candidates"
        actual_scores = [lookup[token_id] for token_id in actual_ids]
        assert sorted(actual_scores, reverse=True) == [item[1] for item in expected], "Wrong top-K score multiset"
        if not name.endswith("topk_unsorted"):
            assert actual_scores == sorted(actual_scores, reverse=True), "Scores are not descending"
        exact.append(actual_ids == expected_ids)
        if name.endswith("argsort") or name in ("merge.B", "merge.C"):
            assert exact[-1], "Exact merge must preserve score/ID ordering"
        overlaps.append(len(set(valid) & expected_valid) / max(1, len(expected_valid)) if expected_valid else 1.0)
    return {
        "validity_and_scores_pass": True,
        "exact_ids_and_order": all(exact),
        "min_valid_id_overlap": min(overlaps),
        "note": "Matching scores alone does not validate relaxed ties for model quality.",
    }


class CapturedCase:
    def __init__(self, fn, unroll):
        self.stream = torch.npu.Stream()
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            for _ in range(5):
                fn()
        self.stream.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.stream):
            # Retain every output so allocator reuse cannot confound the graph.
            self.outputs = [fn() for _ in range(unroll)]
        self.stream.synchronize()

    def replay_and_read(self):
        with torch.npu.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()
        return self.outputs[-1]

    def sample_us(self, repeats, unroll):
        begin, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        with torch.npu.stream(self.stream):
            begin.record()
            for _ in range(repeats):
                self.graph.replay()
            end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000 / (repeats * unroll)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Packet shape only; no distributed process group",
    )
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--pattern", choices=PATTERNS, default="random")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument(
        "--unroll",
        type=int,
        default=16,
        help="Calls captured per graph; also try 1 to check sensitivity",
    )
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--target-step-ms", type=float, default=2.914)
    parser.add_argument("--cpu-check-only", action="store_true")
    parser.add_argument(
        "--pcp-triton", action="store_true", help="Benchmark actual B/C kernels from the installed branch"
    )
    parser.add_argument(
        "--trace-dir", type=Path, help="Write separate NPU replay traces for pack and merge after timing"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to a new file; existing files are not overwritten",
    )
    args = parser.parse_args()
    if args.pcp_triton and (args.cpu_check_only or args.topk != 2048):
        parser.error("--pcp-triton requires an NPU and --topk 2048")
    if args.trace_dir and args.cpu_check_only:
        parser.error("--trace-dir requires an NPU")
    for field in ("world_size", "slots", "unroll", "replays", "rounds", "layers"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.topk < 8 or args.world_size * args.slots * args.topk >= 2**23:
        parser.error("Require topk >= 8 and total candidate count < 2**23 for synthetic FP32 correctness cases")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    return args


def check_cases(args, data, cases, captured=None):
    results = {}
    for index, pattern in enumerate(PATTERNS):
        cpu = make_inputs(args.world_size, args.slots, args.topk, pattern, args.seed + index)
        for key, value in cpu.items():
            data[key].copy_(value.to(data[key].device))
        if captured is not None:
            torch.npu.synchronize()
        results[pattern] = {}
        for name, fn in cases.items():
            output = captured[name].replay_and_read() if captured is not None else fn()
            results[pattern][name] = check_output(name, output, cpu, args.world_size, args.slots, args.topk)
        print(
            f"Correctness: {pattern} passed (relaxed tie ordering is reported separately).",
            flush=True,
        )
    return results


@torch.inference_mode()
def main():
    args = parse_args()
    cpu = make_inputs(args.world_size, args.slots, args.topk, args.pattern, args.seed)
    report = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_version": torch.__version__,
        "scope": "Isolated single-device graph microbenchmark, no QLI/HCCL/model/TPOT measurement",
        "baseline_source": "vllm-ascend 4ee3591: pack_topk_candidates and finish_pcp_indexer_topk post-AllGather",
    }
    if args.cpu_check_only:
        data = {key: value.clone() for key, value in cpu.items()}
        cases = make_cases(data, args.world_size, args.slots, args.topk)
        report["correctness"] = check_cases(args, data, cases)
    else:
        import torch_npu  # NPU registration is unnecessary for CPU-only checks.

        torch.npu.set_device(args.device)
        report["torch_npu_version"] = torch_npu.__version__
        report["device_name"] = torch.npu.get_device_name(args.device)
        data = {key: value.to(f"npu:{args.device}") for key, value in cpu.items()}
        cases = make_cases(data, args.world_size, args.slots, args.topk)
        torch.npu.synchronize()
        if args.pcp_triton:
            add_triton_cases(cases, data, args.world_size, args.slots)
        captured = {name: CapturedCase(fn, args.unroll) for name, fn in cases.items()}
        report["correctness"] = check_cases(args, data, cases, captured)
        # Refresh the original pattern after correctness replay on changed inputs.
        for key, value in cpu.items():
            data[key].copy_(value.to(data[key].device))
        torch.npu.synchronize()
        for case in captured.values():
            for _ in range(3):
                case.replay_and_read()
        samples = {name: [] for name in cases}
        rng = random.Random(args.seed)
        for _ in range(args.rounds):
            names = list(cases)
            rng.shuffle(names)
            for name in names:
                samples[name].append(captured[name].sample_us(args.replays, args.unroll))
        report["timings"] = {
            name: {
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "samples_us": values,
            }
            for name, values in samples.items()
        }
        baseline = report["timings"]["merge.argsort"]["median_us"]
        for name, timing in report["timings"].items():
            print(f"{name:24s} {timing['median_us']:10.3f} us/call", flush=True)
        report["projection"] = {
            "required_us_per_layer": args.target_step_ms * 1000 / args.layers,
            "saved_ms_if_fully_exposed": {
                method: (baseline - report["timings"][f"merge.{method}"]["median_us"]) * args.layers / 1000
                for method in METHODS[1:]
            },
            "warning": "Arithmetic scaling only. Each stage is timed separately. Measure TPOT on the model.",
        }
        if args.pcp_triton:
            times = {name: result["median_us"] for name, result in report["timings"].items()}
            report["stage_savings_us"] = {
                "torch_pack_minus_B_pack": times["pack"] - times["pack.BC"],
                "B_merge_minus_C_merge": times["merge.B"] - times["merge.C"],
            }
        if args.trace_dir:
            prof = torch_npu.profiler
            for name in ("pack", "pack.BC", "merge.argsort", "merge.B", "merge.C"):
                if name in captured:
                    with prof.profile(
                        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                        on_trace_ready=prof.tensorboard_trace_handler(str(args.trace_dir / name)),
                    ):
                        captured[name].replay_and_read()
        print(json.dumps(report["projection"], indent=2), flush=True)
        if args.pcp_triton:
            print(json.dumps(report["stage_savings_us"], indent=2), flush=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.write("\n")
        print(f"Saved: {args.output}", flush=True)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
