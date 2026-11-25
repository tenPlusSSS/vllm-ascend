# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel import pcp_indexer
from vllm_ascend.attention.context_parallel.pcp_indexer import (
    select_pcp_indexer_candidates,
    supports_pcp_indexer_sharding,
)
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


@pytest.mark.parametrize(
    "minimum,cap,expected",
    [
        (131073, 5, (1, 2, 3)),
        (131073, 2, (2, 3)),
        (131073, 1, (2,)),
        (131072, 5, (0, 1, 2, 3)),
        (300000, 5, ()),
        (0, 0, ()),
    ],
)
@pytest.mark.parametrize("full_graph", [False, True])
def test_step_plan(minimum, cap, expected, full_graph):
    batch = SimpleNamespace(
        num_reqs=6,
        num_scheduled_tokens=np.array([1, 1, 1, 1, 3, 1, 1]),
        num_computed_tokens_np=np.array([131071, 131072, 199999, 199999, 250000, 250000, 250000]),
        is_prefilling_np=np.array([False, False, False, False, False, True, False]),
    )
    plan = pcp_indexer.plan_pcp_indexer_step(batch, minimum, cap, full_graph=full_graph)
    if expected:
        assert plan == pcp_indexer.PCPIndexerPlan(expected, cap if full_graph else len(expected), full_graph)
    else:
        assert plan is None


def score_row(query, weights, scales, cache, table, length, topk):
    keys = cache[table.long()].reshape(-1, query.shape[-1])[:length].float()
    logits = ((query.float() @ keys.T).relu() * (weights * scales).float()[:, None]).sum(0)
    order = torch.argsort(logits, descending=True, stable=True)[:topk]
    return order.to(torch.int32), logits[order]


def score_batch(query, weights, scales, cache, metadata, topk):
    indices = torch.full((query.shape[0], 1, topk), -1, dtype=torch.int32)
    scores = torch.full(indices.shape, -torch.inf)
    bounds = metadata.query_start_loc.tolist()
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        for row in range(start, end):
            visible = int(metadata.key_lens[i]) - (end - row - 1)
            order, values = score_row(
                query[row], weights[row], scales[row], cache, metadata.block_table[i], visible, topk
            )
            indices[row, 0, : order.numel()] = order
            scores[row, 0, : order.numel()] = values
    return indices, scores


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("full_graph", [False, True])
def test_multiple_requests_exchange_real_rank_payloads_and_restore_outputs(monkeypatch, world, full_graph):
    # Both collectives exchange the actual tensors emitted by every participant.
    # Keep one model bucket across steps while changing owners, lengths and physical pages.
    rank_local = threading.local()
    monkeypatch.setattr(pcp_indexer, "get_pcp_group", lambda: rank_local.group)
    barrier = threading.Barrier(world, timeout=15)
    packets = [None] * world
    tokens, block, topk, heads, dim = 5, 4, 5, 2, 4
    generator = torch.Generator().manual_seed(193)
    cache = torch.randint(-3, 4, (24, block, dim), generator=generator, dtype=torch.int8)
    storage = [{} for _ in range(world)]
    scenarios = [
        ([0, 1, 0], [7, 12, 1], [1, 1, 1], (0,)),
        ([1, 0, 1], [12, 4, 15], [1, 1, 1], (0, 2)),
        ([1, 0, 1], [1, 15, 16], [1, 1, 1], (0, 1, 2)),
        ([0, 1, 0], [8, 2, 3], [1, 1, 1] if full_graph else [3, 1, 1], (1, 2)),
        ([1], [1], [1], (0,)),
    ]

    def run_step(owners, lengths, scheduled, selected):
        offsets = np.cumsum([0, *scheduled])
        table = torch.randperm(24, generator=generator).view(6, 4)[: len(owners)].int()
        q = torch.randint(-3, 4, (offsets[-1], heads, dim), generator=generator, dtype=torch.int8)
        w = torch.randint(-3, 4, (offsets[-1], heads), generator=generator).half()
        scale = torch.randint(1, 5, w.shape, generator=generator).half() / 2
        q[:, 0, :2] = torch.tensor([-128, 127], dtype=torch.int8)
        w[:, 0] = -0.0
        scale[:, 0] = 0.333251953125
        rows_by_rank = [[i for i, owner in enumerate(owners) if owner == r] for r in range(world)]
        restore = torch.empty(offsets[-1], dtype=torch.int32)
        for r, reqs in enumerate(rows_by_rank):
            pos = r * tokens
            for i in reqs:
                restore[offsets[i] : offsets[i + 1]] = torch.arange(pos, pos + scheduled[i])
                pos += scheduled[i]
        batch = SimpleNamespace(query_start_loc_np=offsets, seq_lens=torch.tensor(lengths, dtype=torch.int32))
        slots = 3 if full_graph else len(selected)
        context = SimpleNamespace(
            global_batch=batch,
            global_block_tables=(table,),
            hidden_restore_idx=restore,
            indexer_plan=pcp_indexer.PCPIndexerPlan(selected, slots, full_graph),
        )

        def run_rank(rank):
            def all_gather(packet, dim):
                packets[rank] = packet
                barrier.wait()
                gathered = torch.cat(packets, dim=dim)
                barrier.wait()
                return gathered

            def all_reduce(packet):
                assert packet.dtype == torch.int8
                assert packet.shape == (slots, heads * (dim + 4))
                for slot in range(slots):
                    if slot >= len(selected) or owners[selected[slot]] != rank:
                        assert (packet[slot] == 0).all()
                packets[rank] = packet
                barrier.wait()
                reduced = torch.stack(packets).sum(dim=0, dtype=torch.int8)
                barrier.wait()
                return reduced

            rank_local.group = SimpleNamespace(
                rank_in_group=rank, world_size=world, all_gather=all_gather, all_reduce=all_reduce
            )
            reqs = rows_by_rank[rank]
            local_rows = [j for i in reqs for j in range(offsets[i], offsets[i + 1])]
            actual = len(local_rows)
            local_ends = np.cumsum([0, *(scheduled[i] for i in reqs)]).tolist()
            padded_reqs = tokens if full_graph else len(reqs)
            # FULL FIA gives its padding rows real model offsets, including on idle ranks.
            local_ends += list(range(actual + 1, actual + 1 + padded_reqs - len(reqs)))
            local_table = torch.zeros((padded_reqs, table.shape[1]), dtype=table.dtype)
            local_table[: len(reqs)] = table[reqs]
            common = SimpleNamespace(
                num_actual_tokens=actual,
                num_input_tokens=tokens,
                num_reqs=padded_reqs,
                query_start_loc=torch.tensor(local_ends, dtype=torch.int32),
                block_table_tensor=local_table,
                seq_lens=torch.tensor([lengths[i] for i in reqs] + [0] * (padded_reqs - len(reqs)), dtype=torch.int32),
            )
            inputs = pcp_indexer.build_pcp_indexer_metadata(common, context, 0, block, storage[rank])
            layout = inputs.metadata
            assert inputs.metadata is layout
            if full_graph:
                assert len(storage[rank]) == 1
                saved = next(iter(storage[rank].values()))
                assert saved.local_table.data_ptr() == inputs.local_table.data_ptr()
                assert "metadata" not in vars(saved)
            else:
                assert not storage[rank]
            for s, req in enumerate(selected):
                blocks = (lengths[req] + block - 1) // block
                first = blocks * rank // world * block
                last = min(lengths[req], blocks * (rank + 1) // world * block)
                assert (layout.key_start[s].item(), layout.shard_seq_len[s].item()) == (first, last - first)
            local_q, local_w, local_scale = (x[local_rows] for x in (q, w, scale))
            if full_graph or actual:

                def pad(x):
                    return torch.cat((x, x.new_zeros((tokens - actual, *x.shape[1:]))))

                local_q, local_w, local_scale = (pad(x) for x in (local_q, local_w, local_scale))
            nq, nw, ns = pcp_indexer.prepare_pcp_indexer_queries(local_q, local_w, local_scale, layout)
            for s, req in enumerate(selected):
                native_row = layout.shared_row[s]
                torch.testing.assert_close(nq[native_row], q[offsets[req]])
                torch.testing.assert_close(nw[native_row].view(torch.int16), w[offsets[req]].view(torch.int16))
                torch.testing.assert_close(ns[native_row].view(torch.int16), scale[offsets[req]].view(torch.int16))
            ids, scores = score_batch(nq, nw, ns, cache, layout, topk)
            result = pcp_indexer.finish_pcp_indexer_topk(ids, scores, layout, local_q.shape[0])
            for row, global_row in enumerate(local_rows):
                req = int(np.searchsorted(offsets, global_row, side="right") - 1)
                visible = lengths[req] - (offsets[req + 1] - global_row - 1)
                expected, _ = score_row(
                    q[global_row], w[global_row], scale[global_row], cache, table[req], visible, topk
                )
                torch.testing.assert_close(result[row, 0, : len(expected)], expected)
                assert (result[row, 0, len(expected) :] == -1).all()
            assert (result[actual:] == -1).all()

        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(run_rank, range(world)))

    for scenario in scenarios:
        run_step(*scenario)


def test_traced_layout_reads_new_lengths_owners_and_slot_counts(monkeypatch):
    monkeypatch.setattr(pcp_indexer, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=2))

    class Layout(torch.nn.Module):
        def forward(self, selection, restored, lengths, selected_table, offsets, local_lengths, local_table):
            inputs = pcp_indexer.PCPIndexerInputs(
                selection,
                restored,
                lengths,
                selected_table,
                offsets,
                local_lengths,
                local_table,
                tokens=2,
                block_size=4,
            )
            batch = inputs.metadata
            return (
                batch.query_start_loc,
                batch.key_lens,
                batch.block_table,
                batch.query_row,
                batch.query_source,
                batch.shared_row,
                batch.key_start,
                batch.shard_seq_len,
                batch.output_source,
            )

    selection = torch.zeros(2, 3, dtype=torch.int64)
    selection[:, -1] = torch.tensor([1, 2])
    restored = torch.tensor([0, 0])
    lengths = torch.tensor([1, 1], dtype=torch.int32)
    table = torch.arange(8, dtype=torch.int32).view(2, 4)
    args = (
        selection,
        restored,
        lengths,
        table,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        lengths.clone(),
        table.clone(),
    )
    layout = Layout()
    traced = torch.export.export(layout, args).module()
    for count, actual, owners, kv in [(2, 2, [2, 1], [13, 7]), (1, 0, [2, 0], [3, 1])]:
        selection[:, -1] = torch.tensor([count, actual])
        restored.copy_(torch.tensor(owners))
        lengths.copy_(torch.tensor(kv))
        table.copy_(table.flip(1))
        torch.testing.assert_close(traced(*args), layout(*args))


def test_candidate_packet_preserves_large_ids_and_invalid_scores():
    start = 2**24 + 1
    pairs = pcp_indexer.pack_topk_candidates(
        torch.tensor([[3, 1, -1, 2, 4]], dtype=torch.int32),
        torch.tensor([[2.0, -3.0, 9.0, float("nan"), -float("inf")]]),
        torch.tensor([start]),
        torch.tensor([4]),
    )
    assert pairs[0, :, 0].tolist() == [-1, -1, -1, start + 1, start + 3]
    assert pairs[0, :, 1].contiguous().view(torch.float32).tolist() == [-torch.inf] * 3 + [-3.0, 2.0]


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_native_sharding_preserves_non_int8_layers(monkeypatch, dtype):
    for family in DeviceAdaptorFamily:
        monkeypatch.setattr(
            "vllm_ascend.attention.context_parallel.pcp_indexer.get_current_hardware_profile",
            lambda family=family: SimpleNamespace(device_adaptor_family=family),
        )
        assert supports_pcp_indexer_sharding(dtype, 32, 128, 2048) == (
            family == DeviceAdaptorFamily.STANDARD and dtype == torch.int8
        )
        assert not supports_pcp_indexer_sharding(dtype, 8, 128, 2048)
        assert not supports_pcp_indexer_sharding(dtype, 32, 64, 2048)
        assert not supports_pcp_indexer_sharding(dtype, 32, 128, 4096)


def test_v2_returns_scores_once_and_reuses_batch_metadata_across_layers(monkeypatch):
    q = torch.zeros(3, 32, 128, dtype=torch.int8)
    cache = torch.zeros(8, 128, 1, 128, dtype=torch.int8)[::2]
    scales = torch.ones(8, 128, 1, 1, dtype=torch.float16)[::2]
    batch = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32),
        key_lens=torch.tensor([129, 128], dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        native_metadata={},
    )
    produced = torch.zeros(1024, dtype=torch.int32)
    outputs = (torch.zeros(3, 1, 2048, dtype=torch.int32), torch.ones(3, 1, 2048))
    producer, native = Mock(return_value=produced), Mock(return_value=outputs)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", producer, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", native, raising=False)
    for _ in range(2):
        assert (
            select_pcp_indexer_candidates(q, torch.ones(3, 32), torch.ones(3, 32), cache, scales, batch, 2048)
            is outputs
        )
    assert producer.call_count == 1
    assert native.call_count == 2
    kwargs = native.call_args.kwargs
    assert kwargs["return_value"] == 1 and kwargs["enable_dcp"] is False
    assert kwargs["mask_mode"] == 3 and kwargs["cmp_ratio"] == 1
    assert kwargs["layout_k"] == "PA_BBND"
    assert kwargs["metadata"] is produced
    assert kwargs["cu_seqlens_q"] is batch.query_start_loc
    assert kwargs["seqused_k"] is batch.key_lens
    assert kwargs["block_table"] is batch.block_table
    assert native.call_args.args[1] is cache
    assert native.call_args.args[4].stride(0) == scales.stride(0)
    assert all(t.dtype == torch.float16 for t in native.call_args.args[2:5])
