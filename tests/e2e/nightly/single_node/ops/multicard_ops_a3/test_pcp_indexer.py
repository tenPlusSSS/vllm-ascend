# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay both decode indexers with new request owners and paged KV inputs on PCP2/4."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel import pcp_indexer
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import enable_custom_op

torch_npu = pytest.importorskip("torch_npu")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(WORLD_SIZE not in (2, 4), reason="requires PCP2 or PCP4")


@pytest.mark.parametrize("topk,minimum", [(32, 129), (2048, 131073)])
@torch.inference_mode()
def test_threshold_changes_and_owner_switch_in_the_same_captured_bucket(monkeypatch, topk, minimum):
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    assert enable_custom_op()
    init_distributed_environment(
        world_size=WORLD_SIZE, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(WORLD_SIZE))],
        local_rank=local_rank,
        backend="hccl",
        group_name="pcp_indexer_test",
        use_device_communicator=True,
    )
    monkeypatch.setattr(pcp_indexer, "get_pcp_group", lambda: group)
    try:
        tokens, heads, dim, block, slots = 2, 32, 128, 128, 3
        scenarios = [
            ([0, 1, 0], [minimum - 1, 1, 128]),
            ([0, 1, 0], [minimum, 128, 1]),
            ([1, 0, 1], [max(8193, minimum + block), minimum, 1]),
            ([0, 1, 0], [minimum + block, minimum + 1, minimum + 3]),
            ([WORLD_SIZE - 1], [max(4097, minimum)]),
            ([1, 0], [1, 127]),
        ]
        width = (max(max(lengths) for _, lengths in scenarios) + block - 1) // block
        rng = torch.Generator().manual_seed(24)
        keys_cpu = torch.randint(-4, 5, (3 * width, block, 1, dim), dtype=torch.int8, generator=rng)
        cache = keys_cpu.npu()
        kscale = torch.ones(cache.shape[:3] + (1,), device="npu", dtype=torch.float16)
        q = torch.zeros(tokens, heads, dim, device="npu", dtype=torch.int8)
        w = torch.ones(tokens, heads, device="npu", dtype=torch.float16)
        scale = torch.ones_like(w)
        local_table = torch.zeros(tokens, width, dtype=torch.int32, device="npu")
        local_lens = torch.zeros(tokens, dtype=torch.int32, device="npu")
        offsets = torch.arange(tokens + 1, device="npu", dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=tokens,
            num_input_tokens=tokens,
            query_start_loc=offsets,
            block_table_tensor=local_table,
            seq_lens=local_lens,
        )
        graph_buffers = {}

        def stage(owners, lengths, capture_mixed=False):
            local = [i for i, owner in enumerate(owners) if owner == rank]
            n = len(owners)
            pages = torch.randperm(3 * width, generator=rng).view(3, width)[:n].int()
            query = torch.randint(-3, 4, (n, heads, dim), generator=rng, dtype=torch.int8)
            weights = torch.randint(-3, 4, (n, heads), generator=rng).half()
            scales = torch.exp2(-torch.randint(0, 4, (n, heads), generator=rng).float()).half()
            if local:
                q[: len(local)].copy_(query[local])
                w[: len(local)].copy_(weights[local])
                scale[: len(local)].copy_(scales[local])
                local_table[: len(local)].copy_(pages[local])
            local_lens.zero_()
            local_lens[: len(local)].copy_(torch.tensor([lengths[i] for i in local], dtype=torch.int32))
            common.num_actual_tokens = len(local)
            restored = [owner * tokens + owners[:i].count(owner) for i, owner in enumerate(owners)]
            batch = SimpleNamespace(
                num_reqs=n,
                num_scheduled_tokens=np.ones(n, dtype=np.int32),
                is_prefilling_np=np.zeros(n, dtype=np.bool_),
                num_computed_tokens_np=np.array(lengths) - 1,
                query_start_loc_np=np.arange(n + 1),
                seq_lens=torch.tensor(lengths, device="npu", dtype=torch.int32),
            )
            plan = pcp_indexer.plan_pcp_indexer_step(batch, minimum, slots, full_graph=True)
            if capture_mixed:
                plan = pcp_indexer.PCPIndexerPlan((0,), slots, True)
            context = SimpleNamespace(
                indexer_plan=plan,
                global_batch=batch,
                hidden_restore_idx=torch.tensor(restored, device="npu", dtype=torch.int32),
                global_block_tables=(pages.npu(),),
            )
            inputs = pcp_indexer.build_pcp_indexer_metadata(common, context, 0, block, graph_buffers)
            return inputs, local, query, weights, scales, pages

        def forward(inputs):
            if inputs is None:
                return DeviceOperator.indexer_select_post_process(
                    q,
                    scale,
                    q.shape,
                    w,
                    (cache, kscale),
                    0,
                    1,
                    SimpleNamespace(block_table=local_table),
                    offsets[1:],
                    local_lens,
                    True,
                    False,
                    True,
                    topk,
                )
            metadata = inputs.metadata
            packed_q, packed_w, packed_s = pcp_indexer.prepare_pcp_indexer_queries(q, w, scale, metadata)
            ids, scores = pcp_indexer.select_pcp_indexer_candidates(
                packed_q, packed_w, packed_s, cache, kscale, metadata, topk
            )
            return pcp_indexer.finish_pcp_indexer_topk(ids, scores, metadata, tokens)

        graphs = {}
        for mixed in (False, True):
            for _ in range(2):
                inputs, *_ = stage([0, 1, 0], [1, 1, 1], capture_mixed=mixed)
                forward(inputs)
            torch.npu.synchronize()
            inputs, *_ = stage([0, 1, 0], [1, 1, 1], capture_mixed=mixed)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = forward(inputs)
            graphs[mixed] = graph, output

        # No matches, one match, two on different owners, all selected, an idle
        # helper and a return to ordinary replay. Only two graphs are captured.
        for owners, lengths in scenarios:
            inputs, local, query, weights, scales, pages = stage(owners, lengths)
            graph, result = graphs[inputs is not None]
            graph.replay()
            result_cpu = result.cpu()
            for row, req in enumerate(local):
                length = lengths[req]
                page_ids = pages[req, : (length + block - 1) // block].long()
                keys = keys_cpu[page_ids].view(-1, dim)[:length].float()
                scores = ((query[req].float() @ keys.T).relu() * (scales[req] * weights[req]).float()[:, None]).sum(0)
                ids = result_cpu[row, 0].long()
                ids = ids[ids >= 0]
                assert ids.numel() == ids.unique().numel() == min(topk, length)
                assert (ids < length).all()
                # The ordinary and V2 operators may choose different boundary ties.
                torch.testing.assert_close(scores[ids].sort(descending=True).values, scores.topk(ids.numel()).values)
            if inputs is not None:
                assert (result_cpu[len(local) :] == -1).all()
    finally:
        group.destroy()
        destroy_distributed_environment()
