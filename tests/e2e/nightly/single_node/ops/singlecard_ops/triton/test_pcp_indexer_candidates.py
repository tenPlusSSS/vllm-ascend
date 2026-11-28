# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check exact PCP pack/merge results across replays on one NPU with synthetic shards."""

import pytest
import torch

from vllm_ascend.attention.context_parallel.pcp_indexer import merge_candidates_torch, pack_topk_candidates
from vllm_ascend.ops.triton.pcp_indexer_candidates import merge_sorted_candidates, pack_sorted_candidates


@pytest.mark.parametrize("slots,world", [(1, 8), (2, 4), (3, 5)])
@torch.inference_mode()
def test_replay_matches_torch_with_ties_padding_and_new_owners(slots, world):
    k, width = 2048, 2**25
    rows, tokens = slots * world, slots + 2
    rng = torch.Generator().manual_seed(721)
    # One row per simulated rank/slot; local IDs above 2**24 must stay distinct.
    ids = (2**24 + torch.arange(k, dtype=torch.int32)).repeat(rows, 1)
    ids[:, ::13] = -1
    ids[:, ::17] = width
    ids = ids[:, torch.randperm(k, generator=rng)].unsqueeze(1)
    score_values = torch.tensor([-float("inf"), -3.0, -0.0, 0.0, 1.0, 1.0 + 2**-23, float("inf"), float("nan")])
    scores = score_values[torch.randint(len(score_values), (rows, 1, k), generator=rng)]
    starts = torch.arange(world, dtype=torch.int64).repeat_interleave(slots) * width
    lengths = torch.full((rows,), width, dtype=torch.int32)
    shared = torch.arange(rows)
    sources = torch.tensor([*(tokens + i for i in range(slots)), 0, tokens + slots])
    inputs = [ids, scores, shared, starts, lengths, sources]
    d_ids, d_scores, d_shared, d_starts, d_lengths, d_sources = [x.npu() for x in inputs]

    def forward():
        packet = pack_sorted_candidates(d_ids, d_scores, d_shared, d_starts, d_lengths)
        args = (packet, d_ids, d_sources, tokens, slots, world)
        return packet, merge_candidates_torch(*args), merge_sorted_candidates(*args)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packet, b, c = forward()
    for phase in range(3):
        lengths[::2] = 0 if phase == 1 else width
        if phase == 2:
            lengths.zero_()
        selected_rows, routes = shared.roll(phase), sources.roll(phase)
        d_shared.copy_(selected_rows)
        d_sources.copy_(routes)
        d_lengths.copy_(lengths)
        torch.npu.synchronize()
        graph.replay()
        # A orders each shard by ID, then uses a global stable score sort.
        ref_packet = pack_topk_candidates(ids[selected_rows, 0], scores[selected_rows, 0], starts, lengths)
        expected = merge_candidates_torch(ref_packet, ids, routes, tokens, slots, world)
        order = ref_packet[..., 1].contiguous().view(torch.float32).argsort(dim=1, descending=True, stable=True)
        expected_packet = ref_packet.gather(1, order[..., None].expand(-1, -1, 2))
        assert torch.equal(packet.cpu(), expected_packet)
        assert torch.equal(b.cpu(), expected)
        assert torch.equal(c.cpu(), expected)
