# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the imported QLI V2 scores, without any custom scoring kernel."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.context_parallel.pcp_indexer import select_pcp_indexer_candidates

torch_npu = pytest.importorskip("torch_npu")

# arch22 QLI V2 uses SetFixpipePreQuantFlag(0x3a800000): QK / 1024.
# Compare native keys directly; every PCP rank uses the same positive scale.
QLI_V2_QK_SCALE = 1.0 / 1024.0


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("length", [1, 129, 4097])
@pytest.mark.parametrize("topk", [32, 2048])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_native_fp32_scores_with_mixed_causal_rows_and_padding(heads, length, topk, strided):
    torch.manual_seed(19)
    blocks = (length + 127) // 128
    pages = blocks + 2
    stride = 2 if strided else 1
    cache = torch.zeros(pages * stride, 128, 1, 128, dtype=torch.int8, device="npu")[::stride]
    scales = torch.ones(pages * stride, 128, 1, 1, dtype=torch.float16, device="npu")[::stride]
    keys_cpu = torch.randint(-4, 5, cache.shape, dtype=torch.int8)
    kscales_cpu = torch.exp2(-torch.randint(0, 4, scales.shape).float()).half()
    cache.copy_(keys_cpu.npu())
    scales.copy_(kscales_cpu.npu())
    q_cpu = torch.randint(-3, 4, (4, heads, 128), dtype=torch.int8)
    qs_cpu = torch.exp2(-torch.randint(0, 4, (4, heads)).float()).half()
    weights_cpu = torch.randint(-4, 5, (4, heads)).half()
    width = max(blocks, 2)
    table_cpu = torch.zeros(3, width, dtype=torch.int32)
    table_cpu[0, :blocks] = torch.arange(blocks).flip(0)
    table_cpu[1, :2] = torch.tensor([blocks + 1, blocks])
    # A decode, a two-token chunked prefill, and an unused appended request.
    batch = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3, 3], dtype=torch.int32, device="npu"),
        key_lens=torch.tensor([length, 129, 1], dtype=torch.int32, device="npu"),
        block_table=table_cpu.npu(),
        native_metadata={},
    )
    indices, scores = select_pcp_indexer_candidates(
        q_cpu.npu(), weights_cpu.npu(), qs_cpu.npu(), cache, scales, batch, topk
    )
    assert scores.dtype == torch.float32
    assert indices.shape == scores.shape == (4, 1, topk)
    indices, scores = indices.cpu(), scores.cpu()
    for row, request, visible in ((0, 0, length), (1, 1, 128), (2, 1, 129)):
        page_ids = table_cpu[request, : (visible + 127) // 128].long()
        keys = keys_cpu[page_ids].reshape(-1, 128)[:visible].float()
        key_scales = kscales_cpu[page_ids].flatten()[:visible].float()
        products = (q_cpu[row].float() @ keys.T) * qs_cpu[row].float()[:, None] * key_scales[None, :]
        reference = (products.clamp_min(0) * weights_cpu[row].float()[:, None]).sum(0) * QLI_V2_QK_SCALE
        ids, values = indices[row, 0].long(), scores[row, 0]
        valid = ids >= 0
        assert valid.sum() == min(visible, topk)
        assert ids[valid].unique().numel() == valid.sum()
        assert (ids[valid] < visible).all()
        assert torch.isneginf(values[~valid]).all()
        torch.testing.assert_close(values[valid], reference[ids[valid]], rtol=0, atol=0)
        cutoff = reference.topk(min(visible, topk)).values[-1]
        assert (values[valid] >= cutoff).all()
