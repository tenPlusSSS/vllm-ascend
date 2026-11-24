# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch_npu
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.device_op import A5DeviceAdaptor

_KERNEL_BLOCK_SIZE = 128


def _make_builder(
    pcp_size: int = 1,
    dcp_size: int = 1,
    num_speculative_tokens: int | None = None,
    pcp_indexer_sharding: bool = False,
) -> AscendSFAIndexerMetadataBuilder:
    kv_cache_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=160,
        dtype=torch.uint8,
    )
    layer_names = ["model.layers.0.self_attn.indexer.k_cache"]
    vllm_config = MagicMock()
    vllm_config.additional_config = {"enable_pcp_indexer_sharding": pcp_indexer_sharding}
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_topk=2048),
        hf_config=SimpleNamespace(),
        max_model_len=1024,
    )
    vllm_config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        tensor_parallel_size=max(4, dcp_size),
    )
    vllm_config.scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    vllm_config.speculative_config = (
        None if num_speculative_tokens is None else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    with (
        patch(
            "vllm_ascend.attention.indexer.select_common_block_size",
            return_value=_KERNEL_BLOCK_SIZE,
        ),
    ):
        return AscendSFAIndexerMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            torch.device("cpu"),
        )


def _make_common_metadata() -> SimpleNamespace:
    metadata = SimpleNamespace(
        pcp_context=None,
        pcp_cache_group_idx=None,
        num_reqs=2,
        num_actual_tokens=4,
        num_input_tokens=4,
        slot_mapping=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 0, 1, 9]),
        query_start_loc=torch.tensor([0, 2, 4]),
        query_start_loc_cpu=torch.tensor([0, 2, 4]),
        seq_lens=torch.tensor([5, 6, 7]),
        max_query_len=2,
        is_prefilling=torch.tensor([True, True]),
        context_parallel_metadata=None,
        block_table_tensor=torch.arange(6).view(3, 2),
    )

    def replace(**changes):
        values = vars(metadata).copy()
        values.pop("replace", None)
        values.update(changes)
        return SimpleNamespace(**values)

    metadata.replace = replace
    return metadata


def test_sfa_indexer_backend_contract():
    assert AscendSFAIndexerBackend.accept_output_buffer
    assert AscendSFAIndexerBackend.get_name() == "ASCEND_SFA_INDEXER"
    assert AscendSFAIndexerBackend.get_builder_cls() is AscendSFAIndexerMetadataBuilder
    assert AscendSFAIndexerBackend.get_kv_cache_shape(8, 128, 1, 160) == (
        8,
        128,
        1,
        160,
    )
    assert AscendSFAIndexerBackend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize("enabled,spec_tokens,expected", [(False, None, False), (True, None, True), (True, 2, False)])
@pytest.mark.parametrize("capture", [False, True])
def test_pcp_indexer_builder_only_selects_non_speculative_target(monkeypatch, enabled, spec_tokens, expected, capture):
    builder = _make_builder(pcp_size=2, num_speculative_tokens=spec_tokens, pcp_indexer_sharding=enabled)
    common = _make_common_metadata()
    common.pcp_context = _make_decode_pcp_context()
    common.pcp_cache_group_idx = 0
    selected = object()
    select = Mock(return_value=selected)
    monkeypatch.setattr("vllm_ascend.attention.indexer.build_pcp_indexer_metadata", select)
    monkeypatch.setattr(
        "vllm_ascend.attention.indexer.get_cos_and_sin_mla",
        lambda *args, **kwargs: (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8)),
    )
    metadata = builder.build_for_cudagraph_capture(common) if capture else builder.build(0, common)
    assert metadata.pcp_indexer is (selected if expected else None)
    assert select.call_count == int(expected)


@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.parametrize("local_row,actual_tokens", [(1, 2), (-1, 2), (-1, 0)])
def test_pcp_indexer_forward_updates_only_owner_after_cache_write(monkeypatch, local_row, actual_tokens, cooperative):
    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(indexer)
    indexer._use_fused_cache_store = False
    indexer.enable_sparse_li_c8 = cooperative
    indexer.enable_sparse_li_c4 = False
    indexer.c8_k_cache_dtype = torch.int8
    indexer._quantize_li_tensor = lambda q: (q.to(torch.int8), torch.ones(q.shape[:-1], dtype=torch.float16))
    monkeypatch.setattr("vllm_ascend.attention.indexer.supports_pcp_indexer_sharding", lambda *args: True)
    indexer.n_head, indexer.head_dim, indexer.topk_tokens = 2, 4, 4
    indexer.qk_rope_head_dim = 0
    indexer.is_rope_neox_style = True
    indexer.use_torch_npu_lightning_indexer = False
    indexer.wk_weights_proj = Mock()
    indexer.wq_b = Mock(return_value=(torch.ones(2, 8), None))
    key = torch.zeros(2, 128, 1, 4)
    indexer.k_cache = SimpleNamespace(kv_cache=(key, torch.ones(2, 128, 1, 1)))
    indexer.forward_k = Mock(return_value=(torch.ones(2, 1, 4), None, torch.ones(2, 2)))
    events = []

    def gather(k, scale, metadata):
        events.append("gather_k")
        return k, scale, metadata.slot_mapping

    indexer._gather_cache_inputs = gather
    indexer.write_cache = lambda *args: events.append("write_k")
    monkeypatch.setattr("vllm_ascend.attention.indexer.HAS_TRITON", True)
    monkeypatch.setattr("vllm_ascend.attention.indexer.rope_forward_triton_siso", lambda q, *args, **kwargs: q)
    ordinary = torch.tensor([[[0, 1, 2, 3]], [[4, 5, 6, 7]]], dtype=torch.int32)
    native_output = torch.cat((ordinary[:actual_tokens], ordinary[:1]))

    def native_select(*args, **kwargs):
        events.append("native_indexer")
        return native_output, native_scores

    native_scores = torch.ones_like(native_output, dtype=torch.float32)
    native = Mock(side_effect=native_select)
    ordinary_native = Mock(return_value=ordinary)
    monkeypatch.setattr("vllm_ascend.attention.indexer.DeviceOperator.indexer_select_post_process", ordinary_native)
    monkeypatch.setattr("vllm_ascend.attention.indexer.select_pcp_indexer_candidates", native)
    final_topk = torch.tensor([[[50, 60, 70, 80]]], dtype=torch.int32)

    def prepare(q, w, scale, metadata):
        events.append("copy_q")
        return (
            torch.cat((q[:actual_tokens], q[:1])),
            torch.cat((w[:actual_tokens], w[:1])),
            torch.cat((scale[:actual_tokens], scale[:1])),
        )

    def finish(indices, scores, metadata, output_tokens):
        events.append("merge")
        assert indices is native_output
        assert scores is native_scores
        assert output_tokens == 2
        result = ordinary.clone() if actual_tokens else torch.full_like(ordinary, -1)
        if local_row >= 0:
            result[local_row] = final_topk[0]
        return result

    monkeypatch.setattr("vllm_ascend.attention.indexer.prepare_pcp_indexer_queries", prepare)
    monkeypatch.setattr("vllm_ascend.attention.indexer.finish_pcp_indexer_topk", finish)
    shard_lens = torch.tensor([128, 128, 128])
    query_lens = torch.tensor([1, 2, 2 if local_row >= 0 else 3]) if actual_tokens else torch.tensor([1])
    batch_table = torch.tensor([[0, 1], [1, 0], [1, 0]])
    metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=actual_tokens,
        slot_mapping=torch.tensor([0, 128]),
        seq_lens=torch.tensor([128, 256]),
        cum_query_lens=torch.tensor([1, 2]),
        block_table=torch.tensor([[0, 1], [1, 0]]),
        sin=torch.zeros(2, 1, 1, 4),
        cos=torch.ones(2, 1, 1, 4),
        actual_seq_lengths_query=torch.tensor([1, 2]),
        actual_seq_lengths_key=torch.tensor([128, 256]),
        pcp_indexer=SimpleNamespace(
            metadata=SimpleNamespace(block_table=batch_table, query_lens=query_lens, key_lens=shard_lens),
        ),
    )
    hidden = torch.ones(2, 8)
    result = indexer.forward(hidden, hidden, hidden, metadata)
    if not cooperative:
        assert events == ["gather_k", "write_k"]
        native.assert_not_called()
        assert ordinary_native.call_count == int(actual_tokens > 0)
        if actual_tokens:
            torch.testing.assert_close(result, ordinary)
            assert ordinary_native.call_args.args[7] is metadata
        else:
            assert result is None
        return
    ordinary_native.assert_not_called()
    assert events == ["gather_k", "write_k", "copy_q", "native_indexer", "merge"]
    assert native.call_count == 1
    assert native.call_args.args[0].shape[0] == actual_tokens + 1
    assert native.call_args.args[5] is metadata.pcp_indexer.metadata
    assert native.call_args.args[6] == 4
    assert metadata.actual_seq_lengths_key.tolist() == [128, 256]
    expected = ordinary.clone() if actual_tokens else torch.full_like(ordinary, -1)
    if local_row >= 0:
        expected[local_row] = final_topk[0]
    torch.testing.assert_close(result, expected)


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_kernel_metadata(mock_cos_sin, mock_get_ascend_config):
    cos = torch.zeros(5, 1, 1, 8)
    sin = torch.zeros(5, 1, 1, 8)
    mock_cos_sin.return_value = (cos, sin)

    builder = _make_builder()
    assert builder.reorder_batch_threshold is None
    assert builder.get_cudagraph_support(MagicMock(), MagicMock()) is AttentionCGSupport.UNIFORM_BATCH

    common = _make_common_metadata()
    metadata = builder.build(0, common)

    assert isinstance(metadata, AscendSFAIndexerMetadata)
    assert metadata.num_actual_tokens == 4
    assert not metadata.uses_pcp_sharded_layout
    assert torch.equal(metadata.slot_mapping, common.slot_mapping[:4])
    assert torch.equal(metadata.seq_lens, common.seq_lens[:2])
    assert torch.equal(metadata.cum_query_lens, common.query_start_loc[1:3])
    assert torch.equal(metadata.block_table, common.block_table_tensor[:2])
    assert metadata.block_size == _KERNEL_BLOCK_SIZE
    assert torch.equal(metadata.actual_seq_lengths_query, common.query_start_loc[1:3])
    assert torch.equal(metadata.actual_seq_lengths_key, common.seq_lens[:2])
    assert metadata.num_decode_tokens == 0

    positions = mock_cos_sin.call_args.args[0]
    assert torch.equal(positions, common.positions[:4])
    assert mock_cos_sin.call_args.kwargs["use_cache"] is True
    assert torch.equal(metadata.cos, cos[:4])
    assert torch.equal(metadata.sin, sin[:4])
    # Ordinary target execution reuses the framework's graph-stable RoPE
    # tables instead of introducing another asynchronously copied buffer.
    assert metadata.cos.data_ptr() == cos.data_ptr()
    assert metadata.sin.data_ptr() == sin.data_ptr()


def _make_decode_pcp_context() -> SimpleNamespace:
    # Only the global-prefill flag is read when no request is prefilling.
    return SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([False])))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_emits_full_slot_mapping_under_pcp(mock_cos_sin, mock_get_ascend_config):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))

    builder = _make_builder(pcp_size=2)
    common = _make_common_metadata()
    common.pcp_context = _make_decode_pcp_context()
    metadata = builder.build(0, common)

    # Under PCP the commit writes the gathered prefill region too, so the
    # builder emits the full slot mapping instead of the input-token slice.
    assert metadata.uses_pcp_sharded_layout
    assert metadata.slot_mapping is common.slot_mapping


@pytest.mark.parametrize("capture", [False, True])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_native_dcp_decode_boundary(mock_cos_sin, mock_get_ascend_config, capture):
    mock_get_ascend_config.return_value.c8_reshape_optim_enabled = False
    mock_cos_sin.return_value = (torch.zeros(2, 1, 1, 8), torch.zeros(2, 1, 1, 8))
    common = _make_common_metadata()
    common.num_actual_tokens = common.num_input_tokens = 2
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 1
    common.is_prefilling = torch.tensor([False, False])
    builder = _make_builder(pcp_size=1, dcp_size=8)
    metadata = builder.build_for_cudagraph_capture(common) if capture else builder.build(0, common)
    assert metadata.num_decode_tokens == metadata.num_actual_tokens == 2


@pytest.mark.parametrize("pcp_size,dcp_size,expected_decode_tokens", [(2, 1, 1), (2, 2, 1), (1, 8, 1)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_pcp_decode_boundary(
    mock_cos_sin, mock_get_ascend_config, pcp_size, dcp_size, expected_decode_tokens
):
    mock_cos_sin.return_value = (torch.zeros(4, 1, 1, 8), torch.zeros(4, 1, 1, 8))
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 3
    common.is_prefilling = torch.tensor([False, True])

    common.pcp_context = _make_decode_pcp_context()

    metadata = _make_builder(pcp_size=pcp_size, dcp_size=dcp_size).build(0, common)

    # Preserve the boundary for PCP gathering and native DCP fused stores.
    assert metadata.num_decode_tokens == expected_decode_tokens


@pytest.mark.parametrize("pcp_size,dcp_size", [(2, 2), (1, 8)])
@pytest.mark.parametrize("pd_decode_recompute,expected_decode_tokens", [(False, 0), (True, 2)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_preserves_rescheduled_prefill_boundary(
    mock_cos_sin,
    mock_get_ascend_config,
    pcp_size,
    dcp_size,
    pd_decode_recompute,
    expected_decode_tokens,
):
    mock_cos_sin.return_value = (torch.zeros(2, 1, 1, 8), torch.zeros(2, 1, 1, 8))
    common = _make_common_metadata()
    common.num_actual_tokens = 2
    common.num_input_tokens = 2
    common.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.max_query_len = 1
    common.is_prefilling = torch.tensor([True, True])

    with patch(
        "vllm_ascend.attention.indexer.is_pd_decode_recompute_scheduler_enabled",
        return_value=pd_decode_recompute,
    ):
        common.pcp_context = _make_decode_pcp_context()
        metadata = _make_builder(pcp_size=pcp_size, dcp_size=dcp_size).build(0, common)

    # Short rescheduled prefills remain prefills unless the PD decode
    # recompute scheduler intentionally treats them as decode requests.
    assert metadata.num_decode_tokens == expected_decode_tokens


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_owns_replicated_dcp_addresses(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    common = _make_common_metadata()

    metadata = _make_builder(dcp_size=2).build(0, common)

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([0, 1, 512, 513], dtype=torch.int32),
    )
    # The builder derives the replicated view without changing the local view
    # consumed by the SFA cache backend.
    torch.testing.assert_close(common.slot_mapping, torch.tensor([1, 2, 3, 4, 5]))
    torch.testing.assert_close(common.block_table_tensor, torch.arange(6).view(3, 2))


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_draft_metadata_owns_per_step_dcp_buffers(
    mock_cos_sin,
    mock_get_ascend_config,
):
    mock_cos_sin.return_value = (
        torch.zeros(5, 1, 1, 8),
        torch.zeros(5, 1, 1, 8),
    )
    builder = _make_builder(dcp_size=2, num_speculative_tokens=3)
    common = _make_common_metadata()

    step_one = builder.build_for_drafting(common, 1)
    step_one_slots = step_one.slot_mapping.clone()
    step_one_blocks = step_one.block_table.clone()
    # The proposer owns one persistent slot tensor per logical draft step;
    # its address is also the key for every derived metadata buffer.
    common.slot_mapping = common.slot_mapping.clone()
    common.positions = torch.tensor([2, 3, 2, 3, 9])
    step_two = builder.build_for_drafting(common, 2)

    assert step_one.slot_mapping.data_ptr() != step_two.slot_mapping.data_ptr()
    assert step_one.block_table.data_ptr() != step_two.block_table.data_ptr()
    torch.testing.assert_close(step_one.slot_mapping, step_one_slots)
    torch.testing.assert_close(step_one.block_table, step_one_blocks)


@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_graph_capture_owns_stable_per_step_buffers(mock_cos_sin, mock_get_ascend_config):
    mock_cos_sin.side_effect = [
        (torch.full((4, 1, 1, 8), value), torch.full((4, 1, 1, 8), -value)) for value in range(4)
    ]
    builder = _make_builder(num_speculative_tokens=3)
    common = _make_common_metadata()

    first = builder.build_for_graph_capture(common)
    first_runtime = builder.build(common_prefix_len=0, common_attn_metadata=common)
    common.slot_mapping = common.slot_mapping.clone()
    second = builder.build_for_graph_capture(common)
    second_runtime = builder.build_for_drafting(common, draft_index=1)

    for field in ("slot_mapping", "cos", "sin"):
        assert getattr(first, field).data_ptr() != getattr(second, field).data_ptr()
        assert getattr(first, field).data_ptr() == getattr(first_runtime, field).data_ptr()
        assert getattr(second, field).data_ptr() == getattr(second_runtime, field).data_ptr()
    torch.testing.assert_close(first.cos, torch.ones_like(first.cos))
    torch.testing.assert_close(first.sin, -torch.ones_like(first.sin))
    torch.testing.assert_close(second.cos, torch.full_like(second.cos, 3))
    torch.testing.assert_close(second.sin, torch.full_like(second.sin, -3))


@pytest.mark.parametrize("for_cudagraph_capture", [False, True])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_sfa_indexer_metadata_builder_builds_pcp_dcp_slots(
    mock_cos_sin,
    mock_get_ascend_config,
    for_cudagraph_capture,
):
    mock_cos_sin.return_value = (torch.zeros(5, 1, 1, 8), torch.zeros(5, 1, 1, 8))
    common = _make_common_metadata()
    global_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        query_start_loc_np=torch.tensor([0, 3], dtype=torch.int32).numpy(),
        seq_lens=torch.tensor([384], dtype=torch.int32),
        positions=torch.tensor([0, 128, 256], dtype=torch.int64),
        is_prefilling_np=torch.tensor([True]),
    )
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(torch.tensor([[10, 11]], dtype=torch.int32),),
        padded_gather_idx=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        gathered_kv_write_mask=torch.tensor([True, True, True, False]),
    )

    builder = _make_builder(pcp_size=2, dcp_size=2)
    common.pcp_context = pcp_context
    common.pcp_cache_group_idx = 0
    if for_cudagraph_capture:
        metadata = builder.build_for_cudagraph_capture(common)
    else:
        metadata = builder.build(0, common)

    torch.testing.assert_close(
        metadata.block_table,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.slot_mapping,
        torch.tensor([2816, 2560, 2688, -1], dtype=torch.int32),
    )


def test_c4_cache_write_preserves_packed_bytes(monkeypatch):
    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    indexer.enable_sparse_li_c4 = True
    indexer.enable_sparse_li_c8 = False
    key_cache = torch.full((2, 128, 1, 64), 23, dtype=torch.uint8)
    scale_bytes = torch.full((2, 128, 1, 2, 2), 23, dtype=torch.uint8)
    indexer.k_cache = SimpleNamespace(kv_cache=(key_cache, scale_bytes.view(torch.float8_e8m0fnu)))
    packed_key = torch.arange(3 * 64, dtype=torch.uint8).view(3, 64)
    scale = torch.arange(12, dtype=torch.uint8).view(3, 2, 2)
    slots = torch.tensor([1, 129, 255])
    # A CPU tensor cannot use the NPU-only FP4 dtype. Its one-byte storage
    # surrogate exercises the packed-dtype branch without an NPU allocation.
    monkeypatch.setattr(torch_npu, "float4_e2m1fn_x2", torch.uint8, raising=False)

    def scatter(cache, indices, updates):
        assert cache.dtype == updates.dtype == torch.uint8
        cache[indices.flatten()] = updates

    scatter_op = Mock(side_effect=scatter)
    monkeypatch.setattr("vllm_ascend.attention.indexer.DeviceOperator.scatter_cache", scatter_op)

    indexer.write_cache(packed_key, scale.view(torch.float8_e8m0fnu), slots)

    expected_key = torch.full_like(key_cache, 23).view(-1, 64)
    expected_scale = torch.full_like(scale_bytes, 23).view(-1, 2, 2)
    expected_key[slots] = packed_key
    expected_scale[slots] = scale
    assert torch.equal(key_cache.view(-1, 64), expected_key)
    assert torch.equal(scale_bytes.view(-1, 2, 2), expected_scale)
    assert scatter_op.call_count == 2


@pytest.mark.parametrize("query_lengths,key_lengths", [([2, 3], [2, 3]), ([1, 1], [31, 65])])
def test_c4_a5_selector_operator_contract(monkeypatch, query_lengths, key_lengths):
    tokens, heads, head_dim = sum(query_lengths), 64, 128
    query = torch.zeros(tokens * heads, head_dim // 2, dtype=torch.uint8)
    query_scale = torch.zeros(tokens, heads, 2, 2, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    key = torch.zeros(4, 128, 1, head_dim // 2, dtype=torch.uint8)
    key_scale = torch.zeros(4, 128, 1, 2, 2, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weights = torch.arange(tokens * heads, dtype=torch.float32).view(tokens, heads).to(torch.bfloat16)
    query_ends = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
    key_lengths = torch.tensor(key_lengths, dtype=torch.int32)
    metadata = SimpleNamespace(block_table=torch.tensor([[1], [3]], dtype=torch.int32))
    kernel_metadata = object()
    expected = torch.zeros(tokens, 1, 2048, dtype=torch.int32)
    metadata_op = Mock(return_value=kernel_metadata)
    select_op = Mock(return_value=(expected, torch.empty(0)))
    monkeypatch.setattr(A5DeviceAdaptor, "_load_cann_quant_lightning_indexer_ops", lambda: (metadata_op, select_op))

    result = A5DeviceAdaptor.indexer_select_post_process(
        query,
        query_scale,
        (tokens, heads, head_dim),
        weights,
        (key, key_scale),
        0,
        1,
        metadata,
        query_ends,
        key_lengths,
        False,
        True,
        False,
    )

    assert result is expected
    q_arg, k_arg, w_arg, qs_arg, ks_arg = select_op.call_args.args
    assert q_arg.shape == (tokens, heads, head_dim // 2)
    assert q_arg.data_ptr() == query.data_ptr()
    assert k_arg is key and qs_arg is query_scale and ks_arg is key_scale
    assert w_arg.dtype == torch.float32
    torch.testing.assert_close(w_arg, weights.float())
    for call in (metadata_op.call_args, select_op.call_args):
        args = call.kwargs
        assert args["quant_mode"] == 5 and args["topk"] == 2048
        assert args["layout_q"] == "TND" and args["layout_k"] == "PA_BBND"
        assert args["mask_mode"] == 3 and args["cmp_ratio"] == 1
        assert torch.equal(args["cu_seqlens_q"], torch.cat((torch.zeros(1, dtype=torch.int32), query_ends)))
        assert args["seqused_k"] is key_lengths
    assert metadata_op.call_args.kwargs["head_dim"] == head_dim
    assert metadata_op.call_args.kwargs["num_heads_q"] == heads
    assert select_op.call_args.kwargs["metadata"] is kernel_metadata
    assert select_op.call_args.kwargs["block_table"] is metadata.block_table


def test_c4_cache_spec_accounts_for_packed_key_and_mx_scales():
    spec = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
        scale_dim=4,
        scale_dtype=torch.float8_e8m0fnu,
        cache_sparse_li_c4=True,
    )
    assert spec.real_page_size_bytes == 128 * (64 + 4)
    assert spec.page_size_bytes == spec.real_page_size_bytes


@pytest.mark.parametrize("padded_tokens,actual_tokens", [(1, 1), (4, 1), (4, 0)])
@patch("vllm_ascend.attention.indexer.get_ascend_config")
@patch("vllm_ascend.attention.indexer.get_cos_and_sin_mla")
def test_pcp_builder_resolves_replicated_draft_layout_with_padding(
    mock_cos_sin, mock_config, padded_tokens, actual_tokens
):
    common = _make_common_metadata()
    common.num_input_tokens = padded_tokens
    common.num_actual_tokens = actual_tokens
    common.slot_mapping = torch.full((padded_tokens,), -1, dtype=torch.int64)
    common.positions = torch.arange(padded_tokens)
    common.pcp_context = None
    mock_cos_sin.return_value = (torch.zeros(padded_tokens, 1, 1, 8), torch.zeros(padded_tokens, 1, 1, 8))
    metadata = _make_builder(pcp_size=2).build(0, common)
    assert not metadata.uses_pcp_sharded_layout
    assert metadata.slot_mapping.numel() == padded_tokens
