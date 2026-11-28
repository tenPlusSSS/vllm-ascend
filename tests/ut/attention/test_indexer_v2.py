# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.attention import indexer as indexer_module
from vllm_ascend.attention.indexer import (
    AscendSFAIndexerBackend,
    AscendSFAIndexerMetadata,
    AscendSFAIndexerMetadataBuilder,
)
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


@pytest.mark.parametrize("enabled,c8,expected", [(False, True, False), (True, True, True), (True, False, False)])
def test_unsharded_v2_initialization_preserves_layer_precision(monkeypatch, enabled, c8, expected):
    config = SimpleNamespace(
        enable_indexer_v2_unsharded=enabled,
        pcp_indexer_merge_backend="triton_merge",
        is_sparse_li_c8_layer=lambda prefix: c8,
        is_sparse_li_c4_layer=lambda prefix: False,
    )
    vc = SimpleNamespace(
        attention_config=SimpleNamespace(indexer_kv_dtype="int8"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm_moe_dsa")),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=8),
    )
    indexer = SimpleNamespace(
        n_head=32,
        head_dim=128,
        topk_tokens=2048,
        q_lora_rank=1536,
        wq_b=None,
        wk_weights_proj=None,
        k_norm=None,
        softmax_scale=1,
        k_cache=SimpleNamespace(prefix="model.layers.0.self_attn.indexer.k_cache"),
    )
    monkeypatch.setattr(indexer_module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(indexer_module, "get_current_vllm_config", lambda: vc)
    monkeypatch.setattr(indexer_module, "kv_cache_dtype_str_to_dtype", lambda *args: torch.int8)
    monkeypatch.setattr(indexer_module, "is_pcp_decode_sharding_enabled", lambda *args: True)
    monkeypatch.setattr(indexer_module, "supports_pcp_indexer_sharding", lambda *args: True)
    backend = AscendSFAIndexerBackend(indexer, qk_rope_head_dim=64)
    assert backend._use_v2_unsharded is expected
    assert backend.enable_sparse_li_c8 is c8
    assert backend.pcp_indexer_merge_backend == "triton_merge"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({}, None),
        ({"dtype": "auto"}, "indexer_kv_dtype=int8"),
        ({"c8": False}, "indexer_kv_dtype=int8"),
        ({"family": DeviceAdaptorFamily.FP8_OPTIMIZED}, "A2/A3"),
        ({"dcp": 2}, "DCP=1"),
        ({"spec": object()}, "no speculative"),
        ({"offload": True}, "sparse KV offload"),
        ({"filtered": True, "layers": set()}, "no C8 indexer layers"),
        ({"filtered": True, "layers": {1}}, None),
        ({"enabled": False, "dtype": "auto", "c8": False}, None),
    ],
)
def test_unsharded_v2_config_rejects_unsupported_experiments(monkeypatch, overrides, error):
    options = dict(
        enabled=True,
        dtype="int8",
        c8=True,
        family=DeviceAdaptorFamily.STANDARD,
        dcp=1,
        spec=None,
        offload=False,
        filtered=False,
        layers=set(),
    )
    options.update(overrides)
    config = SimpleNamespace(
        enable_indexer_v2_unsharded=options["enabled"],
        enable_sparse_li_c8=options["c8"],
        sparse_kv_offload_config=SimpleNamespace(enabled=options["offload"]),
        _sparse_li_layer_filter_enabled=options["filtered"],
        _sparse_li_c8_layer_ids=options["layers"],
        enable_pcp_indexer_sharding=False,
    )
    vc = SimpleNamespace(
        attention_config=SimpleNamespace(indexer_kv_dtype=options["dtype"]),
        parallel_config=SimpleNamespace(decode_context_parallel_size=options["dcp"]),
        speculative_config=options["spec"],
    )
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=options["family"]),
    )
    if error:
        with pytest.raises(ValueError, match=error):
            AscendConfig._validate_indexer_v2_unsharded(config, vc)
    else:
        AscendConfig._validate_indexer_v2_unsharded(config, vc)


@pytest.mark.parametrize(
    "enabled,decode,cooperative,actual,expected",
    [
        (False, True, False, 1, "legacy"),
        (True, True, False, 1, "v2"),
        (True, False, False, 1, "legacy"),
        (True, True, True, 1, "mixed"),
        (True, True, False, 0, "empty"),
    ],
)
def test_unsharded_v2_forward_routing(monkeypatch, enabled, decode, cooperative, actual, expected):
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend._use_v2_unsharded = enabled
    backend.pcp_indexer_merge_backend = "triton_pack"
    backend._use_fused_cache_store = False
    backend.enable_sparse_li_c8, backend.enable_sparse_li_c4 = True, False
    backend.c8_k_cache_dtype = torch.int8
    backend.n_head, backend.head_dim, backend.topk_tokens = 32, 128, 2048
    backend.qk_rope_head_dim, backend.is_rope_neox_style = 0, True
    backend.use_torch_npu_lightning_indexer = True
    backend.wk_weights_proj = Mock()
    backend.wq_b = Mock(return_value=(torch.ones(1, 32 * 128), None))
    backend._quantize_li_tensor = lambda q: (q.to(torch.int8), torch.ones(1, 32, dtype=torch.float16))
    backend.k_cache = SimpleNamespace(
        kv_cache=(torch.zeros(4, 128, 1, 128, dtype=torch.int8), torch.ones(4, 128, 1, 1, dtype=torch.float16))
    )
    backend.forward_k = Mock(return_value=(torch.zeros(1, 1, 128), None, torch.ones(1, 32)))
    backend._gather_cache_inputs = lambda k, scale, metadata: (k, scale, metadata.slot_mapping)
    backend.write_cache = Mock()
    monkeypatch.setattr(indexer_module, "HAS_TRITON", True)
    monkeypatch.setattr(indexer_module, "rope_forward_triton_siso", lambda q, *args, **kwargs: q)
    monkeypatch.setattr(indexer_module, "supports_pcp_indexer_sharding", lambda *args: True)
    outputs = {name: torch.full((1, 1, 2048), i, dtype=torch.int32) for i, name in enumerate(("legacy", "v2", "mixed"))}
    legacy, v2 = Mock(return_value=outputs["legacy"]), Mock(return_value=outputs["v2"])
    prepare = Mock(side_effect=lambda q, w, s, meta: (q, w, s))
    select = Mock(return_value=(outputs["mixed"], torch.ones(1, 1, 2048)))
    finish = Mock(return_value=outputs["mixed"])
    monkeypatch.setattr(indexer_module.DeviceOperator, "indexer_select_post_process", legacy)
    monkeypatch.setattr(indexer_module, "select_unsharded_indexer_v2", v2)
    monkeypatch.setattr(indexer_module, "prepare_pcp_indexer_queries", prepare)
    monkeypatch.setattr(indexer_module, "select_pcp_indexer_candidates", select)
    monkeypatch.setattr(indexer_module, "finish_pcp_indexer_topk", finish)
    metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=actual,
        slot_mapping=torch.tensor([0]),
        seq_lens=torch.tensor([257]),
        cum_query_lens=torch.tensor([1]),
        block_table=torch.tensor([[3, 1, 2]]),
        sin=torch.zeros(1, 1, 1, 128),
        cos=torch.ones(1, 1, 1, 128),
        actual_seq_lengths_query=torch.tensor([1]),
        actual_seq_lengths_key=torch.tensor([257]),
        use_v2_unsharded=decode,
        pcp_indexer=SimpleNamespace(metadata=object()) if cooperative else None,
    )
    hidden = torch.ones(1, 128)
    result = backend.forward(hidden, hidden, hidden, metadata)
    assert result is (None if expected == "empty" else outputs[expected])
    assert legacy.call_count == int(expected == "legacy")
    assert v2.call_count == int(expected == "v2")
    assert prepare.call_count == select.call_count == finish.call_count == int(expected == "mixed")
    if expected == "mixed":
        assert finish.call_args.args[-1] == "triton_pack"
    if expected == "v2":
        assert v2.call_args.args[6] is metadata.actual_seq_lengths_key
        assert v2.call_args.args[7] is metadata.block_table


@pytest.mark.parametrize(
    "enabled,query_len,prefilling,expected",
    [(False, 1, False, False), (True, 1, False, True), (True, 1, True, False), (True, 2, False, False)],
)
def test_unsharded_v2_metadata_only_marks_single_token_decode(monkeypatch, enabled, query_len, prefilling, expected):
    builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    builder.use_pcp, builder.use_dcp = False, False
    builder.pcp_indexer_sharding = False
    builder.indexer_v2_unsharded = enabled
    builder.kernel_block_size, builder.decode_threshold = 128, 1
    builder._mask_lim_slot_mapping = Mock()
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=2 * query_len,
        num_actual_tokens=2 * query_len,
        positions=torch.arange(2 * query_len),
        slot_mapping=torch.arange(2 * query_len),
        block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32),
        seq_lens=torch.tensor([129, 257], dtype=torch.int32),
        max_query_len=query_len,
        is_prefilling=torch.tensor([False, prefilling]),
        context_parallel_metadata=None,
        pcp_context=None,
    )
    monkeypatch.setattr(indexer_module, "get_cos_and_sin_mla", lambda *args, **kwargs: (torch.ones(4), torch.zeros(4)))
    monkeypatch.setattr("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", lambda *args: False)
    metadata = builder._build(common, buffer_key=0, use_cached_rope=True, copy_rope=False)
    assert metadata.use_v2_unsharded is expected
    assert metadata.pcp_indexer is None
