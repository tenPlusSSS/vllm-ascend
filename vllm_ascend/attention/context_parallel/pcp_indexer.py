# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-selective indexing over contiguous logical K blocks on replicated PCP caches."""

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


@dataclass(frozen=True)
class PCPIndexerPlan:
    """Global scheduled batch rows, in slot order, for one model step."""

    requests: tuple[int, ...]
    slots: int
    full_graph: bool


def plan_pcp_indexer_step(batch, minimum: int, capacity: int, *, full_graph: bool) -> PCPIndexerPlan | None:
    """Use only shared CPU batch state. A configured cap prioritizes the longest eligible requests."""
    n = batch.num_reqs
    lengths = batch.num_computed_tokens_np[:n] + 1
    eligible = np.flatnonzero(
        (batch.num_scheduled_tokens[:n] == 1) & ~batch.is_prefilling_np[:n] & (lengths >= minimum)
    )
    if not eligible.size or capacity <= 0:
        return None
    if eligible.size > capacity:
        order = np.argsort(-lengths[eligible].astype(np.int64), kind="stable")
        eligible = np.sort(eligible[order[:capacity]])
    requests = tuple(eligible.tolist())
    return PCPIndexerPlan(requests, capacity if full_graph else len(requests), full_graph)


@dataclass
class PCPIndexerMetadata:
    query_start_loc: torch.Tensor
    key_lens: torch.Tensor
    block_table: torch.Tensor
    query_row: torch.Tensor  # Owner's local Q row, or -1 on a helper/unused slot.
    query_source: torch.Tensor  # TND row -> row in [padded local Q | shared Q].
    shared_row: torch.Tensor  # Slot -> native QLI row that produced this rank's candidates.
    key_start: torch.Tensor
    shard_seq_len: torch.Tensor
    output_source: torch.Tensor  # Local model row -> row in [local results | merged results | padding].
    native_metadata: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)


@dataclass
class PCPIndexerInputs:
    """Stable inputs to the mixed graph. The QLI layout is computed inside the model forward."""

    selection: torch.Tensor  # Global request/token rows, followed by slot count and local token count.
    restored: torch.Tensor
    lengths: torch.Tensor
    selected_table: torch.Tensor
    local_offsets: torch.Tensor
    local_lengths: torch.Tensor
    local_table: torch.Tensor
    tokens: int
    block_size: int

    @cached_property
    def metadata(self) -> PCPIndexerMetadata:
        return _build_native_layout(self)


def build_pcp_indexer_metadata(
    common, context, cache_group: int, block_size: int, graph_buffers: dict
) -> PCPIndexerInputs | None:
    plan = context.indexer_plan
    if plan is None:
        return None

    slots = plan.slots
    batch = context.global_batch
    count = len(plan.requests)
    # New host storage for each nonblocking transfer, including back-to-back steps.
    selection = np.zeros((2, slots + 1), dtype=np.int64)
    selection[0, :count] = plan.requests
    selection[1, :count] = batch.query_start_loc_np[selection[0, :count]]
    selection[:, -1] = (count, common.num_actual_tokens)

    n, tokens = common.num_reqs, common.num_input_tokens
    local_table = common.block_table_tensor[:n]
    global_table = context.global_block_tables[cache_group]
    key = (cache_group, tokens, n, slots)
    buffers = graph_buffers.get(key) if plan.full_graph else None
    if buffers is None:
        buffers = PCPIndexerInputs(
            selection=torch.empty(selection.shape, dtype=torch.long, device=common.seq_lens.device),
            restored=context.hidden_restore_idx.new_empty((slots,)),
            lengths=batch.seq_lens.new_empty((slots,)),
            selected_table=global_table.new_empty((slots, global_table.shape[1])),
            local_offsets=torch.empty_like(common.query_start_loc[: n + 1]),
            local_lengths=torch.empty_like(common.seq_lens[:n]),
            local_table=torch.empty_like(local_table),
            tokens=tokens,
            block_size=block_size,
        )
        if plan.full_graph:
            graph_buffers[key] = buffers
    elif buffers.selected_table.shape[1] != global_table.shape[1] or buffers.local_table.shape != local_table.shape:
        raise RuntimeError("PCP indexer block-table shape changed within a captured bucket")
    buffers.selection.copy_(torch.from_numpy(selection), non_blocking=True)
    request_rows, token_rows = buffers.selection[:, :slots].unbind()
    torch.index_select(context.hidden_restore_idx, 0, token_rows, out=buffers.restored)
    torch.index_select(batch.seq_lens, 0, request_rows, out=buffers.lengths)
    torch.index_select(global_table, 0, request_rows, out=buffers.selected_table)
    buffers.local_offsets.copy_(common.query_start_loc[: n + 1])
    buffers.local_lengths.copy_(common.seq_lens[:n])
    buffers.local_table.copy_(local_table)
    # Share captured addresses, but create fresh per-forward caches. This records
    # both layout and QLI metadata producers during capture, even after warmup.
    return replace(buffers)


def _build_native_layout(inputs: PCPIndexerInputs) -> PCPIndexerMetadata:
    group = get_pcp_group()
    device = inputs.selection.device
    slot_count = inputs.restored.numel()
    slots = torch.arange(slot_count, dtype=torch.long, device=device)
    count, actual = inputs.selection[:, -1].unbind()
    active = slots < count
    restored = inputs.restored.long()
    tokens, block_size = inputs.tokens, inputs.block_size
    owner_rank = restored // max(tokens, 1)
    local_row = restored - group.rank_in_group * tokens
    owns = active & (owner_rank == group.rank_in_group) & (local_row >= 0) & (local_row < actual)
    query_row = torch.where(owns, local_row, -1)

    lengths = torch.where(active, inputs.lengths, 0)
    blocks = (lengths + block_size - 1) // block_size
    first = blocks * group.rank_in_group // group.world_size
    end = blocks * (group.rank_in_group + 1) // group.world_size
    key_start = first * block_size
    shard_len = (torch.minimum(lengths, end * block_size) - key_start).clamp_min(0)
    local_table = inputs.local_table
    full_table = inputs.selected_table
    columns = first[:, None] + torch.arange(local_table.shape[1], device=device)
    shard_table = torch.where(
        columns < end[:, None], full_table.gather(1, columns.clamp(max=full_table.shape[1] - 1)), 0
    )
    # Native QLI can require a placeholder key even when this rank owns no visible blocks.
    native_len = shard_len.clamp_min(1)
    ends = torch.minimum(inputs.local_offsets[1:], actual.to(inputs.local_offsets.dtype))
    starts = inputs.local_offsets[:-1]
    owner_request = owns[:, None] & (starts <= query_row[:, None]) & (query_row[:, None] < ends)
    req_slot = torch.where(owner_request, slots[:, None], -1).amax(dim=0)
    safe_slot = req_slot.clamp_min(0)
    local_table = torch.where(req_slot[:, None] >= 0, shard_table.index_select(0, safe_slot), local_table)
    local_lengths = torch.where(req_slot >= 0, native_len.index_select(0, safe_slot), inputs.local_lengths)

    # Keep helpers adjacent to *actual* local tokens. Model-input padding lives at the
    # end of the native TND storage, even when the owner's extra request has zero Q length.
    helper = active & ~owns
    helper_ends = actual + helper.cumsum(0)
    shared_row = torch.where(owns, query_row, torch.where(helper, helper_ends - 1, 0))
    rows = torch.arange(tokens + slot_count, device=device)
    helper_slot = torch.searchsorted(helper_ends, rows, right=True).clamp(max=slot_count - 1)
    query_source = torch.where(rows < actual, rows, tokens + helper_slot)
    local_rows = rows[:tokens]
    out_slot = torch.where(owns[:, None] & (query_row[:, None] == local_rows), slots[:, None], -1).amax(dim=0)
    output_source = torch.where(out_slot >= 0, tokens + out_slot, local_rows)
    output_source = torch.where(local_rows < actual, output_source, tokens + slot_count)
    query_ends = torch.cat((ends, helper_ends.to(ends.dtype)))
    return PCPIndexerMetadata(
        query_start_loc=torch.nn.functional.pad(query_ends, (1, 0)),
        key_lens=torch.cat((local_lengths, native_len)),
        block_table=torch.cat((local_table, shard_table)),
        query_row=query_row,
        query_source=query_source,
        shared_row=shared_row,
        key_start=key_start,
        shard_seq_len=shard_len,
        output_source=output_source,
    )


def supports_pcp_indexer_sharding(dtype: torch.dtype, heads: int, head_dim: int, topk: int) -> bool:
    """PR #20878's arch22 score output accepts INT8 Q/K and FP16 scales.

    Preserve floating-point fallback layers instead of quantizing their cache
    merely to enter the cooperative path.
    """
    return (
        get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.STANDARD
        and dtype == torch.int8
        and heads in (32, 64)
        and head_dim == 128
        and 0 < topk <= 2048
    )


def select_pcp_indexer_candidates(
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    metadata: PCPIndexerMetadata,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One PR #20878 QLI V2 call returns both local indices and FP32 scores.

    PA_BBND is V2's spelling for the existing [blocks, block_size, N, D]
    cache. PCP partitions each selected request's table and length.
    Preserve the native score scale (arch22 uses QK / 1024) on every rank.
    """
    heads, dim = query.shape[1:]
    shape_key = (heads, dim, topk)
    if shape_key not in metadata.native_metadata:
        metadata.native_metadata[shape_key] = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            heads,
            1,
            dim,
            topk,
            2,  # V2 QUANT_MODE_INT8 (the old torch_npu API used 0).
            cu_seqlens_q=metadata.query_start_loc,
            seqused_k=metadata.key_lens,
            batch_size=metadata.key_lens.shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query.contiguous(),
        key_cache,
        weights.to(torch.float16).contiguous(),
        query_scale.reshape(query.shape[:-1]).to(torch.float16).contiguous(),
        key_scale.reshape(key_cache.shape[:2] + (1,)),
        topk,
        2,
        cu_seqlens_q=metadata.query_start_loc,
        seqused_k=metadata.key_lens,
        block_table=metadata.block_table,
        metadata=metadata.native_metadata[shape_key],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=1,
        enable_dcp=False,
    )


def select_unsharded_indexer_v2(
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    query_ends: torch.Tensor,
    key_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Scan full owner-local requests with QLI V2, returning indices only.

    Build metadata on every forward so its producer is captured along with
    the scan, including after graph warmup. Replay then consumes current
    device lengths instead of a metadata tensor frozen during warmup.
    No PCP query exchange, partitioning or candidate collective is needed.
    """
    heads, dim = query.shape[1:]
    quant_mode = 2  # QLI V2 INT8 mode, matching the cooperative path.
    query_start_loc = torch.cat((query_ends.new_zeros(1), query_ends))
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        heads,
        1,
        dim,
        topk,
        quant_mode,
        cu_seqlens_q=query_start_loc,
        seqused_k=key_lens,
        batch_size=key_lens.shape[0],
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
    )
    indices, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query.contiguous(),
        key_cache,
        weights.to(torch.float16).contiguous(),
        query_scale.reshape(query.shape[:-1]).to(torch.float16).contiguous(),
        key_scale.reshape(key_cache.shape[:2] + (1,)),
        topk,
        quant_mode,
        cu_seqlens_q=query_start_loc,
        seqused_k=key_lens,
        block_table=block_table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=0,
        enable_dcp=False,
    )
    return indices


def prepare_pcp_indexer_queries(
    query: torch.Tensor, weights: torch.Tensor, query_scale: torch.Tensor, metadata: PCPIndexerMetadata
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange all selected C8 queries in one fixed-size, root-independent collective."""
    slots = metadata.query_row.numel()
    tokens = metadata.output_source.numel()
    heads, dim = query.shape[1:]
    q_bytes, weight_bytes = heads * dim, 2 * heads
    scales = query_scale.reshape(-1, heads).to(torch.float16)
    weights = weights.to(torch.float16)
    if query.shape[0]:
        row = metadata.query_row.clamp_min(0)
        payload = torch.cat(
            (
                query.index_select(0, row).reshape(slots, q_bytes),
                weights.index_select(0, row).contiguous().view(torch.int8),
                scales.index_select(0, row).contiguous().view(torch.int8),
            ),
            dim=1,
        )
        payload = torch.where(metadata.query_row[:, None] >= 0, payload, 0)
    else:
        payload = query.new_zeros((slots, q_bytes + 2 * weight_bytes))
    # Each active slot has exactly one owner. All other ranks contribute zero bytes,
    # so INT8 SUM preserves the owner's Q and both FP16 fields bit for bit.
    shared = get_pcp_group().all_reduce(payload)
    shared_q, shared_w, shared_s = shared.split((q_bytes, weight_bytes, weight_bytes), dim=1)

    def assemble(local: torch.Tensor, remote: torch.Tensor) -> torch.Tensor:
        # The source map uses the padded model extent as the base of the shared rows.
        if local.shape[0] < tokens:
            local = torch.cat((local, local.new_zeros((tokens - local.shape[0], *local.shape[1:]))))
        return torch.cat((local[:tokens], remote)).index_select(0, metadata.query_source)

    return (
        assemble(query, shared_q.reshape(slots, heads, dim)),
        assemble(weights, shared_w.contiguous().view(torch.float16)),
        assemble(scales, shared_s.contiguous().view(torch.float16)),
    )


def pack_topk_candidates(indices, scores, key_start, shard_len):
    """Produce [slot, K, (global ID, FP32 score bits)] in local ID order."""
    valid = (indices >= 0) & (indices < shard_len[:, None]) & ~torch.isnan(scores) & (scores > -float("inf"))
    ids = torch.where(valid, indices + key_start[:, None], -1).to(torch.int32)
    ids, order = torch.sort(ids, dim=-1, stable=True)
    values = scores.masked_fill(~valid, -float("inf")).gather(1, order).contiguous()
    return torch.stack((ids, values.view(torch.int32)), dim=-1)


def finish_pcp_indexer_topk(
    topk_indices, topk_scores, metadata: PCPIndexerMetadata, output_tokens: int, backend: str = "torch"
):
    """Merge independent slots across ranks, then restore only their owners' rows."""
    group = get_pcp_group()
    slots, k = metadata.shared_row.numel(), topk_indices.shape[-1]
    use_triton = backend != "torch" and k == 2048
    if use_triton:
        # Keep the default backend usable on installations without Triton.
        from vllm_ascend.ops.triton.pcp_indexer_candidates import pack_sorted_candidates

        pairs = pack_sorted_candidates(
            topk_indices, topk_scores, metadata.shared_row, metadata.key_start, metadata.shard_seq_len
        )
    else:
        indices = topk_indices.index_select(0, metadata.shared_row).squeeze(1)
        scores = topk_scores.index_select(0, metadata.shared_row).squeeze(1).float()
        pairs = pack_topk_candidates(indices, scores, metadata.key_start, metadata.shard_seq_len)
    gathered = group.all_gather(pairs, dim=0)
    if use_triton and backend == "triton_merge":
        from vllm_ascend.ops.triton.pcp_indexer_candidates import merge_sorted_candidates

        return merge_sorted_candidates(
            gathered, topk_indices, metadata.output_source, output_tokens, slots, group.world_size
        )
    return merge_candidates_torch(
        gathered, topk_indices, metadata.output_source, output_tokens, slots, group.world_size
    )


def merge_candidates_torch(gathered, topk_indices, output_source, output_tokens, slots, world):
    k = topk_indices.shape[-1]
    gathered = gathered.reshape(world, slots, k, 2).transpose(0, 1).reshape(slots, world * k, 2)
    # Both packers put equal-score candidates in ascending ID order within
    # each shard. Contiguous shards arrive in rank order.
    scores = gathered[:, :, 1].contiguous().view(torch.float32)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    merged = gathered[:, :, 0].gather(1, order).unsqueeze(1)
    tokens = output_source.numel()
    padding = merged.new_full((1, 1, k), -1)
    combined = torch.cat((topk_indices[:tokens], merged, padding))
    return combined.index_select(0, output_source[:output_tokens])
