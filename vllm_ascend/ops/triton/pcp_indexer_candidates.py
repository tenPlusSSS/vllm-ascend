# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP adaptation of the upstream sharded indexer's radix pack and bitonic Top-K merge."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import _resolve_triton_ascend_op

_sort = _resolve_triton_ascend_op("sort")
_LOG_K = 11  # The experimental kernels specialize to K=2048.
_RANKS_PER_GROUP = 4


@triton.jit
def _score_priority(bits):
    # Signed INT32 comparison gives IEEE score order. +/-0 share a key.
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _permute(words, positions):
    return tl.gather(words.to(tl.float32, bitcast=True), positions, 0).to(tl.int32, bitcast=True)


@triton.jit
def _pack_kernel(
    indices,
    scores,
    shared_row,
    key_start,
    shard_len,
    packet,
    ID_ROW: tl.constexpr,
    ID_COL: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    SCORE_COL: tl.constexpr,
    LOG_K: tl.constexpr,
):
    slot = tl.program_id(0)
    k: tl.constexpr = 1 << LOG_K
    lane = tl.arange(0, k)
    row = tl.load(shared_row + slot)
    ids = tl.load(indices + row * ID_ROW + lane * ID_COL)
    bits = tl.load(scores + row * SCORE_ROW + lane * SCORE_COL).to(tl.int32, bitcast=True)
    # Match torch pack exactly: +inf is valid, NaN and -inf are invalid.
    valid = (ids >= 0) & (ids < tl.load(shard_len + slot)) & ((bits & 0x7FFFFFFF) <= 0x7F800000) & (bits != -8388608)
    ids = tl.where(valid, ids, 0)
    bits = tl.where(valid, bits, -8388608)  # FP32 -infinity bit pattern

    # Stable LSD radix on [descending FP32 score | ascending 31-bit local ID].
    # Sorting (digit, current lane) as one exactly representable FP32 integer
    # recovers the permutation even with a value-only Ascend sort instruction.
    digit_bits: tl.constexpr = 24 - LOG_K
    for shift in tl.static_range(0, 63, digit_bits):
        descending = (_score_priority(bits) ^ 0x7FFFFFFF).to(tl.uint32)
        if shift >= 31:
            digit = descending >> (shift - 31)
        elif shift + digit_bits <= 31:
            digit = ids.to(tl.uint32) >> shift
        else:
            digit = (ids.to(tl.uint32) >> shift) | (descending << (31 - shift))
        digit = (digit & ((1 << digit_bits) - 1)).to(tl.int32)
        tagged = ((digit << LOG_K) + lane).to(tl.float32)
        order = _sort(tagged, descending=False).to(tl.int32) & (k - 1)
        ids = _permute(ids, order)
        bits = _permute(bits, order)
    global_ids = tl.where(bits != -8388608, ids + tl.load(key_start + slot).to(tl.int32), -1)
    pair = tl.reshape(tl.join(global_ids, bits), (2 * k,))
    tl.store(packet + slot * (2 * k) + tl.arange(0, 2 * k), pair)


def pack_sorted_candidates(indices, scores, shared_row, key_start, shard_len):
    """Read native QLI rows by slot and produce [slots, K, (global ID, FP32 bits)]."""
    slots = shared_row.numel()
    k = indices.shape[-1]
    if k != 1 << _LOG_K or indices.dtype != torch.int32 or scores.dtype != torch.float32:
        raise ValueError("PCP Triton pack requires K=2048, INT32 indices and FP32 scores")
    packet = indices.new_empty((slots, k, 2))
    if slots:
        _pack_kernel[(slots,)](
            indices,
            scores,
            shared_row,
            key_start,
            shard_len,
            packet,
            indices.stride(0),
            indices.stride(-1),
            scores.stride(0),
            scores.stride(-1),
            _LOG_K,
            multibuffer=False,
        )
    return packet


@triton.jit
def _better(priority, ids, other_priority, other_ids):
    return (priority > other_priority) | ((priority == other_priority) & (ids < other_ids))


@triton.jit
def _merge_lists(packet, first, end, slot, SLOTS: tl.constexpr, LOG_K: tl.constexpr, RAW_BITS: tl.constexpr):
    # Each rank/group supplies an already sorted list. A reversed incoming
    # list forms a bitonic sequence with the accumulator. Discard the losing
    # half, then finish only the log2(K) compare/exchange steps on the winners.
    k: tl.constexpr = 1 << LOG_K
    lane = tl.arange(0, k)
    offsets = slot * (2 * k) + tl.arange(0, 2 * k)
    words = tl.load(packet + first * SLOTS * (2 * k) + offsets)
    ids, priority = tl.split(tl.reshape(words, (k, 2)))
    if RAW_BITS:
        priority = _score_priority(priority)
    for rank in range(first + 1, end):
        words = tl.load(packet + rank * SLOTS * (2 * k) + offsets)
        incoming_ids, incoming = tl.split(tl.reshape(words, (k, 2)))
        if RAW_BITS:
            incoming = _score_priority(incoming)
        other_ids = _permute(incoming_ids, k - 1 - lane)
        other = _permute(incoming, k - 1 - lane)
        take = _better(other, other_ids, priority, ids)
        priority = tl.where(take, other, priority)
        ids = tl.where(take, other_ids, ids)
        for step in range(LOG_K):
            stride = 1 << (LOG_K - 1 - step)
            other = _permute(priority, lane ^ stride)
            other_ids = _permute(ids, lane ^ stride)
            take = _better(other, other_ids, priority, ids) == ((lane & stride) == 0)
            priority = tl.where(take, other, priority)
            ids = tl.where(take, other_ids, ids)
    return ids, priority


@triton.jit
def _merge_groups(packet, partial, SLOTS: tl.constexpr, WORLD: tl.constexpr, GROUP: tl.constexpr, LOG_K: tl.constexpr):
    slot = tl.program_id(0)
    group = tl.program_id(1)
    first = group * GROUP
    ids, priority = _merge_lists(packet, first, tl.minimum(first + GROUP, WORLD), slot, SLOTS, LOG_K, True)
    width: tl.constexpr = 2 << LOG_K
    tl.store(
        partial + (group * SLOTS + slot) * width + tl.arange(0, width), tl.reshape(tl.join(ids, priority), (width,))
    )


@triton.jit
def _merge_to_output(
    packet,
    native,
    sources,
    output,
    TOKENS: tl.constexpr,
    SLOTS: tl.constexpr,
    LISTS: tl.constexpr,
    NATIVE_ROW: tl.constexpr,
    NATIVE_COL: tl.constexpr,
    RAW_BITS: tl.constexpr,
    LOG_K: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(sources + row)
    k: tl.constexpr = 1 << LOG_K
    lane = tl.arange(0, k)
    if source < TOKENS:
        ids = tl.load(native + source * NATIVE_ROW + lane * NATIVE_COL)
    elif source < TOKENS + SLOTS:
        ids, _ = _merge_lists(packet, 0, LISTS, source - TOKENS, SLOTS, LOG_K, RAW_BITS)
    else:
        ids = tl.full((k,), -1, tl.int32)
    tl.store(output + row * k + lane, ids)


def merge_sorted_candidates(packet, native_indices, output_source, output_tokens, slots, world):
    """Return full global winners for selected requests, local rows or -1 for all others."""
    k = 1 << _LOG_K
    groups = triton.cdiv(world, _RANKS_PER_GROUP)
    output = native_indices.new_empty((min(output_tokens, output_source.numel()), 1, k))
    if not output.shape[0]:
        return output
    if groups > 1:
        partial = packet.new_empty((groups, slots, k, 2))
        _merge_groups[(slots, groups)](packet, partial, slots, world, _RANKS_PER_GROUP, _LOG_K, multibuffer=False)
        packet = partial
    _merge_to_output[(output.shape[0],)](
        packet,
        native_indices,
        output_source,
        output,
        output_source.numel(),
        slots,
        groups if groups > 1 else world,
        native_indices.stride(0),
        native_indices.stride(-1),
        groups == 1,
        _LOG_K,
        multibuffer=False,
    )
    return output
