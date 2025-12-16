import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
import yunchang.comm.extract_local
import yunchang.ring.utils
from flash_attn import flash_attn_func
from xfuser.core.distributed import (
    get_ring_parallel_rank,
    get_ring_parallel_world_size,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP
from yunchang.kernels import AttnType
from yunchang.ring.utils import RingComm, update_out_and_lse


@torch.jit.script
def _update_out_and_lse_inf_robust(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)

    # old
    # lse = lse - F.logsigmoid(lse - block_lse)  # <- (-inf) - (-inf) = NaN

    # new
    max_lse = torch.maximum(lse, block_lse)
    lse = max_lse + torch.log(1 + torch.exp(-torch.abs(lse - block_lse)))

    return out, lse


yunchang.ring.utils._update_out_and_lse = _update_out_and_lse_inf_robust


_RING_COMM_STREAM = None
_WARMUPED = False


def _get_ring_comm_stream():
    global _RING_COMM_STREAM

    if _RING_COMM_STREAM is None:
        _RING_COMM_STREAM = torch.cuda.Stream()

    return _RING_COMM_STREAM


def zigzag_extract_local_patched(value, rank, world_size, rd=None, ud=None, dim=1, *args, **kwargs):
    """
    value is a tensor of shape (bs, seqlen, ...)
    """

    rd = get_ring_parallel_world_size() if rd is None else rd
    ud = get_ulysses_parallel_world_size() if ud is None else ud

    input_dim = value.dim()
    assert input_dim >= 2

    shape = list(value.shape)
    seqlen = shape[dim]

    value_chunks = value.chunk(2 * rd, dim=dim)

    r_rank = get_ring_parallel_rank()
    u_rank = get_ulysses_parallel_rank()

    assert get_ring_parallel_world_size() == rd, (
        f"Ring parallel world size mismatch {get_ring_parallel_world_size()} != {rd}"
    )
    assert get_ulysses_parallel_world_size() == ud, (
        f"Ulysses parallel world size mismatch {get_ulysses_parallel_world_size()} != {ud}"
    )

    local_value = torch.cat([value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]], dim=dim).chunk(ud, dim=dim)[
        u_rank
    ]

    new_shape = shape
    new_shape[dim] = seqlen // world_size
    return local_value.reshape(new_shape).contiguous()


def all_gather_zigzag(local_tensor, rd=None, ud=None, dim=1, *args, **kwargs):
    """
    Inverse of zigzag_extract_local_patched (All-Gather with Reordering).
    """

    rd = get_ring_parallel_world_size() if rd is None else rd
    ud = get_ulysses_parallel_world_size() if ud is None else ud

    ring_pg = PROCESS_GROUP.RING_PG
    ulysses_pg = PROCESS_GROUP.ULYSSES_PG

    # --- 1. ulysses All-Gather ---

    # (B, H, S_local, D) -> (S_local, H, B, D)
    local_tensor_trans = local_tensor.transpose(0, dim).contiguous()

    # out shape (S_local*ud, H, B, D)
    shape_trans = list(local_tensor_trans.shape)
    shape_trans[0] *= ud
    concatenated_chunk_trans = torch.empty(shape_trans, dtype=local_tensor.dtype, device=local_tensor.device)

    dist.all_gather_into_tensor(concatenated_chunk_trans, local_tensor_trans, group=ulysses_pg)

    # (S_local*ud, H, B, D) -> (B, H, S_local*ud, D)
    concatenated_chunk = concatenated_chunk_trans.transpose(0, dim).contiguous()

    # --- 2. Chunk Split ---
    chunk_O_r, chunk_O_other = concatenated_chunk.chunk(2, dim=dim)

    chunk_O_r = chunk_O_r.contiguous()
    chunk_O_other = chunk_O_other.contiguous()

    # --- 3. Ring All-Gather ---

    # first half chunks
    chunk_O_r_trans = chunk_O_r.transpose(0, dim).contiguous()
    shape_r_trans = list(chunk_O_r_trans.shape)
    shape_r_trans[0] *= rd
    gathered_O_r_trans = torch.empty(shape_r_trans, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(gathered_O_r_trans, chunk_O_r_trans, group=ring_pg)
    gathered_O_r = gathered_O_r_trans.transpose(0, dim).contiguous()

    # second half chunks
    chunk_O_other_trans = chunk_O_other.transpose(0, dim).contiguous()
    shape_other_trans = list(chunk_O_other_trans.shape)
    shape_other_trans[0] *= rd
    gathered_O_other_trans = torch.empty(shape_other_trans, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(gathered_O_other_trans, chunk_O_other_trans, group=ring_pg)
    gathered_O_other = gathered_O_other_trans.transpose(0, dim).contiguous()

    # --- 4. Final Assembly ---
    other_chunks_list = gathered_O_other.chunk(rd, dim=dim)
    gathered_O_other_reversed = torch.cat(list(reversed(other_chunks_list)), dim=dim)

    global_tensor = torch.cat([gathered_O_r, gathered_O_other_reversed], dim=dim)

    return global_tensor.contiguous()


yunchang.comm.extract_local.EXTRACT_FUNC_DICT["zigzag"] = zigzag_extract_local_patched


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def torch_attention_with_sinks_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_query_heads, head_size = query.shape
    _, _, num_kv_heads, _ = key.shape

    query_transposed = query.transpose(1, 2)
    key_transposed = key.transpose(1, 2)
    value_transposed = value.transpose(1, 2)

    num_queries_per_kv = num_query_heads // num_kv_heads
    key_transposed = repeat_kv(key_transposed, num_queries_per_kv)
    value_transposed = repeat_kv(value_transposed, num_queries_per_kv)

    attn_weights = torch.matmul(query_transposed, key_transposed.transpose(-2, -1)) * scale

    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        attn_weights.masked_fill_(causal_mask[None, None, :, :], float("-inf"))

    q_indices = torch.arange(seq_len, device=query.device)[:, None]
    k_indices = torch.arange(seq_len, device=query.device)[None, :]

    if window_size[0] != -1:
        past_mask = (q_indices - k_indices) >= window_size[0]
        attn_weights.masked_fill_(past_mask[None, None, :, :], float("-inf"))

    if window_size[1] != -1:
        future_mask = (k_indices - q_indices) > window_size[1]
        attn_weights.masked_fill_(future_mask[None, None, :, :], float("-inf"))

    row_max_val, _ = torch.max(attn_weights, dim=-1)  # shape: (B, Nq, S)

    # If max is -inf, it means the entire row is masked.
    is_inf_row = torch.isinf(row_max_val)  # shape: (B, Nq, S)

    all_masked = False
    if torch.all(is_inf_row).item():
        all_masked = True
        return None, None, all_masked

    expanded_sinks = sinks.view(1, -1, 1, 1).expand(batch_size, -1, seq_len, 1)
    combined_logits = torch.cat([attn_weights, expanded_sinks], dim=-1)

    m, _ = torch.max(combined_logits, dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), 0.0, m)

    p = torch.exp(combined_logits - m)
    l = torch.sum(p, dim=-1)
    lse = m.squeeze(-1) + torch.log(l + 1e-9)

    probs = p / (l.unsqueeze(-1) + 1e-9)
    scores = probs[..., :-1]
    scores = scores.to(query.dtype)

    output_transposed = torch.matmul(scores, value_transposed)
    output = output_transposed.transpose(1, 2).contiguous()

    final_lse = torch.where(is_inf_row, -torch.inf, lse)

    mask_for_output = is_inf_row.transpose(1, 2).unsqueeze(-1)
    final_output = torch.where(mask_for_output, 0.0, output)

    return final_output, final_lse, all_masked


configs = []
for block_q in [8, 16, 32]:
    for block_n in [32, 64, 128]:
        for num_warps in [4, 8]:
            configs.append(triton.Config({"BLOCK_Q_PER_HEAD": block_q, "BLOCK_N": block_n}, num_warps=num_warps))


@triton.autotune(
    configs=configs,
    key=["q_seq_len", "kv_seq_len", "IS_CAUSAL", "WINDOW_SIZE_PAST"],
)
@triton.jit
def kernel_attention_contiguous_vllm_ported(
    output_ptr,
    lse_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    sinks_ptr,
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [q_seq_len, q_seq_len]
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_lse_batch,
    stride_lse_head,
    stride_lse_seq,
    stride_q_batch,
    stride_q_seq,
    stride_q_head,
    stride_k_batch,
    stride_k_seq,
    stride_k_head,
    stride_v_batch,
    stride_v_seq,
    stride_v_head,
    qq_bias_stride_0,  # int (qq_bias.stride(0))
    q_seq_len,
    kv_seq_len,
    scale,
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    BLOCK_Q_PER_HEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW_SIZE_PAST: tl.constexpr,
    WINDOW_SIZE_FUTURE: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
):
    BLOCK_M_FUSED: tl.constexpr = BLOCK_Q_PER_HEAD * NUM_QUERIES_PER_KV

    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    start_m_q_pos = tl.program_id(2) * BLOCK_Q_PER_HEAD
    offs_m_fused = tl.arange(0, BLOCK_M_FUSED)
    offs_q_pos = start_m_q_pos + (offs_m_fused // NUM_QUERIES_PER_KV)
    offs_q_head_offset = offs_m_fused % NUM_QUERIES_PER_KV
    head_idx = kv_head_idx * NUM_QUERIES_PER_KV + offs_q_head_offset

    k_ptrs_base = key_ptr + (batch_idx * stride_k_batch + kv_head_idx * stride_k_head)
    v_ptrs_base = value_ptr + (batch_idx * stride_v_batch + kv_head_idx * stride_v_head)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    d_mask = offs_d < HEAD_SIZE
    d_mask_q_v = d_mask[None, :]
    d_mask_k = d_mask[:, None]
    q_mask = offs_q_pos < q_seq_len

    q_ptrs = (
        query_ptr
        + batch_idx * stride_q_batch
        + offs_q_pos[:, None] * stride_q_seq
        + head_idx[:, None] * stride_q_head
        + offs_d[None, :]
    )

    acc = tl.zeros([BLOCK_M_FUSED, HEAD_SIZE_PADDED], dtype=tl.float32)
    m_i = tl.load(sinks_ptr + head_idx, mask=q_mask, other=float("-inf")).to(tl.float32)
    l_i = tl.full([BLOCK_M_FUSED], 1.0, dtype=tl.float32)

    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask_q_v, other=0.0)
    q = (q * scale).to(q.dtype)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + head_idx, mask=q_mask, other=0.0)

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + offs_q_pos[:, None] * qq_bias_stride_0

    end_n = kv_seq_len
    start_n = 0
    while start_n < end_n:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n[None, :] < kv_seq_len

        k_ptrs = k_ptrs_base + (offs_d[:, None] * 1 + offs_n[None, :] * stride_k_seq)
        k_load = tl.load(k_ptrs, mask=k_mask & d_mask_k, other=0.0)

        k = k_load

        qk = tl.zeros([BLOCK_M_FUSED, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if USE_ALIBI_SLOPES:
            alibi_bias = alibi_slope[:, None] * offs_n[None, :]
            qk += alibi_bias

        if USE_QQ_BIAS:
            key_pos = offs_n[None, :]
            qq_bias_mask = (key_pos >= 0) & (key_pos < q_seq_len)
            qq_bias = tl.load(qq_bias_row_ptrs + key_pos, mask=qq_bias_mask & q_mask[:, None], other=0.0)
            qk += qq_bias

        mask = q_mask[:, None] & (offs_n[None, :] < kv_seq_len)
        if IS_CAUSAL:
            mask = mask & (offs_q_pos[:, None] >= offs_n[None, :])
        if WINDOW_SIZE_PAST != -1:
            past_mask = (offs_q_pos[:, None] - offs_n[None, :]) >= WINDOW_SIZE_PAST
            qk = tl.where(past_mask, float("-inf"), qk)
        if WINDOW_SIZE_FUTURE != -1:
            future_mask = (offs_n[None, :] - offs_q_pos[None, :]) > WINDOW_SIZE_FUTURE
            qk = tl.where(future_mask, float("-inf"), qk)

        qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_j = tl.sum(p, 1)
        m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptrs_base + (offs_n[:, None] * stride_v_seq + offs_d[None, :])
        v_mask = offs_n[:, None] < kv_seq_len
        v_load = tl.load(v_ptrs, mask=v_mask & d_mask_q_v, other=0.0)
        v = v_load

        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_j
        m_i = m_ij

        start_n += BLOCK_N

    lse = m_i + tl.log(l_i)
    lse_ptrs = lse_ptr + (batch_idx * stride_lse_batch + head_idx * stride_lse_head + offs_q_pos * stride_lse_seq)
    tl.store(lse_ptrs, lse, mask=q_mask)

    acc = acc / l_i[:, None]

    out_ptrs = output_ptr + (
        batch_idx * stride_out_batch
        + offs_q_pos[:, None] * stride_out_seq
        + head_idx[:, None] * stride_out_head
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & d_mask_q_v)


def triton_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_size: tuple,
    alibi_slopes: torch.Tensor = None,
    qq_bias: torch.Tensor = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out_scale: float = 1.0,
):
    """b, s, nh, hd = query.shape
    out = torch.empty_like(query)
    lse = torch.empty((b, nh, s), dtype=torch.float32, device=query.device)
    return out, lse"""

    """out, lse, _ = flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        return_attn_probs=True)
    return out, lse"""

    USE_ALIBI_SLOPES = alibi_slopes is not None
    USE_QQ_BIAS = qq_bias is not None

    batch_size, q_seq_len, num_query_heads, head_size = query.shape
    _, kv_seq_len, num_kv_heads, _ = key.shape

    assert sinks.numel() == num_query_heads

    output = torch.empty_like(query)
    lse_output = torch.empty((batch_size, num_query_heads, q_seq_len), dtype=torch.float32, device=query.device)

    num_queries_per_kv = num_query_heads // num_kv_heads

    def grid(meta):
        return (
            batch_size,
            num_kv_heads,
            triton.cdiv(q_seq_len, meta["BLOCK_Q_PER_HEAD"]),
        )

    PADDED_HEAD_SIZE = triton.next_power_of_2(head_size)

    if alibi_slopes is None:
        alibi_slopes = torch.empty(0, dtype=query.dtype, device=query.device)
    if qq_bias is None:
        qq_bias = torch.empty(0, dtype=query.dtype, device=query.device)

    kernel_attention_contiguous_vllm_ported[grid](
        output,
        lse_output,
        query,
        key,
        value,
        sinks,
        alibi_slopes,
        qq_bias,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse_output.stride(0),
        lse_output.stride(1),
        lse_output.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        qq_bias.stride(0) if USE_QQ_BIAS else 0,
        q_seq_len,
        kv_seq_len,
        scale,
        k_scale,
        v_scale,
        out_scale,
        NUM_QUERIES_PER_KV=num_queries_per_kv,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=PADDED_HEAD_SIZE,
        IS_CAUSAL=is_causal,
        WINDOW_SIZE_PAST=window_size[0],
        WINDOW_SIZE_FUTURE=window_size[1],
        USE_ALIBI_SLOPES=USE_ALIBI_SLOPES,
        USE_QQ_BIAS=USE_QQ_BIAS,
    )

    return output, lse_output


def call_block_attn(
    query,
    key,
    value,
    sinks,
    softmax_scale,
    causal,
    window_size,
):
    # Padding for MLA where qk_head_dim != v_head_dim
    # q, k: [bs, seqlen, num_heads, qk_head_dim]
    # v: [bs, seqlen, num_heads, v_head_dim]
    original_v_head_dim = value.shape[-1]
    if value.shape[-1] != query.shape[-1]:
        pad_len = query.shape[-1] - value.shape[-1]
        value = F.pad(value, (0, pad_len))

    attn = "flash"

    if window_size != (-1, -1):
        attn = "triton"

    if attn == "flash":
        out, lse, _ = flash_attn_func(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )
    elif attn == "triton":
        out, lse = triton_attention_forward(
            query,
            key,
            value,
            scale=softmax_scale,
            is_causal=causal,
            window_size=window_size,
        )
    else:
        raise ValueError(f"Unsupported attention type: {attn}")

    if out.shape[-1] != original_v_head_dim:
        out = out[..., :original_v_head_dim]

    return out, lse


def moreh_gpt_attention(
    module,
    query,
    key,
    value,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH

    ulysses_size = dist.get_world_size(module.ulysses_pg)
    ring_size = dist.get_world_size(module.ring_pg)

    comm = RingComm(module.ring_pg)

    global _WARMUPED

    if not _WARMUPED:
        tensor = torch.empty_like(key)
        received_tensor: torch.Tensor = comm.send_recv(tensor)
        comm_stream = _get_ring_comm_stream()
        with torch.cuda.stream(comm_stream):
            comm.commit()
        comm.wait()
        received_tensor += 1.0
        _WARMUPED = True

    query_layer = query
    key_layer = key
    value_layer = value

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))
    comm = RingComm(module.ring_pg)
    assert comm.world_size <= 16, "Ring Attention only supports up to 16 devices."

    out = None
    lse = None

    next_k, next_v = None, None

    original_window_size = window_size

    chunk_len = query_layer.shape[1]
    comm_stream = _get_ring_comm_stream()

    for step in range(comm.world_size):
        current_is_early_stop = window_size[0] != -1 and step == 2
        next_is_early_stop = window_size[0] != -1 and step == 1
        if current_is_early_stop:
            break
        if step + 1 != comm.world_size and not next_is_early_stop:
            next_k: torch.Tensor = comm.send_recv(key_layer)
            next_v: torch.Tensor = comm.send_recv(value_layer)
            with torch.cuda.stream(comm_stream):
                comm.commit()

        key, value = key_layer, value_layer

        if original_window_size[0] == -1:
            adjusted_left = -1
        else:
            adjusted_left = original_window_size[0] - step * chunk_len

        adjusted_right = original_window_size[1] if step == 0 else -1
        adjusted_window_size = (adjusted_left, adjusted_right)

        if not causal or step <= comm.rank:
            block_out, block_lse = call_block_attn(
                query_layer,
                key,
                value,
                sinks,
                softmax_scale,
                causal and step == 0,
                adjusted_window_size,
            )

            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size and not next_is_early_stop:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    output = out

    return output


def _moreh_gpt_attention_balanced_window(
    module,
    query,
    key,
    value,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH
    assert window_size[1] == -1, "Only left-side window size is supported in balanced ring attention."
    assert causal is True, "Balanced Ring Attention requires causal=True."

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    ulysses_size = dist.get_world_size(module.ulysses_pg)

    comm0 = RingComm(module.ring_pg)
    comm1 = RingComm(module.ring_pg)
    comm1.send_rank, comm1.recv_rank = comm1.recv_rank, comm1.send_rank

    global _WARMUPED

    if not _WARMUPED:
        for comm in [comm0, comm1]:
            tensor = torch.empty_like(key)
            received_tensor: torch.Tensor = comm.send_recv(tensor)
            comm_stream = _get_ring_comm_stream()
            with torch.cuda.stream(comm_stream):
                comm.commit()
            comm.wait()
            received_tensor += 1.0
        _WARMUPED = True

    query_layer = query
    key_layer = key
    value_layer = value

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))

    block_seq_len = query_layer.shape[1] // 2
    query0 = query_layer[:, :block_seq_len]
    query1 = query_layer[:, block_seq_len:]
    key_layer0 = key_layer[:, :block_seq_len]
    key_layer1 = key_layer[:, block_seq_len:]
    value_layer0 = value_layer[:, :block_seq_len]
    value_layer1 = value_layer[:, block_seq_len:]

    out = None
    lse = None

    b, s, h, _ = query_layer.shape
    out = torch.zeros_like(query_layer)
    lse = torch.full((b, s, h, 1), dtype=torch.float32, device=query_layer.device, fill_value=-float("inf"))

    original_window_size = window_size
    chunk_len_zigzag = query_layer.shape[1] // 2

    for step in range(comm0.world_size):
        current_is_early_stop = step == 2
        next_is_early_stop = step == 1
        if current_is_early_stop:
            break
        if step + 1 != comm0.world_size and not next_is_early_stop:
            next_k0: torch.Tensor = comm0.send_recv(key_layer0.contiguous())
            next_k1: torch.Tensor = comm1.send_recv(key_layer1.contiguous())
            next_v0: torch.Tensor = comm0.send_recv(value_layer0.contiguous())
            next_v1: torch.Tensor = comm1.send_recv(value_layer1.contiguous())
            comm0.commit()
            comm1.commit()

        key, value = key_layer, value_layer

        if original_window_size[0] == -1:
            adjusted_left = -1
        else:
            adjusted_left = original_window_size[0] - step * chunk_len_zigzag

        adjusted_window_size = (adjusted_left, -1)

        if step == 0:
            if comm0.rank == comm0.world_size - 1:
                block_out, block_lse = call_block_attn(
                    query_layer,
                    key,
                    value,
                    sinks,
                    softmax_scale,
                    causal,
                    adjusted_window_size,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                block_out, block_lse = call_block_attn(
                    query0,
                    key_layer0,
                    value_layer0,
                    sinks,
                    softmax_scale,
                    causal,
                    adjusted_window_size,
                )
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(None, block_seq_len)),
                )

                block_out, block_lse = call_block_attn(
                    query1,
                    key_layer1,
                    value_layer1,
                    sinks,
                    softmax_scale,
                    causal,
                    adjusted_window_size,
                )
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

        elif step == 1:
            if comm1.rank != 0:
                block_out, block_lse = call_block_attn(
                    query0,
                    key_layer0,
                    value_layer0,
                    sinks,
                    softmax_scale,
                    False,
                    adjusted_window_size,
                )
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(None, block_seq_len)),
                )
            if comm1.rank != comm1.world_size - 1:
                block_out, block_lse = call_block_attn(
                    query1,
                    key_layer1,
                    value_layer1,
                    sinks,
                    softmax_scale,
                    False,
                    adjusted_window_size,
                )
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )
        else:
            assert False, "Step should not be greater than 1 in balanced windowed attention."

        if step + 1 != comm0.world_size and not next_is_early_stop:
            comm0.wait()
            comm1.wait()
            key_layer0 = next_k0
            key_layer1 = next_k1
            value_layer0 = next_v0
            value_layer1 = next_v1

    out = out.to(query.dtype)
    if dist.get_world_size(module.ulysses_pg) > 1:
        output = SeqAllToAll4D.apply(module.ulysses_pg, out, module.gather_idx, module.scatter_idx)
    else:
        output = out

    return output


def _moreh_gpt_attention_balanced_full(
    module,
    query,
    key,
    value,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH
    assert window_size == (-1, -1), "Balanced Ring Attention currently only supports full attention (no windowing)."
    assert causal is True, "Balanced Ring Attention requires causal=True."

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    ulysses_size = dist.get_world_size(module.ulysses_pg)

    comm = RingComm(module.ring_pg)

    global _WARMUPED

    if not _WARMUPED:
        tensor = torch.empty_like(key)
        received_tensor: torch.Tensor = comm.send_recv(tensor)
        comm_stream = _get_ring_comm_stream()
        with torch.cuda.stream(comm_stream):
            comm.commit()
        comm.wait()
        received_tensor += 1.0
        _WARMUPED = True

    if ulysses_size > 1:
        query_layer = SeqAllToAll4D.apply(module.ulysses_pg, query, module.scatter_idx, module.gather_idx)
        key_layer = SeqAllToAll4D.apply(module.ulysses_pg, key, module.scatter_idx, module.gather_idx)
        value_layer = SeqAllToAll4D.apply(module.ulysses_pg, value, module.scatter_idx, module.gather_idx)
        ulysses_rank = dist.get_rank(module.ulysses_pg)
        sinks = sinks.chunk(ulysses_size, dim=0)[ulysses_rank].contiguous()
    else:
        query_layer = query
        key_layer = key
        value_layer = value

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))

    assert causal, "Balanced Ring Attention requires causal=True"
    block_seq_len = query_layer.shape[1] // 2
    query1 = query_layer[:, block_seq_len:]

    out = None
    lse = None

    next_k, next_v = None, None

    window_size = (-1, -1)
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(key_layer)
            next_v: torch.Tensor = comm.send_recv(value_layer)
            comm.commit()

        key, value = key_layer, value_layer

        if step == 0:
            block_out, block_lse = call_block_attn(
                query_layer,
                key,
                value,
                sinks,
                softmax_scale,
                causal,
                window_size,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        elif step <= comm.rank:
            key0 = key[:, :block_seq_len]
            value0 = value[:, :block_seq_len]

            if key0.shape[1] > 0:
                block_out, block_lse = call_block_attn(
                    query_layer,
                    key0,
                    value0,
                    sinks,
                    softmax_scale,
                    False,
                    window_size,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        else:
            block_out, block_lse = call_block_attn(
                query1,
                key,
                value,
                sinks,
                softmax_scale,
                False,
                window_size,
            )
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    if dist.get_world_size(module.ulysses_pg) > 1:
        output = SeqAllToAll4D.apply(module.ulysses_pg, out, module.gather_idx, module.scatter_idx)
    else:
        output = out

    return output


def moreh_gpt_attention_balanced(
    module,
    query,
    key,
    value,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert module.attn_type == AttnType.TORCH
    assert window_size[1] == -1, "Balanced Ring Attention currently only supports left windowing."
    assert causal is True, "Balanced Ring Attention requires causal=True."

    if window_size[0] == -1:
        return _moreh_gpt_attention_balanced_full(
            module,
            query,
            key,
            value,
            sinks,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            is_kernel_bhsd=is_kernel_bhsd,
        )
    else:
        return _moreh_gpt_attention_balanced_window(
            module,
            query,
            key,
            value,
            sinks,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            is_kernel_bhsd=is_kernel_bhsd,
        )