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


def call_block_attn(
    query,
    key,
    value,
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
        if step + 1 != comm.world_size:
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
                softmax_scale,
                causal and step == 0,
                adjusted_window_size,
            )

            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            key_layer = next_k
            value_layer = next_v

    out = out.to(query.dtype)
    output = out

    return output


def _moreh_mla_ring_attention_balanced_full(
    module,
    query,
    kv_c_normed,
    k_pe,
    sinks,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    is_kernel_bhsd: bool = True,
) -> torch.Tensor:
    assert module.use_pack_qkv is False, "Packed QKV is not supported in this attention implementation."
    assert window_size == (-1, -1), "Balanced Ring Attention currently only supports full attention (no windowing)."
    assert causal is True, "Balanced Ring Attention requires causal=True."

    comm = RingComm(module.ring_pg)

    global _WARMUPED

    if not _WARMUPED:
        tensor = torch.empty_like(kv_c_normed)
        received_tensor: torch.Tensor = comm.send_recv(tensor)
        comm_stream = _get_ring_comm_stream()
        with torch.cuda.stream(comm_stream):
            comm.commit()
        comm.wait()
        received_tensor += 1.0
        _WARMUPED = True

    query_layer = query
    kv_c_normed_layer = kv_c_normed
    k_pe_layer = k_pe

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query_layer.size(-1))

    assert causal, "Balanced Ring Attention requires causal=True"
    block_seq_len = query_layer.shape[1] // 2
    query1 = query_layer[:, block_seq_len:]

    out = None
    lse = None

    next_kv_c, next_k_pe = None, None

    window_size = (-1, -1)
    
    # Pre-project local K, V for the first step
    # Re-use projections if possible OR just project on the fly inside the loop.
    # To save memory, we project inside the loop.
    
    comm_stream = _get_ring_comm_stream()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_kv_c: torch.Tensor = comm.send_recv(kv_c_normed_layer)
            next_k_pe: torch.Tensor = comm.send_recv(k_pe_layer)
            with torch.cuda.stream(comm_stream):
                comm.commit()

        # Project KV
        # kv_c_normed_layer: [B, S, Lkv]
        # k_pe_layer: [B, S, 1, R] or [B, S, R] -> check dimensions
        
        # NOTE: We assume module has kv_b_proj
        # In MLACommonImpl:
        # kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        # k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        
        # We need to reshape for Linear projection:
        B, S, _ = kv_c_normed_layer.shape
        kv_nope_flat = module.kv_b_proj(kv_c_normed_layer.view(-1, kv_c_normed_layer.shape[-1]))[0]
        kv_nope = kv_nope_flat.view(B, S, module.num_heads, module.qk_nope_head_dim + module.v_head_dim)
        k_nope, v = kv_nope.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)
        
        # k_pe might be [B, S, 1, R] or [B, S, R] ?
        # In common.py calling site:
        # k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1) -> [S, 1, R]
        # Then allgathered. 
        # So we should expect k_pe to be compatible with k_nope expansion
        
        # If k_pe_layer is [B, S, R], unsqueeze it.
        curr_k_pe = k_pe_layer
        if curr_k_pe.dim() == 3: # [B, S, D]
             curr_k_pe = curr_k_pe.unsqueeze(2) # [B, S, 1, D]

        k = torch.cat((k_nope, curr_k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        value = v
        key = k

        if step == 0:
            block_out, block_lse = call_block_attn(
                query_layer,
                key,
                value,
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
            kv_c_normed_layer = next_kv_c
            k_pe_layer = next_k_pe

    out = out.to(query.dtype)
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
    assert window_size == (-1, -1), "Balanced Ring Attention currently only supports full attention (no windowing)."
    assert causal is True, "Balanced Ring Attention requires causal=True."

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
    output = out

    return output