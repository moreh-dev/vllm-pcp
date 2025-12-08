import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from unittest.mock import MagicMock
from transformers import DeepseekV2Config

# Adjust path to include the current directory so we can import from vllm and resources
import sys
sys.path.append(os.getcwd())

from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig
from yunchang.globals import PROCESS_GROUP
from yunchang.kernels import AttnType
from resources.ring_attention import moreh_gpt_attention

class MockModule:
    def __init__(self, ring_pg, ulysses_pg, use_pack_qkv=False, attn_type=AttnType.TORCH):
        self.ring_pg = ring_pg
        self.ulysses_pg = ulysses_pg
        self.use_pack_qkv = use_pack_qkv
        self.attn_type = attn_type
        self.scatter_idx = 2 
        self.gather_idx = 1 
        self.ulysses_pg_group = ulysses_pg

def get_vllm_config():
    # Create minimal config objects
    model_config = MagicMock(spec=ModelConfig)
    model_config.max_model_len = 4096
    model_config.dtype = torch.bfloat16
    
    # We need hf_config for some checks
    hf_config = DeepseekV2Config(
        num_hidden_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        model_type="deepseek_v2",
        vocab_size=1000,
    )
    # Mock rope_parameters as expected by vLLM DeepseekV2 model
    hf_config.rope_parameters = {"rope_type": "default", "rope_theta": 10000, "factor": 1.0}
    model_config.hf_config = hf_config

    cache_config = MagicMock(spec=CacheConfig)
    cache_config.block_size = 16
    cache_config.sliding_window = None
    cache_config.cache_dtype = "auto"
    cache_config.calculate_kv_scales = False

    scheduler_config = MagicMock(spec=SchedulerConfig)
    scheduler_config.max_num_batched_tokens = 4096
    scheduler_config.max_num_seqs = 256
    
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.quant_config = None
    # Use a real CompilationConfig or fully mocked one with required attributes
    # CustomOp checks custom_ops list count of "all"/"none"
    # It also checks mode
    from vllm.config.compilation import CompilationConfig
    vllm_config.compilation_config = CompilationConfig()
    vllm_config.compilation_config.custom_ops = ["all"]
    vllm_config.compilation_config.static_forward_context = {}

    return vllm_config, hf_config, cache_config


def run_test(rank, world_size):
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
        
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Init vLLM distributed state
    from vllm.distributed import parallel_state
    # Initialize the "World" state wrapper
    # Note: init_distributed_environment checks if dist is initialized and uses it.
    parallel_state.init_distributed_environment() 
    
    # Initialize Model Parallelism
    # We use TP=1 because we want each rank to have a full set of heads (standard CP logic often interacts with TP, 
    # but for this specific Ring Attention test, we are manually handling the ring communication and 
    # want the layer to think it's standalone or full-heads).
    # We set ring_model_parallel_size=world_size to match the manual ring setup, although we manually manage groups too.
    parallel_state.ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ring_model_parallel_size=world_size  # Optional: mirrors the user's note about RP group
    )
    
    # Create Process Groups
    # Ring PG covers everyone
    ring_pg = dist.new_group(list(range(world_size)))
    # Ulysses PG covers just self for this test (pure ring test)
    ulysses_pg = dist.new_group([rank])
    
    # Set global PGs for yunchang
    PROCESS_GROUP.RING_PG = ring_pg
    PROCESS_GROUP.ULYSSES_PG = ulysses_pg
    
    # --- Config ---
    hidden_size = 2048
    num_heads = 16 # total heads
    
    # MLA specifics
    # qk_head_dim = qk_nope + qk_rope = 256 + 64 = 320 (example)
    # v_head_dim = 256 (example, different from qk)
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    
    # Update: User requested comparison with equal dimensions first, then changed ring_attention to support diff.
    # Let's use DIFFERENT dimensions to prove the user's change works.
    # qk = 192, v = 128
    
    q_lora_rank = 1024
    kv_lora_rank = 512
    
    vllm_config, hf_config, cache_config = get_vllm_config()
    
    # Import config setter
    from vllm.config.vllm import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        # Initialize Layer
        layer = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=4096,
            cache_config=cache_config,
        ).to(device).to(torch.bfloat16)
        
        # Create Inputs
        batch_size = 1
        seq_len = 128
        total_seq_len = seq_len * world_size
        
        # We want to test that if we have a full sequence, processed by ring attention (chunked),
        # matches the full sequence processed by vanilla attention.
        
        # Input for Vanilla (Whole Global Sequence)
        # We need to construct a global input and then slice it for the local rank to simulate "Ring" input.
        # BUT, to verify correctness, it's easier to verify:
        # Vanilla(Global_Input) == Gather(Ring_Forward(Local_Input))
        
        torch.manual_seed(42)
        # Global hidden states
        global_hidden_states = torch.randn(batch_size, total_seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        positions = torch.arange(total_seq_len, device=device).unsqueeze(0)
        
        # --- 1. Vanilla Forward ---
        # We run this only on rank 0 roughly, or everyone runs it on global data.
        # Ideally everyone runs it to verify.
        with torch.no_grad():
            vanilla_output = layer(positions, global_hidden_states)
        
        # --- 2. Ring Attention Forward ---
        # Prepare Local Input
        start_idx = rank * seq_len
        end_idx = (rank + 1) * seq_len
        local_hidden_states = global_hidden_states[:, start_idx:end_idx, :].clone()
        local_positions = positions[:, start_idx:end_idx].clone()
        
        # We need to manually perform the projections + RoPE because DeepseekV2MLAAttention
        # wraps everything inside. We want to test "MLA Projections -> Ring Attention -> Output Projection"
        
        # Step A: Projections (mimic layer.forward_native up to attention call)
        # layer has: fused_qkv_a_proj, q_b_proj, kv_b_proj, etc.
        
        with torch.no_grad():
            # MLA Logic extracted from forward_native
            # 1. Down Projection (Wa)
            qkv_lora = layer.fused_qkv_a_proj(local_hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [layer.q_lora_rank, layer.kv_lora_rank + layer.qk_rope_head_dim],
                dim=-1
            )
            
            # 2. Q Up Projection (Wb)
            q_c = layer.q_a_layernorm(q_c)
            q = layer.q_b_proj(q_c)[0]
            q = q.view(batch_size, seq_len, layer.num_heads, layer.qk_head_dim)
            
            # 3. KV Split & Norm
            kv_c, k_pe = kv_lora.split([layer.kv_lora_rank, layer.qk_rope_head_dim], dim=-1)
            kv_c_normed = layer.kv_a_layernorm(kv_c)
            
            # 4. KV Up Projection (Wb) for Key Generation (for ring attn we need full keys/values)
            # Wait, MLA usually keeps KV compressed. 
            # But ring_attention.py expects: query, key, value tensors.
            # "Vanilla MLA" = MLAAttention(use_sparse=False) -> uses cuda_impl/triton_impl
            # If we use ring_attention, we need materialized K and V?
            # Yes, ring_attention.py does standard attention. 
            # MLA can be viewed as standard attention if we project K and V fully.
            # K = [kv_c_normed @ W_Uk, k_pe]
            # V = [kv_c_normed @ W_Uv]
            
            # layer.kv_b_proj projects to (qk_nope_head_dim + v_head_dim)
            # It takes kv_c_normed
            kv_up = layer.kv_b_proj(kv_c_normed)[0]
            kv_up = kv_up.view(batch_size, seq_len, layer.num_heads, layer.qk_nope_head_dim + layer.v_head_dim)
            k_nope, v = kv_up.split([layer.qk_nope_head_dim, layer.v_head_dim], dim=-1)
            
            # 5. RoPE
            # k_pe needs to be expanded/repeated?
            # In forward_native:
            # k_pe = k_pe.unsqueeze(1) (batch, 1, seq, dim) ? No, typically (batch, seq, head, dim) or sim.
            # In forward_native: k_pe = k_pe.unsqueeze(1) -> (B, 1, S, D) ?
            # Let's check code:
            # k_pe = k_pe.unsqueeze(1)
            # rotary_emb(positions, q[..., nope:], k_pe)
            # q shape is (B*S, H, D) flattened or (B, S, H, D) depending on impl.
            # Here we reshaped to (B, S, H, D).
            
            # Apply RoPE
            q_nope = q[..., :layer.qk_nope_head_dim]
            q_pe = q[..., layer.qk_nope_head_dim:]
            
            # k_pe comes from latent, shape (B, S, D_rope).
            # We need to broadcast it to heads?
            # In MLA, k_pe is shared across heads usually (Multi-Query for PE part).
            # Wrapper says: k_pe = k_pe.unsqueeze(1) before rotary.
            # `rotary_emb` handles the broadcasting if needed?
            
            # vllm rotary: forward(positions, q, k)
            # positions: (B, S)
            # q: (B, S, H, D) or (B, H, S, D)?
            # wrapper.forward_native:
            # q = q.view(-1, self.num_heads, self.qk_head_dim) # (TotalTokens, H, D)
            # positions flattened.
            # We are using batch mode here for ring attention.
            
            # Let's simulate flattening for RoPE to match vLLM utility if needed, or just map carefully.
            # vLLM rope expects (num_tokens, num_heads, head_size).
            
            # Let's verify shapes.
            # q: (1, 128, 16, 192) -> view -> (128, 16, 192)
            q_flat = q.view(-1, layer.num_heads, layer.qk_head_dim)
            
            k_pe_flat = k_pe.reshape(-1, 1, layer.qk_rope_head_dim) # (128, 1, 64)
            
            q_nope_flat = q_flat[..., :layer.qk_nope_head_dim]
            q_pe_flat = q_flat[..., layer.qk_nope_head_dim:]
            
            positions_flat = local_positions.view(-1)
            
            q_pe_rot, k_pe_rot = layer.rotary_emb(positions_flat, q_pe_flat, k_pe_flat)
            
            # Re-assemble Q
            q_final = torch.cat([q_nope_flat, q_pe_rot], dim=-1) # (128, 16, 192)
            
            # Re-assemble K
            # K consists of k_nope and k_pe_rot.
            # k_nope comes from kv_up (B, S, H, D_nope). 
            # k_pe_rot is (B*S, 1, D_rope). We need to broadcast k_pe to H heads?
            # Yes, MLA uses shared PE.
            # But for Ring Attention (standard MHA kernel), we need materialized K per head.
            k_nope_flat = k_nope.view(-1, layer.num_heads, layer.qk_nope_head_dim)
            k_pe_expanded = k_pe_rot.expand(-1, layer.num_heads, -1)
            k_final = torch.cat([k_nope_flat, k_pe_expanded], dim=-1) # (128, 16, 192)
            
            # V
            v_final = v.view(-1, layer.num_heads, layer.v_head_dim) # (128, 16, 128)
            
            # Reshape back to (B, S, H, D) for Ring Attention
            q_ring = q_final.view(batch_size, seq_len, layer.num_heads, layer.qk_head_dim)
            k_ring = k_final.view(batch_size, seq_len, layer.num_heads, layer.qk_head_dim)
            v_ring = v_final.view(batch_size, seq_len, layer.num_heads, layer.v_head_dim)
            
            # --- Step B: Ring Attention ---
            # ring_attention.py expects (B, S, H, D).
            # We need mock module.
            mock_module = MockModule(ring_pg, ulysses_pg, attn_type=AttnType.TORCH)
            
            # Sinks?
            # Only used if has_sink?
            # The kernel def: sinks: torch.Tensor,
            # In ring_attention.py: sinks = sinks.chunk(...)
            # We can pass dummy sinks if not using them or zeros.
            sinks = torch.zeros(batch_size, layer.num_heads, 64, device=device, dtype=torch.bfloat16) # dummy
            # Wait, kernel expects sinks to be [num_query_heads] in one place, or (B, H, Sinks)?
            # `torch_attention_with_sinks_forward` uses sinks. 
            # `triton_attention_forward` doc: `sinks_ptr`.
            # For simplicity, pass zeros or empty. ring_attention.py takes `sinks`.
            # Let's provide zeros.
            
            # Also need softmax_scale.
            scale = layer.scaling
            
            # Ring Attention Call
            # Ensure imports and updated code are used.
            # The user updated `moreh_gpt_attention` (which calls `call_block_attn`).
            # `moreh_gpt_attention` signature: (module, query, key, value, sinks, ..., causal, window_size)
            
            ring_out = moreh_gpt_attention(
                mock_module,
                q_ring,
                k_ring,
                v_ring,
                sinks,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
            )
            # ring_out: (B, S, H, V_D) -> (1, 128, 16, 128)
            
            # --- Step C: Output Projection ---
            # Flatten for linear
            ring_out_flat = ring_out.reshape(-1, layer.num_heads * layer.v_head_dim)
            final_ring_output = layer.o_proj(ring_out_flat)[0] # (128, Hidden)
            
            # Result is local. We need to gather to verify against global vanilla output.
            # Gather all outputs
            all_ring_outputs = [torch.zeros_like(final_ring_output) for _ in range(world_size)]
            dist.all_gather(all_ring_outputs, final_ring_output, group=ring_pg)
            
            full_ring_output = torch.cat(all_ring_outputs, dim=0) # (256, Hidden)
            
            # Reshape to (B, TotalSeq, Hidden)
            full_ring_output = full_ring_output.view(batch_size, total_seq_len, hidden_size)
            
        # --- 3. Compare ---
        if rank == 0:
            print(f"Vanilla Output Shape: {vanilla_output.shape}")
            print(f"Ring Output Shape: {full_ring_output.shape}")
            
            # Close check
            # BF16 might have some tolerance issues.
            tolerance = 1e-2
            diff = (vanilla_output - full_ring_output).abs().max().item()
            mean_diff = (vanilla_output - full_ring_output).abs().mean().item()
            print(f"Max Diff: {diff}")
            print(f"Mean Diff: {mean_diff}")
            
            assert diff < tolerance, f"Mismatch! Max diff {diff} exceeds tolerance {tolerance}"
            print("TEST PASSED: Vanilla MLA matches Manual CP Ring MLA.")

    dist.barrier()
    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()


if __name__ == "__main__":
    world_size = 2
    try:
        if torch.cuda.device_count() < world_size:
            print(f"Skipping test: Need {world_size} GPUs, found {torch.cuda.device_count()}")
        else:
            mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test Failed with exception: {e}")
        # For non-cuda env just to dry run logic (will fail on device set/kernel)
        pass
