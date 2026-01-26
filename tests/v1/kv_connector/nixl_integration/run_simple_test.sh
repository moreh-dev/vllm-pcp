#!/bin/bash
set -xe

# Parse command line arguments
KV_BUFFER_DEVICE="cuda"  # Default to cuda
while [[ $# -gt 0 ]]; do
  case $1 in
    --kv_buffer_device)
      KV_BUFFER_DEVICE="$2"
      shift 2
      ;;
    --ring_parallel_size)
      RING_PARALLEL_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--kv_buffer_device <cuda|cpu>]"
      exit 1
      ;;
  esac
done

echo "Running simple correctness test with kv_buffer_device=$KV_BUFFER_DEVICE"

DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"} # Default to HND, optional NHD
if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
  KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
else
  KV_CONFIG_HETERO_LAYOUT=''
fi

# Build the kv-transfer-config once
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"'${KV_CONFIG_HETERO_LAYOUT}'}'
else
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}"}"
fi

# Fixed model for simple test
MODEL_NAME="/model/"
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
PREFILLER_RP_SIZE=${PREFILLER_RP_SIZE:-${RING_PARALLEL_SIZE:-1}}
DECODER_RP_SIZE=${DECODER_RP_SIZE:-${RING_PARALLEL_SIZE:-1}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 2
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  else
    echo "1"
  fi
}

run_test() {
  cleanup_instances
  
  echo "================================"
  echo "Testing model: $MODEL_NAME"
  echo "================================"

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Calculate total GPUs needed for prefill
  PREFILL_GPUS_PER_INSTANCE=$((PREFILLER_TP_SIZE * PREFILLER_RP_SIZE))
  PREFILL_GPU_START=0

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate starting GPU for this prefill instance
    INSTANCE_GPU_START=$((PREFILL_GPU_START + i * PREFILL_GPUS_PER_INSTANCE))

    # Build GPU list for this instance
    GPU_ID=$INSTANCE_GPU_START
    for (( j=1; j < PREFILL_GPUS_PER_INSTANCE; j++ )); do
      GPU_ID="${GPU_ID},$((INSTANCE_GPU_START + j))"
    done

    PORT=$((8100 + i))
    SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME \
    --port $PORT \
    --enforce-eager \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '$KV_CONFIG'"

    if [[ -n "$PREFILLER_RP_SIZE" ]]; then
      BASE_CMD="$BASE_CMD --ring-parallel-size $PREFILLER_RP_SIZE"
    fi

    eval "$BASE_CMD &"

    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Calculate GPU allocation for decode instances
  DECODE_GPUS_PER_INSTANCE=$((DECODER_TP_SIZE * DECODER_RP_SIZE))
  TOTAL_PREFILL_GPUS=$((NUM_PREFILL_INSTANCES * PREFILL_GPUS_PER_INSTANCE))
  DECODE_GPU_START=$TOTAL_PREFILL_GPUS

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate starting GPU for this decode instance
    INSTANCE_GPU_START=$((DECODE_GPU_START + i * DECODE_GPUS_PER_INSTANCE))

    # Build GPU list for this instance
    GPU_ID=$INSTANCE_GPU_START
    for (( j=1; j < DECODE_GPUS_PER_INSTANCE; j++ )); do
      GPU_ID="${GPU_ID},$((INSTANCE_GPU_START + j))"
    done

    PORT=$((8200 + i))
    SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT=$DECODER_KV_LAYOUT \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME \
    --port $PORT \
    --enforce-eager \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --kv-transfer-config '$KV_CONFIG'"

    if [[ -n "$DECODER_RP_SIZE" ]]; then
      BASE_CMD="$BASE_CMD --ring-parallel-size $DECODER_RP_SIZE"
    fi
  
    if [[ -z "$DP_EP" ]]; then
      BASE_CMD="${BASE_CMD} --tensor-parallel-size $DECODER_TP_SIZE"
    else
      BASE_CMD="${BASE_CMD} --data-parallel-size $DECODER_TP_SIZE \
      --tensor-parallel-size 1 --enable-expert-parallel"
    fi

    eval "$BASE_CMD &"

    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server
  PROXY_CMD="python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &
  sleep 5

  # Run simple correctness test
  echo "Running simple correctness test..."
  python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/simple_correctness.py

  # Cleanup
  cleanup_instances
  sleep 3
}

run_test

echo "Test completed!"
