#!/bin/bash
set -xe

# Multi-node configuration
# Node 1: Prefill (RP=8)
# Node 2: Decode (TP=8)
PREFILL_HOST="192.168.3.44"
DECODER_HOST="192.168.3.33"

PREFILL_PORT=${PREFILL_PORT:-8100}
DECODER_PORT=${DECODER_PORT:-8200}

# Parallelism configuration
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
PREFILLER_RP_SIZE=${PREFILLER_RP_SIZE:-8}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-8}
DECODER_RP_SIZE=${DECODER_RP_SIZE:-1}

# Other configurations
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-V2-Lite"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
BLOCK_SIZE=${BLOCK_SIZE:-128}
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-"cuda"}

# Find the git repository root directory
GIT_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
RELATIVE_TEST_DIR="tests/v1/kv_connector/nixl_integration"
WORKDIR=$GIT_ROOT  # Assuming same path on all nodes

# Detect KV layout
DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"}
if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
  KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
else
  KV_CONFIG_HETERO_LAYOUT=''
fi

# Build kv-transfer-config
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  PREFILL_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"'${KV_CONFIG_HETERO_LAYOUT}'}'
  DECODE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"'${KV_CONFIG_HETERO_LAYOUT}'}'
else
  PREFILL_KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}"}"
  DECODE_KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}"}"
fi

# Function to clean up instances on all nodes
cleanup_instances() {
  echo "Cleaning up vLLM instances on all nodes..."
  ssh $PREFILL_HOST "pkill -f vllm.entrypoints.openai.api_server || true"
  ssh $DECODER_HOST "pkill -f vllm.entrypoints.openai.api_server || true"
  pkill -f "toy_proxy_server.py" || true
  sleep 2
}

# Trap signals for cleanup
trap cleanup_instances SIGINT SIGTERM EXIT

# Waits for vLLM to start
wait_for_server() {
  local host=$1
  local port=$2
  echo "Waiting for server $host:$port to start..."
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Initial cleanup
cleanup_instances

echo "================================"
echo "Starting Multi-Node Test"
echo "Prefill Node: $PREFILL_HOST (RP=$PREFILLER_RP_SIZE)"
echo "Decode Node:  $DECODER_HOST (TP=$DECODER_TP_SIZE)"
echo "================================"

# Start prefill instance on Node 1 (RP=8)
echo "Starting prefill instance on $PREFILL_HOST..."
ssh $PREFILL_HOST "cd $WORKDIR && \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_KV_CACHE_LAYOUT='HND' \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_HOST \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $PREFILL_PORT \
  --enforce-eager \
  --block-size $BLOCK_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --tensor-parallel-size $PREFILLER_TP_SIZE \
  --ring-parallel-size $PREFILLER_RP_SIZE \
  --kv-transfer-config '$PREFILL_KV_CONFIG'" > prefill_remote.log 2>&1 &

# Start decode instance on Node 2 (TP=8)
echo "Starting decode instance on $DECODER_HOST..."
ssh $DECODER_HOST "cd $WORKDIR && \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_KV_CACHE_LAYOUT=$DECODER_KV_LAYOUT \
  UCX_NET_DEVICES=all \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $DECODER_PORT \
  --enforce-eager \
  --block-size $BLOCK_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --tensor-parallel-size $DECODER_TP_SIZE \
  --ring-parallel-size $DECODER_RP_SIZE \
  --kv-transfer-config '$DECODE_KV_CONFIG'" > decode_remote.log 2>&1 &

# Wait for all instances to start
wait_for_server $PREFILL_HOST $PREFILL_PORT
wait_for_server $DECODER_HOST $DECODER_PORT

# Start the proxy server locally
PROXY_CMD="python3 ${GIT_ROOT}/${RELATIVE_TEST_DIR}/toy_proxy_server.py --port 8192"
PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOST}"
PROXY_CMD+=" --prefiller-ports ${PREFILL_PORT}"
PROXY_CMD+=" --decoder-hosts ${DECODER_HOST}"
PROXY_CMD+=" --decoder-ports ${DECODER_PORT}"

echo "Starting proxy server locally..."
$PROXY_CMD > proxy_local.log 2>&1 &
sleep 5

# Run simple correctness test
echo "Running correctness test..."
python3 ${GIT_ROOT}/${RELATIVE_TEST_DIR}/simple_correctness.py

echo "Multi-node test completed!"
