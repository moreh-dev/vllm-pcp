#!/bin/bash
# Script for Node 1 (192.168.3.44) - Prefill Server + Proxy + Test Runner
set -xe

MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-V2-Lite"}
PREFILLER_TP_SIZE=1
PREFILLER_RP_SIZE=8
PREFILL_PORT=8100
SIDE_CHANNEL_PORT=5559
DECODER_HOST="192.168.3.33"
DECODER_PORT=8200

# Get script directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GIT_ROOT=$(cd "$SCRIPT_DIR/../../../../../" && pwd)

# Build kv-transfer-config for prefill
PREFILL_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Cleanup local processes on exit
trap 'pkill -f vllm.entrypoints.openai.api_server || true; pkill -f toy_proxy_server.py || true' EXIT

# 1. Start prefill instance (RP=8)
echo "Starting prefill instance on 192.168.3.44 (RP=$PREFILLER_RP_SIZE)..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_KV_CACHE_LAYOUT='HND' \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_HOST=192.168.3.44 \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $PREFILL_PORT \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size $PREFILLER_TP_SIZE \
  --ring-parallel-size $PREFILLER_RP_SIZE \
  --kv-transfer-config "$PREFILL_KV_CONFIG" &

# 2. Wait for local prefill server
echo "Waiting for local prefill server..."
until curl -s localhost:${PREFILL_PORT}/v1/completions > /dev/null; do
  sleep 2
done

# 3. Wait for remote decode server
echo "Waiting for remote decode server on $DECODER_HOST..."
until curl -s ${DECODER_HOST}:${DECODER_PORT}/v1/completions > /dev/null; do
  sleep 2
done

# 4. Start proxy server
echo "Starting proxy server..."
python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts 192.168.3.44 \
  --prefiller-ports $PREFILL_PORT \
  --decoder-hosts $DECODER_HOST \
  --decoder-ports $DECODER_PORT &

echo "Prefill server and Proxy are running!"
echo "You can now run 'bash run_client.sh' to send the prompt."

# Wait for background processes to keep logs visible
wait
