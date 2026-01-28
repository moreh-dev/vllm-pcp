#!/bin/bash
# Script for Node 2 (192.168.3.33) - Decode Server
set -xe

MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-V2-Lite"}
DECODER_TP_SIZE=8
DECODER_RP_SIZE=1
PORT=8200
SIDE_CHANNEL_PORT=5659
DECODER_KV_LAYOUT="HND"

# Build kv-transfer-config for decode
DECODE_KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

echo "Starting decode instance on 192.168.3.33 (TP=$DECODER_TP_SIZE)..."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_KV_CACHE_LAYOUT=$DECODER_KV_LAYOUT \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $PORT \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size $DECODER_TP_SIZE \
  --ring-parallel-size $DECODER_RP_SIZE \
  --kv-transfer-config "$DECODE_KV_CONFIG"
