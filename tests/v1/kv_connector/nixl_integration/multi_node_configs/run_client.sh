#!/bin/bash
# Script to run the correctness test client
set -xe

# Get script directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GIT_ROOT=$(cd "$SCRIPT_DIR/../../../../../" && pwd)

echo "Running correctness test client (targeting proxy at localhost:8192)..."
python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/simple_correctness.py
