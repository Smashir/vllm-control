#!/usr/bin/env bash
# ===========================================
# start_vllm.sh
# 指定した vLLM モデルを起動する
# ===========================================

usage() {
    echo "Usage: $0 <model_name>"
    echo "例: $0 mistral-nemo-ja-2408"
    exit 1
}

MODEL_NAME="$1"
[[ -z "$MODEL_NAME" ]] && usage

sudo systemctl start "vllm@${MODEL_NAME}.service" && \
echo "[OK] Started vLLM model: ${MODEL_NAME}"
