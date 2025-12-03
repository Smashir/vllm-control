#!/usr/bin/env bash
# ===========================================
# disable_vllm.sh
# 指定した vLLM モデルの永続化解除
# ===========================================

usage() {
    echo "Usage: $0 <model_name>"
    echo "例: $0 mistral-nemo-ja-2408"
    exit 1
}

MODEL_NAME="$1"
[[ -z "$MODEL_NAME" ]] && usage

sudo systemctl disable "vllm@${MODEL_NAME}.service" && \
echo "[OK] Disabled auto-start for model: ${MODEL_NAME}"
