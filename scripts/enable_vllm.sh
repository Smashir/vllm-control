#!/usr/bin/env bash
# ===========================================
# enable_vllm.sh
# 指定した vLLM モデルを永続化 (自動起動有効)
# ===========================================

usage() {
    echo "Usage: $0 <model_name>"
    echo "例: $0 mistral-nemo-ja-2408"
    exit 1
}

MODEL_NAME="$1"
[[ -z "$MODEL_NAME" ]] && usage

sudo systemctl enable "vllm@${MODEL_NAME}.service" && \
echo "[OK] Enabled auto-start for model: ${MODEL_NAME}"
