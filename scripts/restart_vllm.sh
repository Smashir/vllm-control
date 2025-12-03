#!/usr/bin/env bash
# ===========================================
# restart_vllm.sh
# 指定した vLLM モデルを再起動する
# ===========================================

usage() {
    echo "Usage: $0 <model_name>"
    echo "例: $0 mistral-nemo-ja-2408"
    exit 1
}

MODEL_NAME="$1"
[[ -z "$MODEL_NAME" ]] && usage

sudo systemctl restart "vllm@${MODEL_NAME}.service" && \
echo "[OK] Restarted vLLM model: ${MODEL_NAME}"
