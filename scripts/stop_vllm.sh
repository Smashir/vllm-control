#!/usr/bin/env bash
# ===========================================
# stop_vllm.sh
# 指定した vLLM モデルを停止する。
# 引数なしの場合はすべて停止。
# ===========================================

usage() {
    echo "Usage: $0 [model_name | all]"
    echo "例: $0 mistral-nemo-ja-2408"
    echo "例: $0 all"
    exit 1
}

MODEL_NAME="$1"

if [[ -z "$MODEL_NAME" || "$MODEL_NAME" == "all" ]]; then
    echo "[INFO] 停止対象: すべてのvLLMインスタンス"
    sudo systemctl stop 'vllm@*'
else
    sudo systemctl stop "vllm@${MODEL_NAME}.service"
fi

echo "[OK] 停止完了。"
