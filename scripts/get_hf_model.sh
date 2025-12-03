#!/usr/bin/env bash
# =======================================================
# hf_get_model.sh : 高速・安全 Hugging Face モデルダウンロード (vllm_env対応)
# 自動: hf が無ければ vllm_env に自動インストール
# =======================================================
set -e

source "$(dirname "$0")/_paths.sh"

MODEL_ID="$1"
if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 <huggingface-model-id>"
  echo "例: $0 huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Captioner-abliterated"
  exit 1
fi

# === 基本設定 ===
VENV="${VENV:-${ROOT}/.venv}"
HF_CMD="${VENV}/bin/hf"

BASE_DIR="${MODEL_ROOT}"

# === 高速・安全設定 ===
export HF_HUB_ENABLE_SYMLINKS=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=1800

# =======================================================
# hf が無ければ自動インストール
# =======================================================
if [[ ! -x "$HF_CMD" ]]; then
  echo "⚠️  'hf' コマンドが見つかりません。自動インストールを行います。"
  echo "   → pip install -U huggingface_hub hf_transfer"

  # vllm_env が存在することを確認
  if [[ ! -d "$VLLM_ENV" ]]; then
    echo "❌ Error: vLLM environment not found at $VLLM_ENV"
    exit 1
  fi

  # 自動インストール
  "$VLLM_ENV/bin/python" -m pip install -U huggingface_hub

  # もう一度確認
  if [[ ! -x "$HF_CMD" ]]; then
    echo "❌ Error: hf コマンドのインストールに失敗しました"
    exit 1
  fi

  # hf_transfer があるか確認
  if ! "$VLLM_ENV/bin/python" -c "import hf_transfer" 2>/dev/null; then
    echo "❌ Error: hf_transfer がインストールされていません"
    # 自動インストール
    "$VLLM_ENV/bin/python" -m pip install -U hf_transfer
  fi

fi

# =======================================================
# モデル保存先決定
# =======================================================
MODEL_DIR_NAME=$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr '_' '-' )
MODEL_PATH="${BASE_DIR}/${MODEL_DIR_NAME}"

echo "==============================="
echo "📦 Hugging Face モデル取得 (最適化版)"
echo "==============================="
echo "Model ID:     ${MODEL_ID}"
echo "Save to:      ${MODEL_PATH}"
echo "Python Env:   ${VLLM_ENV}"
echo "-------------------------------"

mkdir -p "${MODEL_PATH}"

# === 実行 ===
echo "🚀 Downloading with parallel hf-transfer..."
"${HF_CMD}" download "${MODEL_ID}" \
  --local-dir "${MODEL_PATH}" \
  --exclude "training/*" "wandb/*"

echo "✅ Download complete."
echo "📁 Saved to: ${MODEL_PATH}"
ls -1 "${MODEL_PATH}" | head -n 10
