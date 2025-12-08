#!/usr/bin/env bash
# =======================================================
# hf_get_model.sh : 高速・安全 Hugging Face モデルダウンロード
# 自動: hf が無ければ自動インストール
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
  if [[ ! -d "$VENV" ]]; then
    echo "❌ Error: vLLM environment not found at $VENV"
    exit 1
  fi

  # 自動インストール
  "$VENV/bin/python" -m pip install -U huggingface_hub

  # もう一度確認
  if [[ ! -x "$HF_CMD" ]]; then
    echo "❌ Error: hf コマンドのインストールに失敗しました"
    exit 1
  fi

  # hf_transfer があるか確認
  if ! "$VENV/bin/python" -c "import hf_transfer" 2>/dev/null; then
    echo "❌ Error: hf_transfer がインストールされていません"
    # 自動インストール
    "$VENV/bin/python" -m pip install -U hf_transfer
  fi

fi


# =======================================================
# 🔐 HF ログインチェック（gated model のため）
# =======================================================
echo "🔐 Checking HuggingFace authentication..."

if ! "$HF_CMD" auth whoami >/dev/null 2>&1; then
  echo "⚠️  You are NOT logged in to HuggingFace."
  echo "   → Running 'hf auth login' ..."
  "$HF_CMD" auth login
else
  echo "✔ Logged in as: $("$HF_CMD" auth whoami)"
fi

# =======================================================
# モデル保存先決定
# =======================================================
MODEL_DIR_NAME=$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr '_' '-' )
MODEL_PATH="${BASE_DIR}/${MODEL_DIR_NAME}"

echo "==============================="
echo "📦 Hugging Face モデル取得"
echo "==============================="
echo "Model ID:     ${MODEL_ID}"
echo "Save to:      ${MODEL_PATH}"
echo "Python Env:   ${VENV}"
echo "-------------------------------"

mkdir -p "${MODEL_PATH}"

# =======================================================
# wandb フォルダ存在チェック（404 防止）
# =======================================================
echo "🔍 Checking presence of wandb/ ..."

if curl -s https://huggingface.co/api/models/${MODEL_ID}/tree \
    | grep -q '"path": "wandb/' ; then
  echo "→ wandb/ FOUND. Will exclude it."
  EXCLUDE_WANDB=" wandb/*"
else
  echo "→ wandb/ NOT found. Will NOT exclude it."
  EXCLUDE_WANDB=""
fi



# === 実行 ===
echo "🚀 Downloading with parallel hf-transfer..."
"${HF_CMD}" download "${MODEL_ID}" \
  --local-dir "${MODEL_PATH}" \
  --exclude "training/*${EXCLUDE_WANDB}"

echo "✅ Download complete."
echo "📁 Saved to: ${MODEL_PATH}"
ls -1 "${MODEL_PATH}" | head -n 10
