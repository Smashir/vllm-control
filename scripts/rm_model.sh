#!/usr/bin/env bash
# ============================================================
# rm_model.sh
# ------------------------------------------------------------
# 指定した vLLM モデルを安全に削除。
# - systemスコープの systemd 登録を解除
# - ~/systemd_units/env/<model>.env を削除
# ============================================================

set -e

source "$(dirname "$0")/_paths.sh"

MODEL_NAME="$1"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

#REAL_HOME=$(getent passwd aiuser | cut -d: -f6)
#MODEL_DIR="${REAL_HOME}/models/${MODEL_NAME}"

MODEL_DIR="${MODEL_ROOT}/${MODEL_NAME}"
#ENV_FILE="${REAL_HOME}/systemd_units/env/${MODEL_NAME}.env"
ENV_FILE="${ENV_DIR}/${MODEL_NAME}.env"

SERVICE_NAME="vllm@${MODEL_NAME}.service"
SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}"
SYSTEMD_LINK="/etc/systemd/system/multi-user.target.wants/${SERVICE_NAME}"

echo "==============================="
echo "🧹 vLLM モデル削除スクリプト"
echo "==============================="
echo "Target Model: ${MODEL_NAME}"
echo "Model Dir:    ${MODEL_DIR}"
echo "Env File:     ${ENV_FILE}"
echo "Service:      ${SERVICE_NAME}"
echo "-------------------------------"

# ============================================================
# [1] systemd サービス停止・無効化
# ------------------------------------------------------------
if systemctl list-units --full -all | grep -q "${SERVICE_NAME}"; then
  echo "🛑 Stopping ${SERVICE_NAME}..."
  sudo systemctl stop "${SERVICE_NAME}" || true
fi

if systemctl list-unit-files | grep -q "${SERVICE_NAME}"; then
  echo "🔧 Disabling ${SERVICE_NAME}..."
  sudo systemctl disable "${SERVICE_NAME}" || true
fi

# ============================================================
# [2] systemd ユニットファイルとリンク削除
# ------------------------------------------------------------
if [[ -f "${SYSTEMD_UNIT}" ]]; then
  echo "🗑 Removing systemd unit file..."
  sudo rm -v "${SYSTEMD_UNIT}"
fi

if [[ -L "${SYSTEMD_LINK}" ]]; then
  echo "🧩 Removing systemd link..."
  sudo rm -v "${SYSTEMD_LINK}"
fi

# ============================================================
# [3] .env ファイル削除
# ------------------------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  echo "🗑 Removing env file..."
  rm -v "${ENV_FILE}"
else
  echo "⚠️ No env file found (skip)"
fi

# ============================================================
# [4] モデルディレクトリ削除
# ------------------------------------------------------------
if [[ -d "${MODEL_DIR}" ]]; then
  echo "🗑 Removing model directory..."
  rm -rf "${MODEL_DIR}"
else
  echo "⚠️ No model directory found (skip)"
fi

# ============================================================
# [5] ログ・キャッシュ整理
# ------------------------------------------------------------
echo "🧾 Clearing old logs..."
sudo journalctl --vacuum-time=1s > /dev/null 2>&1 || true

CACHE_DIR="${REAL_HOME}/.cache/huggingface/hub"
if [[ -d "${CACHE_DIR}" ]]; then
  echo "🧹 Optionally clearing HF cache..."
  echo "   (uncomment rm -rf to enable full wipe)"
  #rm -rf "${CACHE_DIR}"
fi

# ============================================================
# [6] systemd 再読込
# ------------------------------------------------------------
sudo systemctl daemon-reload

echo "✅ Model ${MODEL_NAME} cleanup complete."
