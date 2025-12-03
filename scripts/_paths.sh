#!/usr/bin/env bash
# ------------------------------------------------------------
# vllm-orchestrator : 共通パス定義
# ------------------------------------------------------------

# この scripts ディレクトリ自身
SCRIPTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vllm control の最上位ディレクトリ
ROOT="$(cd "${SCRIPTS_ROOT}/.." && pwd)"

# vllm control 専用の Python 仮想環境
VENV="${ROOT}/.venv"

# 実行ユーザのホーム
REAL_HOME="$(getent passwd "$USER" | cut -d: -f6)"

# モデル格納場所（共通）
MODEL_ROOT="${REAL_HOME}/models"

# systemd unit / env ファイルの所在
UNIT_DIR="${ROOT}/systemd_units"
ENV_DIR="${UNIT_DIR}/env"


