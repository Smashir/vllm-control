#!/bin/bash
# ============================================================
# check_vllm.sh v2 - vLLMサービス診断スクリプト（強化版）
# ============================================================
# 使い方:
#   check_vllm.sh <model_name> [--diag]
#   vllmctl.sh check <model_name> [--diag]
#
# 主な機能:
#   - systemd サービス状態確認
#   - Main PID / CPU / Memory / GPUメモリ
#   - /proc/<pid>/environ の VLLM_* 環境変数
#   - .env ファイル読み取り（PORTなど）
#   - API疎通チェック (localhost:PORT/v1/models)
#   - --diag で journalctl 最新100行出力
# ============================================================

source "$(dirname "$0")/_paths.sh"


MODEL="$1"
DIAG="$2"
#BASE_DIR="/home/aiuser/systemd_units/env"
BASE_DIR="${ENV_DIR}"

ENV_FILE="${BASE_DIR}/${MODEL}.env"
SVC="vllm@${MODEL}.service"

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model_name> [--diag]"
    exit 1
fi

echo "=== vLLM Service Check ==="
echo

# ------------------------------------------------------------
# 1. systemd サービス状態
# ------------------------------------------------------------
echo "【Systemd サービス】"
sudo systemctl list-units --type=service --all | grep "vllm@" | grep "$MODEL" || {
    echo "[WARN] 該当サービスが存在しません: $SVC"
    exit 1
}
echo

STATUS=$(systemctl is-active "$SVC" 2>/dev/null)
if [[ "$STATUS" == "active" ]]; then
    echo "[OK] サービスは稼働中"
else
    echo "[NG] サービスは停止中: $STATUS"
fi

# ------------------------------------------------------------
# 2. Main PID, メモリ, CPU
# ------------------------------------------------------------
PID=$(systemctl show -p MainPID "$SVC" | cut -d= -f2)
if [[ "$PID" -gt 0 ]]; then
    echo
    echo "【プロセス情報】"
    ps -p "$PID" -o pid,pcpu,pmem,cmd --no-headers
else
    echo "[WARN] MainPIDが取得できません。"
fi

# ------------------------------------------------------------
# 3. GPUメモリ状況
# ------------------------------------------------------------
echo
echo "【GPU情報】"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -n 1 2>/dev/null \
  || echo "(nvidia-smi 未検出)"

# ------------------------------------------------------------
# 4. 環境変数確認 (VLLM関係)
# ------------------------------------------------------------
if [[ "$PID" -gt 0 ]]; then
    echo
    echo "【実行中の環境変数（VLLM関係）】"
    sudo cat /proc/$PID/environ | tr '\0' '\n' | grep '^VLLM_' || echo "(なし)"
fi

# ------------------------------------------------------------
# 5. .env ファイル内容とPORT抽出
# ------------------------------------------------------------
echo
if [[ -f "$ENV_FILE" ]]; then
    echo "【環境ファイル: $ENV_FILE】"
    grep -E '^(MODEL_PATH|PORT|DTYPE|QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN)' "$ENV_FILE"
    PORT=$(grep -oP '(?<=PORT=)\d+' "$ENV_FILE" 2>/dev/null || echo 8000)
else
    echo "[WARN] 環境ファイルが見つかりません: $ENV_FILE"
    PORT=8000
fi

# ------------------------------------------------------------
# 6. API疎通確認
# ------------------------------------------------------------
echo
echo "【API応答確認】"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" || echo 000)
echo "  → HTTP ${STATUS_CODE}"

# ------------------------------------------------------------
# 7. 詳細診断 (--diag)
# ------------------------------------------------------------
if [[ "$DIAG" == "--diag" ]]; then
    echo
    echo "【journalctl 最新ログ（100行）】"
    sudo journalctl -u "$SVC" -n 100 --no-pager | tail -n 100
fi

echo
echo "=== チェック完了: $MODEL ($STATUS) ==="
