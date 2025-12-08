#!/bin/bash
# ============================================================
# status_vllm.sh - 全モデル一覧ステータス表示
# ============================================================

source "$(dirname "$0")/_paths.sh"

ENV_DIR="${ENV_DIR}"
UNIT_PREFIX="vllm@"

printf "=== vLLM Status Overview ===\n\n"

# .env が一つもないとき
shopt -s nullglob
env_files=("${ENV_DIR}"/*.env)
if (( ${#env_files[@]} == 0 )); then
    echo "[INFO] 登録されたモデル (.env) がありません。"
    exit 0
fi

# 一覧ヘッダ
printf "%-25s %-10s %-10s %-8s %-10s\n" "MODEL" "STATE" "ENABLED" "PORT" "API"
printf "%-25s %-10s %-10s %-8s %-10s\n" "-----" "-----" "-------" "----" "----"

for f in "${env_files[@]}"; do
    model=$(basename "$f" .env)
    svc="${UNIT_PREFIX}${model}.service"

    # Active 状態
    state=$(systemctl is-active "$svc" 2>/dev/null)
    [[ -z "$state" ]] && state="unknown"

    # enable 状態（enabled/disabled/static）
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null)
    [[ -z "$enabled" ]] && enabled="unknown"

    # PORT 抜き出し
    port=$(grep -oP '(?<=PORT=)\d+' "$f" 2>/dev/null)
    [[ -z "$port" ]] && port="—"

    # API 疎通（active のみチェック）
    if [[ "$state" == "active" && "$port" != "—" ]]; then
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/vllm/health" 2>/dev/null)
        [[ -z "$code" ]] && code="---"
    else
        code="---"
    fi

    printf "%-25s %-10s %-10s %-8s %-10s\n" \
        "$model" "$state" "$enabled" "$port" "$code"
done

echo
echo "=== 完了 ==="
