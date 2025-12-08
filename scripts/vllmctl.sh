#!/bin/bash
# ============================================================
# vllmctl.sh (統合版)
# ------------------------------------------------------------
# vLLM モデル管理ユーティリティ
# 各種操作: start, stop, restart, enable, disable, check, list, env, config, fetch, remove
# - fetch: Hugging Face からモデルをダウンロード
# - remove: モデル削除
# ============================================================

source "$(dirname "$0")/_paths.sh"

#UNIT_DIR="/home/aiuser/systemd_units"
UNIT_DIR="${UNIT_DIR}"

ENV_DIR="${UNIT_DIR}/env"
#SCRIPTS_DIR="/home/aiuser/ai_envs/vllm_env/scripts"
SCRIPTS_DIR="${SCRIPTS_ROOT}"

print_help() {
    cat <<EOF
vllmctl - vLLM モデル管理コマンド

使用法:
  vllmctl <command> [model]

コマンド一覧:
  start <model>       モデルを起動
  stop [model]        モデルを停止 (指定なしで全停止)
  restart <model>     モデルを再起動
  enable <model>      サービスを有効化
  disable <model>     サービスを無効化
  check [--diag]      稼働状況を確認
  list                利用可能なモデル一覧を表示
  env <model>         環境ファイルの内容を表示
  config <model>      自動環境設定を実行 (auto_vllm_config 連携)
  fetch <repo> [name] Hugging Face からモデルを取得
  remove <model>      モデルを削除
  status              全モデルの稼働状況を表示
  help                このヘルプを表示
EOF
}

# --- 内部関数 ---
systemd_exec() {
    local cmd="$1"
    local model="$2"
    sudo systemctl "$cmd" "vllm@${model}.service"
}

case "$1" in
    start)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        systemd_exec start "$2"
        ;;
    stop)
        if [[ -n "$2" ]]; then
            systemd_exec stop "$2"
        else
            sudo systemctl stop 'vllm@*.service'
        fi
        ;;
    restart)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        systemd_exec restart "$2"
        ;;
    enable)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        systemd_exec enable "$2"
        ;;
    disable)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        systemd_exec disable "$2"
        ;;
    check)
        # vllmctl check / vllmctl check --diag / vllmctl check <model> --diag
        shift
        bash "${SCRIPTS_DIR}/check_vllm.sh" "$@"
        ;;
    list)
        ls "${ENV_DIR}"/*.env 2>/dev/null | xargs -n1 basename | sed 's/\.env$//'
        ;;
    env)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        cat "${ENV_DIR}/${2}.env"
        ;;
    config)
        [[ -z "$2" ]] && { echo "モデル名を指定してください。"; exit 1; }
        bash "${SCRIPTS_DIR}/auto_vllm_config.sh" "$2"
        ;;
    fetch)
        [[ -z "$2" ]] && { echo "モデルリポジトリ名を指定してください。"; exit 1; }
        bash "${SCRIPTS_DIR}/get_hf_model.sh" "$2" "$3"
        ;;
    remove)
        [[ -z "$2" ]] && { echo "削除するモデル名を指定してください。"; exit 1; }
        bash "${SCRIPTS_DIR}/rm_model.sh" "$2"
        ;;
    status)
        bash "${SCRIPTS_DIR}/status_vllm.sh"
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        print_help
        ;;
esac



