#!/bin/bash
# ============================================================
# auto_vllm_config.sh v6.5
# ------------------------------------------------------------
# ・config.json の内容を最優先で dtype / quantization / kv-cache-dtype に反映
# ・vLLM 非対応値 (例: fp16) のみ警告付きで安全フォールバック
# ・VRAM 推定を行い .env ファイルを生成
# ============================================================

set -euo pipefail

source "$(dirname "$0")/_paths.sh"


# ------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------
#ENV_DIR="/home/aiuser/systemd_units/env"
#MODEL_ROOT="${VLLM_MODEL_ROOT:-/home/aiuser/models}"

# MODEL_ROOT や ENV_DIR は _paths.sh で定義済み

DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_GPU_UTIL="0.90"
DEFAULT_MAX_LEN="None"
DEFAULT_NUM_SEQS="8"

# KV cache VRAM 見積りにだけ使う内部用の長さ（env には書かない）
DEFAULT_KV_EST_LEN=16384

GIB=$((1024 * 1024 * 1024))

# ------------------------------------------------------------
# GPU情報取得
# ------------------------------------------------------------
get_gpu_info() {
  local name mb
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
  echo "$name,$mb"
}

# ------------------------------------------------------------
# config.json 読み取り（dtype, kv_cache_dtype, quantization）
# ------------------------------------------------------------
read_model_config() {
  local cfg="$1/config.json"
  [[ ! -f "$cfg" ]] && { echo "[ERROR] config.json が見つかりません: $cfg"; exit 1; }

  local dtype kv_dtype quant hidden_size num_layers heads_kv head_dim model_type
  dtype=$(jq -r '.torch_dtype // .text_config.torch_dtype // "unknown"' "$cfg")
  kv_dtype=$(jq -r '.text_config.kv_cache_dtype // "auto"' "$cfg")
  quant=$(jq -r '.quantization_config.quantization_method // "None"' "$cfg")
  hidden_size=$(jq -r '.hidden_size // .text_config.hidden_size // 0' "$cfg")
  num_layers=$(jq -r '.text_config.num_hidden_layers // .num_hidden_layers // 0' "$cfg")
  heads_kv=$(jq -r '.text_config.num_key_value_heads // .text_config.num_attention_heads // 0' "$cfg")
  head_dim=$(jq -r '.text_config.head_dim // 0' "$cfg")
  model_type=$(jq -r '.model_type // "unknown"' "$cfg")

  echo "$dtype,$kv_dtype,$quant,$hidden_size,$num_layers,$heads_kv,$head_dim,$model_type"
}

# ------------------------------------------------------------
# bytes_per_param の算出
# ------------------------------------------------------------
bytes_per_param_from_dtype_quant() {
  local dtype="$1" quant="$2" bytes=4
  case "$dtype" in
    "float32"|"float") bytes=4 ;;
    "bfloat16"|"float16"|"half") bytes=2 ;;
    "fp8"|"fp8_e4m3"|"fp8_e5m2") bytes=1 ;;
    *) bytes=2 ;;
  esac
  case "$quant" in
    "awq"|"gptq"|"marlin"|"squeezellm") bytes=1 ;;
    "aqlm") bytes=0.5 ;;
  esac
  echo "$bytes"
}

# ------------------------------------------------------------
# safetensors 合計サイズ
# ------------------------------------------------------------
get_weights_bytes() {
  local dir="$1"
  local s=$(find "$dir" -type f -name "*.safetensors" -printf "%s\n" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
  [[ "${s:-0}" -gt 0 ]] && { echo "$s"; return; }
  du -sb "$dir" 2>/dev/null | awk '{print $1+0}'
}

# ------------------------------------------------------------
# KVキャッシュ 上限計算
# ------------------------------------------------------------
kv_bytes_upper_bound() {
  local L="$1" H_KV="$2" D="$3" kv_dtype="$4" max_len="$5" num_seqs="$6"
  local bpe=2
  case "$kv_dtype" in
    "bfloat16"|"bf16") bpe=2 ;;
    "fp8"|"fp8_e4m3"|"fp8_e5m2"|"fp8_inc") bpe=1 ;;
    "auto") bpe=2 ;;  # conservative estimate
  esac
  awk -v L="$L" -v H="$H_KV" -v D="$D" -v B="$bpe" -v T="$max_len" -v N="$num_seqs" \
      'BEGIN{print 2*L*H*D*B*T*N}'
}

# ------------------------------------------------------------
# モデルの MAX_MODEL_LEN を決定する
#   1. config.json からそれっぽいフィールドを jq で読む
#   2. ダメなら get_max_model_len.py で vLLM から推定
#   3. それでもダメなら "None" を返す
# ------------------------------------------------------------
get_max_model_len() {
  local model_dir="$1"
  local cfg="$model_dir/config.json"
  local max_len=""

  # 1) config.json から読む（jq がある場合）
  if command -v jq >/dev/null 2>&1 && [[ -f "$cfg" ]]; then
    max_len=$(jq -r '
      .max_model_len //
      .max_position_embeddings //
      .text_config.max_position_embeddings //
      .rope_scaling.max_position_embeddings //
      .n_positions //
      .max_sequence_length //
      .seq_length //
      "unset"
    ' "$cfg" 2>/dev/null || echo "unset")

    if [[ "$max_len" != "null" && "$max_len" != "unset" && "$max_len" =~ ^[0-9]+$ ]]; then
      echo "$max_len"
      return 0
    fi
  fi

  # 2) vLLM 側から推定（get_max_model_len.py）
  if [[ -n "${VLLM_ANALYZER_PYTHON:-}" && -x "${VLLM_ANALYZER_PYTHON}" \
     && -n "${VLLM_ANALYZER_SCRIPT:-}" && -f "${VLLM_ANALYZER_SCRIPT}" ]]; then
    echo "[auto_vllm_config] config.json から max length を取得できなかったため vLLM から推定します..." >&2
    if max_len="$("${VLLM_ANALYZER_PYTHON}" "${VLLM_ANALYZER_SCRIPT}" "${model_dir}" 2>/dev/null)"; then
      if [[ "$max_len" =~ ^[0-9]+$ ]]; then
        echo "$max_len"
        return 0
      fi
    fi
  fi

  # 3) どうしてもダメなら "None"
  echo "None"
}


# ------------------------------------------------------------
# .env ファイル生成
# ------------------------------------------------------------
generate_env_file() {
  local model="$1" dtype="$2" kv_dtype="$3" quant="$4" host="$5" port="$6" vram_util="$7" \
        max_len="$8" num_seqs="$9" weights_gib="${10}" kv_gib="${11}" vram_gib="${12}"
  mkdir -p "$ENV_DIR"
  local env_file="$ENV_DIR/${model}.env"

  cat >"$env_file" <<EOF
# ============================================================
# Auto-generated vLLM Environment File (v6.5)
# ============================================================
MODEL_PATH=${MODEL_ROOT}/${model}
HOST=${host}
PORT=${port}
DTYPE=${dtype}
KV_CACHE_DTYPE=${kv_dtype}
QUANTIZATION=${quant}
MAX_MODEL_LEN=${max_len}
GPU_MEMORY_UTILIZATION=${vram_util}
MAX_NUM_SEQS=${num_seqs}
TENSOR_PARALLEL_SIZE=1
# ------------------------------------------------------------
# 内訳 (GiB):
#   Weights: ${weights_gib}
#   KV Upper Bound (T=${max_len}, N=${num_seqs}, dtype=${kv_dtype}): ${kv_gib}
#   GPU VRAM: ${vram_gib}
# ------------------------------------------------------------
EOF
  echo "[OK] ${env_file} を生成しました。"
}

# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
generate_env() {
  local model="$1"
  [[ -z "$model" ]] && { echo "使い方: $0 <モデル名>"; exit 1; }

  local path="$MODEL_ROOT/$model"
  [[ ! -d "$path" ]] && { echo "[ERROR] モデルディレクトリが存在しません: $path"; exit 1; }

  echo "[INFO] モデル構成を解析中: $model"
  IFS=',' read -r gpu_name vram_mb <<<"$(get_gpu_info)"

  # ------------------------------------------------------------
  # VRAM の安全取得ロジック（DGX Spark / GB10 対策）
  # ------------------------------------------------------------
  # nvidia-smi が正常に値を返す場合はそのまま使う
  if [[ "$vram_mb" =~ ^[0-9]+$ ]]; then
      :
  else
      # Blackwell 系 (GB10 / GB100 / GB200...) は nvidia-smi が VRAM を返せない
      if [[ "$gpu_name" =~ ^NVIDIA.GB[0-9]+ ]]; then
          echo "[WARN] Blackwell GPU detected ($gpu_name). nvidia-smi VRAM=N/A → fallback to 128GB"
          vram_mb=$((128 * 1024))   # 128GB ← NVIDIA 公式仕様
      else
          echo "[WARN] nvidia-smi VRAM=N/A (GPU=$gpu_name) → fallback to safe default 80GB"
          vram_mb=$((80 * 1024))
      fi
  fi

  # バイト変換
  local vram_bytes=$((vram_mb * 1024 * 1024))
  local vram_gib=$(awk -v b="$vram_bytes" -v g="$GIB" 'BEGIN{printf "%.2f", b/g}')


  IFS=',' read -r dtype kv_dtype quant hidden num_layers heads_kv head_dim model_type <<<"$(read_model_config "$path")"

  # 無効値フォールバック
  [[ "$dtype" == "unknown" || -z "$dtype" ]] && dtype="bfloat16"
  if [[ "$kv_dtype" == "fp16" || "$kv_dtype" == "float16" ]]; then
    echo "[WARN] vLLM は --kv-cache-dtype=${kv_dtype} 非対応のため bfloat16 に変更します。" >&2
    kv_dtype="bfloat16"
  fi

  local host="$DEFAULT_HOST"
  local port="$DEFAULT_PORT"
  local gpu_util="$DEFAULT_GPU_UTIL"
  #local max_len="$DEFAULT_MAX_LEN"
  local max_len=$(get_max_model_len "$path")
  echo "[INFO] MAX_MODEL_LEN detected from config.json: ${max_len}"

  local num_seqs="$DEFAULT_NUM_SEQS"

  local bytes_per_param; bytes_per_param=$(bytes_per_param_from_dtype_quant "$dtype" "$quant")
  local weights_bytes; weights_bytes=$(get_weights_bytes "$path")
  local weights_gib=$(awk -v b="$weights_bytes" -v g="$GIB" 'BEGIN{printf "%.3f", b/g}')

  echo "[INFO] GPU: $gpu_name (${vram_gib} GiB VRAM)"
  echo "[INFO] dtype=${dtype}, kv_cache_dtype=${kv_dtype}, quant=${quant}"
  echo "[INFO] Weights: ${weights_gib} GiB"

  # KV cache VRAM 見積り用の長さ（数値が取れなければ内部デフォルトを使う）
  local kv_est_len="$max_len"
  if ! [[ "$kv_est_len" =~ ^[0-9]+$ ]]; then
    kv_est_len="$DEFAULT_KV_EST_LEN"
  fi

  local kv_bytes
  kv_bytes=$(kv_bytes_upper_bound \
      "$num_layers" "$heads_kv" "$head_dim" "$kv_dtype" \
      "$kv_est_len" "$num_seqs")

  local kv_gib=$(awk -v b="$kv_bytes" -v g="$GIB" 'BEGIN{printf "%.3f", b/g}')

  generate_env_file "$model" "$dtype" "$kv_dtype" "$quant" "$host" "$port" "$gpu_util" "$max_len" "$num_seqs" "$weights_gib" "$kv_gib" "$vram_gib"
}

# ------------------------------------------------------------
# ヘルプ表示
# ------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  cat <<'HELP'
auto_vllm_config.sh v6.5
-----------------------------------------------
config.json の内容を最優先で dtype / quantization / kv-cache-dtype に反映。
vLLM 非対応値 (fp16 など) は自動で警告・フォールバック。

使い方:
  auto_vllm_config.sh <モデル名>

出力:
  ${ENV_DIR}/<モデル名>.env

HELP
  exit 0
fi

generate_env "$1"

