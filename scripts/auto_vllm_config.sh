#!/bin/bash
# ============================================================
# auto_vllm_config.sh v6.7
# ------------------------------------------------------------
# ・config.json の内容を最優先で dtype / quantization / kv-cache-dtype に反映
# ・vLLM 非対応値 (例: fp16) のみ警告付きで安全フォールバック
# ・VRAM 推定を行い .env ファイルを生成
# ・MTP / trust-remote-code / vLLM実行環境 / tiktoken cache を自動反映
# ・共存運用向けにGPU予約量とKV cache上限を自動反映
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
DEFAULT_NUM_SEQS="${VLLM_DEFAULT_NUM_SEQS:-1}"
DEFAULT_SAFE_MAX_LEN="${VLLM_SAFE_MAX_MODEL_LEN:-32768}"
DEFAULT_AUTO_TRUST_REMOTE_CODE="${VLLM_AUTO_TRUST_REMOTE_CODE:-1}"
DEFAULT_AUTO_MTP="${VLLM_AUTO_MTP:-1}"
DEFAULT_AUTO_FLASHINFER_AUTOTUNE="${VLLM_AUTO_FLASHINFER_AUTOTUNE:-0}"

# 共存運用向け。Style-Bert-VITS2 等のためにVRAMを残し、
# vLLMのKV cacheを必要量へ制限する。
DEFAULT_RESERVED_VRAM_GIB="${VLLM_RESERVED_VRAM_GIB:-32}"
DEFAULT_KV_CACHE_GIB="${VLLM_KV_CACHE_GIB:-8}"
DEFAULT_AUTO_KV_CACHE_BYTES="${VLLM_AUTO_KV_CACHE_BYTES:-1}"

# KV cache VRAM 見積りにだけ使う内部用の長さ（env には書かない）
DEFAULT_KV_EST_LEN=16384

GIB=$((1024 * 1024 * 1024))

gib_to_bytes() {
  local gib="$1"
  awk -v g="$gib" 'BEGIN{printf "%.0f", g * 1024 * 1024 * 1024}'
}

calc_gpu_util_with_reserve() {
  local vram_gib="$1"
  local reserve_gib="$2"

  awk -v total="$vram_gib" -v reserve="$reserve_gib" '
    BEGIN {
      if (total <= 0) {
        printf "0.90";
        exit;
      }
      util = (total - reserve) / total;
      if (util > 0.90) util = 0.90;
      if (util < 0.50) util = 0.50;
      printf "%.2f", util;
    }
  '
}

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

  local dtype kv_dtype quant hidden_size num_layers heads_kv head_dim model_type architectures mtp_layers
  dtype=$(jq -r '.torch_dtype // .text_config.torch_dtype // "unknown"' "$cfg")
  kv_dtype=$(jq -r '.text_config.kv_cache_dtype // "auto"' "$cfg")
  quant=$(jq -r '.quantization_config.quantization_method // "None"' "$cfg")
  hidden_size=$(jq -r '.hidden_size // .text_config.hidden_size // 0' "$cfg")
  num_layers=$(jq -r '.text_config.num_hidden_layers // .num_hidden_layers // 0' "$cfg")
  heads_kv=$(jq -r '.text_config.num_key_value_heads // .text_config.num_attention_heads // 0' "$cfg")
  head_dim=$(jq -r '.text_config.head_dim // 0' "$cfg")
  model_type=$(jq -r '.model_type // "unknown"' "$cfg")
  architectures=$(jq -r '(.architectures // .text_config.architectures // []) | join("+")' "$cfg")
  mtp_layers=$(jq -r '.mtp_num_hidden_layers // .text_config.mtp_num_hidden_layers // 0' "$cfg")

  echo "$dtype,$kv_dtype,$quant,$hidden_size,$num_layers,$heads_kv,$head_dim,$model_type,$architectures,$mtp_layers"
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
# MTPサポート推定
# ------------------------------------------------------------
detect_mtp_support() {
  local dir="$1" mtp_layers="$2"

  if [[ "${DEFAULT_AUTO_MTP}" != "1" ]]; then
    echo 0
    return 0
  fi

  if [[ "$mtp_layers" =~ ^[0-9]+$ && "$mtp_layers" -gt 0 ]]; then
    echo 1
    return 0
  fi

  if [[ -f "$dir/model.safetensors.index.json" ]] \
     && grep -q '"mtp\.' "$dir/model.safetensors.index.json"; then
    echo 1
    return 0
  fi

  echo 0
}

escape_env_arg() {
  # systemd EnvironmentFile の double-quoted value に入れるため、JSON内の " を escape する。
  printf '%s' "$1" | sed 's/"/\\"/g'
}

build_extra_args() {
  local trust_remote_code="$1" has_mtp="$2" flashinfer_autotune="$3" max_num_batched_tokens="$4" kv_cache_bytes="$5"
  local args=""

  if [[ "$trust_remote_code" == "1" ]]; then
    args="$args --trust-remote-code"
  fi

  if [[ "$has_mtp" == "1" ]]; then
    args="$args --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}"
  fi

  if [[ -n "$max_num_batched_tokens" && "$max_num_batched_tokens" != "None" ]]; then
    args="$args --max-num-batched-tokens ${max_num_batched_tokens}"
  fi

  if [[ -n "$kv_cache_bytes" && "$kv_cache_bytes" != "0" ]]; then
    args="$args --kv-cache-memory-bytes ${kv_cache_bytes}"
  fi

  if [[ "$flashinfer_autotune" == "1" ]]; then
    args="$args --enable-flashinfer-autotune"
  fi

  echo "${args# }"
}


# ------------------------------------------------------------
# .env ファイル生成
# ------------------------------------------------------------
generate_env_file() {
  local model="$1" dtype="$2" kv_dtype="$3" quant="$4" host="$5" port="$6" vram_util="$7" \
        max_len="$8" num_seqs="$9" weights_gib="${10}" kv_gib="${11}" vram_gib="${12}" extra_args="${13}" \
        model_type="${14}" architectures="${15}" has_mtp="${16}" raw_max_len="${17}" \
        reserved_vram_gib="${18}" kv_cache_gib="${19}" kv_cache_bytes="${20}"
  mkdir -p "$ENV_DIR"
  local env_file="$ENV_DIR/${model}.env"

  cat >"$env_file" <<EOF
# ============================================================
# Auto-generated vLLM Environment File (v6.7)
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

# Cache tiktoken encodings locally
TIKTOKEN_ENCODINGS_BASE=${REAL_HOME}/llm/vllm-spark/tiktoken_encodings

# vLLM 仮想環境と作業ディレクトリ
VLLM_ENV=${REAL_HOME}/llm/vllm-spark/.venv
VLLM_WORKDIR=${REAL_HOME}/llm/vllm-spark/vllm
VLLM_PYTHON=${REAL_HOME}/llm/vllm-spark/.venv/bin/python

# Auto-detected launch options
#   model_type=${model_type}
#   architectures=${architectures}
#   raw_max_model_len=${raw_max_len}
#   mtp_detected=${has_mtp}
#   reserved_vram_gib=${reserved_vram_gib}
#   kv_cache_gib=${kv_cache_gib}
#   kv_cache_bytes=${kv_cache_bytes}
EXTRA_ARGS="${extra_args}"
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


  IFS=',' read -r dtype kv_dtype quant hidden num_layers heads_kv head_dim model_type architectures mtp_layers <<<"$(read_model_config "$path")"

  # 無効値フォールバック
  [[ "$dtype" == "unknown" || -z "$dtype" ]] && dtype="bfloat16"
  if [[ "$kv_dtype" == "fp16" || "$kv_dtype" == "float16" ]]; then
    echo "[WARN] vLLM は --kv-cache-dtype=${kv_dtype} 非対応のため bfloat16 に変更します。" >&2
    kv_dtype="bfloat16"
  fi

  local host="$DEFAULT_HOST"
  local port="$DEFAULT_PORT"
  local gpu_util
  gpu_util=$(calc_gpu_util_with_reserve "$vram_gib" "$DEFAULT_RESERVED_VRAM_GIB")
  echo "[INFO] GPU_MEMORY_UTILIZATION selected=${gpu_util} (reserved=${DEFAULT_RESERVED_VRAM_GIB} GiB)"
  #local max_len="$DEFAULT_MAX_LEN"
  local raw_max_len=$(get_max_model_len "$path")
  local max_len="$raw_max_len"
  if [[ "$max_len" =~ ^[0-9]+$ && "$DEFAULT_SAFE_MAX_LEN" =~ ^[0-9]+$ && "$max_len" -gt "$DEFAULT_SAFE_MAX_LEN" ]]; then
    echo "[WARN] MAX_MODEL_LEN=${max_len} は初回起動には大きいため ${DEFAULT_SAFE_MAX_LEN} に制限します。raw=${max_len}" >&2
    max_len="$DEFAULT_SAFE_MAX_LEN"
  fi
  echo "[INFO] MAX_MODEL_LEN detected=${raw_max_len}, selected=${max_len}"

  local num_seqs="$DEFAULT_NUM_SEQS"

  local bytes_per_param; bytes_per_param=$(bytes_per_param_from_dtype_quant "$dtype" "$quant")
  local weights_bytes; weights_bytes=$(get_weights_bytes "$path")
  local weights_gib=$(awk -v b="$weights_bytes" -v g="$GIB" 'BEGIN{printf "%.3f", b/g}')

  echo "[INFO] GPU: $gpu_name (${vram_gib} GiB VRAM)"
  echo "[INFO] dtype=${dtype}, kv_cache_dtype=${kv_dtype}, quant=${quant}"
  echo "[INFO] model_type=${model_type}, architectures=${architectures}, mtp_layers=${mtp_layers}"
  echo "[INFO] Weights: ${weights_gib} GiB"

  local has_mtp; has_mtp=$(detect_mtp_support "$path" "$mtp_layers")
  local max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
  local kv_cache_bytes=""
  if [[ "${DEFAULT_AUTO_KV_CACHE_BYTES}" == "1" ]]; then
    kv_cache_bytes=$(gib_to_bytes "$DEFAULT_KV_CACHE_GIB")
  fi
  local extra_args_raw; extra_args_raw=$(build_extra_args "$DEFAULT_AUTO_TRUST_REMOTE_CODE" "$has_mtp" "$DEFAULT_AUTO_FLASHINFER_AUTOTUNE" "$max_num_batched_tokens" "$kv_cache_bytes")
  local extra_args; extra_args=$(escape_env_arg "$extra_args_raw")
  echo "[INFO] mtp_detected=${has_mtp}, kv_cache_gib=${DEFAULT_KV_CACHE_GIB}, kv_cache_bytes=${kv_cache_bytes}, EXTRA_ARGS=${extra_args_raw}"

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

  generate_env_file "$model" "$dtype" "$kv_dtype" "$quant" "$host" "$port" "$gpu_util" "$max_len" "$num_seqs" "$weights_gib" "$kv_gib" "$vram_gib" "$extra_args" "$model_type" "$architectures" "$has_mtp" "$raw_max_len" "$DEFAULT_RESERVED_VRAM_GIB" "$DEFAULT_KV_CACHE_GIB" "$kv_cache_bytes"
}

# ------------------------------------------------------------
# ヘルプ表示
# ------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  cat <<'HELP'
auto_vllm_config.sh v6.7
-----------------------------------------------
config.json の内容を最優先で dtype / quantization / kv-cache-dtype に反映。
vLLM 非対応値 (fp16 など) は自動で警告・フォールバック。
MTP重みを検出した場合、--speculative-config method=mtp num_speculative_tokens=1 を自動付与。

使い方:
  auto_vllm_config.sh <モデル名>

任意の環境変数:
  VLLM_SAFE_MAX_MODEL_LEN=32768
  VLLM_DEFAULT_NUM_SEQS=1
  VLLM_MAX_NUM_BATCHED_TOKENS=4096
  VLLM_AUTO_FLASHINFER_AUTOTUNE=1
  VLLM_AUTO_MTP=0
  VLLM_RESERVED_VRAM_GIB=32
  VLLM_KV_CACHE_GIB=8
  VLLM_AUTO_KV_CACHE_BYTES=1

出力:
  ${ENV_DIR}/<モデル名>.env

HELP
  exit 0
fi

generate_env "$1"

