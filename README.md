# vllm-control

This repository contains operational scripts and systemd units
for managing multiple vLLM instances on DGX Spark.

## Directory structure

scripts/          - operational CLI tools (start_vllm.sh, stop_vllm.sh, vllmctl.sh, ...)
systemd_units/    - systemd unit templates and env files (env/ excluded from git)
.gitignore        - excludes .venv and env/ (critical)
.venv/            - local virtual environment (not versioned)

## Not included in Git

- .venv/
- systemd_units/env/*.env (model-specific configs)
- logs/

## Usage

Add `~/control/vllm/scripts` to PATH, then:

    vllmctl start <model>
    vllmctl stop <model>
    start_vllm.sh start <model>

## Auto config knobs

`vllmctl config <model>` generates a model-specific env file under
`systemd_units/env/`.

The auto config script is intended to choose safe first-boot values for vLLM on
DGX Spark, while leaving room for companion GPU services such as
Style-Bert-VITS2.

Useful environment variables:

~~~bash
# Keep VRAM for other local GPU services such as Style-Bert-VITS2.
# On a 128 GiB DGX Spark, 32 GiB reserve gives GPU_MEMORY_UTILIZATION=0.75.
VLLM_RESERVED_VRAM_GIB=32

# Explicit KV cache cap for vLLM. Default target is 8 GiB.
# This is usually enough for MAX_MODEL_LEN=32768 and MAX_NUM_SEQS=1.
VLLM_KV_CACHE_GIB=8

# Disable explicit KV cache cap if needed.
VLLM_AUTO_KV_CACHE_BYTES=0

# Safe first-boot model length.
VLLM_SAFE_MAX_MODEL_LEN=32768

# Safe first-boot concurrency.
VLLM_DEFAULT_NUM_SEQS=1

# Optional scheduler token budget. Useful for experiments with batching,
# long prefill, or speculative decoding, but not always beneficial for
# single-request generation.
VLLM_MAX_NUM_BATCHED_TOKENS=4096

# Optional experiment only. This runs FlashInfer autotune at server startup.
VLLM_AUTO_FLASHINFER_AUTOTUNE=1

# Disable automatic MTP speculative decoding if needed.
VLLM_AUTO_MTP=0
~~~

Example: shared-GPU operation with Style-Bert-VITS2:

~~~bash
VLLM_RESERVED_VRAM_GIB=32 \
VLLM_KV_CACHE_GIB=8 \
./scripts/vllmctl.sh config huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated
~~~

Expected generated settings on a 128 GiB DGX Spark:

~~~ini
GPU_MEMORY_UTILIZATION=0.75
EXTRA_ARGS="--trust-remote-code --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1} --kv-cache-memory-bytes 8589934592"
~~~

Notes:

- `GPU_MEMORY_UTILIZATION` controls the vLLM instance memory budget.
- `--kv-cache-memory-bytes` is a more direct cap for KV cache memory.
- For shared-GPU operation, prefer explicit KV cache control over a fixed
  `GPU_MEMORY_UTILIZATION=0.90`.
- `num_speculative_tokens=1` was the best MTP setting observed for the tested
  Qwen3.5/Qwen3.6 MoE model.
- `num_speculative_tokens=2` was slower in testing because second-token
  acceptance rate dropped.
- FlashInfer autotune worked, but did not show a clear speed gain in the
  single-request benchmark.
