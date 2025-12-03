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

