#!/usr/bin/env python3
"""
vLLM のログから "Using max model len XXXX" を拾って MAX_MODEL_LEN を取得するスクリプト。

使い方:
    python get_max_model_len.py <モデルパス or HF ID>

標準出力:
    数値だけ (例: 65536)

終了コード:
    0: 成功（max model len を取得できた）
    1: 失敗（ログから取得できなかった）
"""

import sys
import subprocess
import re
import time
from typing import Optional


LOG_PATTERN = re.compile(r"Using max model len (\d+)")


def run_and_capture_max_len(model: str, timeout: int = 600) -> Optional[int]:
    """
    別プロセスで vLLM を起動し、ログから max model len を抜き出す。
    """

    # 子プロセス側で走らせる Python コード
    # （単純に vLLM の LLM を初期化するだけ）
    child_code = f"""
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs

engine_args = EngineArgs(model={model!r})
LLM(**engine_args.__dict__)
"""

    # -u でバッファリングを切って、ログを即時に流してもらう
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    max_len: Optional[int] = None
    start_time = time.time()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # 必要ならデバッグ用に STDERR にそのまま流す
            # sys.stderr.write(line)
            m = LOG_PATTERN.search(line)
            if m:
                max_len = int(m.group(1))
                # 取得できたらすぐにプロセスを止める
                proc.terminate()
                break

            # タイムアウトチェック
            if time.time() - start_time > timeout:
                proc.terminate()
                break

    finally:
        try:
            # 終了を少しだけ待つ
            proc.wait(timeout=10)
        except Exception:
            # それでも終わらなければ kill
            proc.kill()

    return max_len


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: get_max_model_len.py <モデルパス or HF ID>", file=sys.stderr)
        return 1

    model = sys.argv[1]
    max_len = run_and_capture_max_len(model)

    if max_len is None:
        print(
            f"[ERROR] vLLM logs did not contain 'Using max model len ...' for model: {model}",
            file=sys.stderr,
        )
        return 1

    # auto_vllm_config から使いやすいように、数値だけ出す
    print(max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
