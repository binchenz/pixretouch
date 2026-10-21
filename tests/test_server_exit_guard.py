"""server 进程生命周期兜底测试（B 类挂起防护，lead 裁决 2026-10-21）。

外部行为断言：_exit_grace_reaper(grace) 在解释器被非 daemon 线程卡住无法自然退出时，
于宽限期后强制 os._exit(0)——用子进程真实复现「第三方原生线程卡退出」场景：
起一个 sleep(999) 的非 daemon 线程（等价于历史 mediapipe executor atexit join 挂起），
主流程调用 reaper 后结束；无兜底则进程挂 999s，有兜底则 ~grace 后 rc=0 退出。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def test_exit_guard_reaps_hung_interpreter():
    code = (
        "import threading, time, sys;"
        f"sys.path.insert(0, {str(REPO_SRC)!r});"
        "from pixretouch.server import _exit_grace_reaper;"
        "threading.Thread(target=lambda: time.sleep(999), name='stuck-native').start();"
        "_exit_grace_reaper(0.5);"
    )
    env = dict(os.environ)
    env.setdefault("PIXRETOUCH_HOME", os.path.join(os.sep, "tmp", "pixretouch-exit-guard-test"))
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, timeout=30, env=env,
    )
    elapsed = time.time() - t0
    assert r.returncode == 0, f"兜底应以 rc=0 强制退出: stderr={r.stderr[-300:]!r}"
    assert elapsed < 10, f"应在宽限期附近退出而非挂死, 实际 {elapsed:.1f}s"
