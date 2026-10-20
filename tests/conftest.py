"""测试共享基建（工单 03 确立）。

pixretouch.config 在**模块导入时**固化 PROJECT_HOME 及全部派生路径（读环境变量），
而 pytest 保证 conftest 先于一切测试模块导入 → 在此把 PIXRETOUCH_HOME 指向
会话级临时目录，杜绝测试触碰仓库真实 var/（铁律 8）。

各测试内再用 monkeypatch 把 config.DB_PATH/CACHE_DIR/VAR_DIR/MODELS_DIR 指到
tmp_path 做逐测试隔离（范本见 test_capability_inpaint.isolated_home）；
engine.reset_session() 负责会话清场。
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_SESSION_HOME = Path(tempfile.mkdtemp(prefix="pixretouch-pytest-home-"))
os.environ["PIXRETOUCH_HOME"] = str(_SESSION_HOME)


def _cleanup() -> None:
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)


atexit.register(_cleanup)
