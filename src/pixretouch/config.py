"""路径与镜像源配置。

隐私红线（ADR-0001 / README）：album-inbox、var/（库+缓存+导出）、models/（权重）
全部 gitignore，照片永不入仓库。
MCP stdio 纪律：任何模块不得向 stdout 打印（stdout 是协议信道），日志一律走 stderr。
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent                    # src/pixretouch
PROJECT_HOME = Path(os.environ.get("PIXRETOUCH_HOME", PKG_ROOT.parents[2]))

VAR_DIR = PROJECT_HOME / "var"
DB_PATH = VAR_DIR / "pixretouch.db"
CACHE_DIR = VAR_DIR / "cache"          # 内容寻址渲染缓存（ADR-0002）
RENDERS_DIR = VAR_DIR / "renders"
EXPORTS_DIR = VAR_DIR / "exports"
MODELS_DIR = PROJECT_HOME / "models"   # 模型权重缓存（gitignore）
ALBUM_INBOX = PROJECT_HOME / "album-inbox"

# 供给链：pip 走清华源；权重走 hf-mirror，ModelScope 兜底（工单 01 验收项）
PIP_MIRROR = os.environ.get("PIXRETOUCH_PIP_MIRROR", "https://pypi.tuna.tsinghua.edu.cn/simple")
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
MODELSCOPE_BASE = "https://www.modelscope.cn"

DISK_FLOOR_GB = 10  # doctor 判 degraded 的磁盘余量下限


def ensure_dirs() -> None:
    for d in (VAR_DIR, CACHE_DIR, RENDERS_DIR, EXPORTS_DIR, MODELS_DIR, ALBUM_INBOX):
        d.mkdir(parents=True, exist_ok=True)
