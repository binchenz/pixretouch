"""引擎会话与上下文（工单 03 确立的模式基建）。

- 会话 = SQLite 连接 + 内容寻址缓存，线程本地惰性单例：
  sqlite 连接不可跨线程（check_same_thread），而 FastMCP 可能在工作线程派发同步工具；
  WAL 模式支持多连接并发（ADR-0002），故按线程各持一份。
- EngineContext = 交给渲染函数的临时上下文（scratch 临时产物目录 + payload 预构建对象）。
- 能力模块与 server 工具都不自管连接，一律 get_session()。
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, data

_local = threading.local()
_init_lock = threading.Lock()


def get_session() -> tuple[sqlite3.Connection, data.ContentCache]:
    """当前线程的 (conn, cache)；首次调用时按 config 当前值惰性建库。"""
    sess = getattr(_local, "session", None)
    if sess is None:
        with _init_lock:
            config.ensure_dirs()
            conn = data.connect(config.DB_PATH)
            data.init_schema(conn)
            sess = (conn, data.ContentCache(conn, config.CACHE_DIR))
            _local.session = sess
    return sess


def reset_session() -> None:
    """关闭当前线程会话；下次 get_session() 按 config 当前值重建（测试钩子）。"""
    sess = getattr(_local, "session", None)
    if sess is not None:
        sess[0].close()
        _local.session = None


def png_sha256(img: Any) -> str:
    """PIL 图像的 PNG 字节 sha256（PIL PNG 编码无时间戳，同图确定）。

    mask/alpha/权重图类产物的规范哈希：进 records.mask_sha256 与缓存键（ADR-0002）。
    """
    import io

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


@dataclass
class EngineContext:
    """渲染函数上下文。

    scratch: 临时产物目录（工具层在 data.apply 后负责清理临时文件，缓存中已有固化副本）。
    payload: 工具层预构建的对象（如 mask），避免渲染函数重复计算。
    cache: ContentCache 引用（工具层赋值）——外部素材的**正规取用入口**：
        ctx.cache.path_for(素材内容 sha)。素材类 payload 只是省算快路径，
        渲染函数必须能仅凭 params 的内容 sha 经 cache 重建（lead 裁决 2026-10-21）。
    纪律：payload 不得影响输出确定性——仅凭 params 重建必须得到一致结果
    （缓存键只含 输入哈希/能力/params/mask_sha，见 ADR-0002）。
    """

    scratch: Path
    payload: dict[str, Any] = field(default_factory=dict)
    cache: Any = None

    def scratch_file(self, prefix: str, suffix: str) -> Path:
        self.scratch.mkdir(parents=True, exist_ok=True)
        return self.scratch / f"{prefix}_{uuid.uuid4().hex}{suffix}"


def get_context() -> EngineContext:
    return EngineContext(scratch=config.VAR_DIR / "scratch")
