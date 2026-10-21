"""记录链查询与回退工具（库级，非能力——不渲染、不进能力目录）。

rollback_to / history 的 op 薄包装：data 层函数（工单 02 交付）→ 结构化错误面。
存在理由：记录链语义（回滚 = 移动当前态指针，Spec 用户故事 18）必须有 MCP 入口，
skill 质检回路的处置路径（history 定位 → rollback_to → 换参重做）与 workflow
联络表淘汰重修（用户故事 17）都消费它；qa 验证底座（mcp_client 模板）同样假定。

MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import logging
from typing import Any

from . import data
from .engine import get_session

log = logging.getLogger("pixretouch.chainops")


def history_op(image_id: int) -> dict[str, Any]:
    """history 工具实现：根 → 当前态的有序记录链。"""
    conn, _cache = get_session()
    try:
        hist = data.history(conn, int(image_id))
        if not hist:
            raise KeyError(f"image {image_id} has no current record")
        current = data.get_current(conn, int(image_id))["id"]
        return {"status": "ok", "image_id": int(image_id),
                "current_record_id": current, "records": hist}
    except Exception as e:  # noqa: BLE001
        log.exception("history(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def rollback_to_op(image_id: int, record_id: int) -> dict[str, Any]:
    """rollback_to 工具实现：移动当前态指针到目标记录（记录链只追加，不删除）。

    校验目标记录属于该图像（data 层 KeyError → 结构化返回）。
    """
    conn, _cache = get_session()
    try:
        cur = data.rollback_to(conn, int(image_id), int(record_id))
        return {"status": "ok", "image_id": int(image_id), "current": cur}
    except Exception as e:  # noqa: BLE001
        log.exception("rollback_to(image_id=%s, record_id=%s) 失败", image_id, record_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
