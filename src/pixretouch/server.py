"""常驻 MCP 服务入口（ADR-0003）。

stdio 传输；stdout 是协议信道，任何日志只走 stderr。
基于 mcp 2.x：from mcp.server import MCPServer（v1 的 mcp.server.fastmcp 已移除）。
工具随能力工单逐个挂载：doctor（工单01）、blemish_remove（工单03）。
挂载模式（后续能力照抄）：from .capabilities import <name> 触发注册副作用，
@mcp.tool() 只做薄包装，逻辑全部在能力模块的 <tool>_op() 里。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server import MCPServer  # mcp 2.x API（v1 的 mcp.server.fastmcp 已移除）

from . import doctor as doctor_mod
from .capabilities import inpaint as inpaint_cap  # noqa: F401  能力注册副作用（工单 03）

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("pixretouch.server")

mcp = MCPServer("pixretouch")


@mcp.tool()
def doctor() -> dict[str, Any]:
    """环境健康自诊断：python/torch-MPS/核心库/磁盘余量/模型缓存/镜像源可达性，结构化报告。"""
    return doctor_mod.run_checks()


@mcp.tool()
def blemish_remove(
    image_id: int,
    points: list[list[int]] | None = None,
    boxes: list[list[int]] | None = None,
    brush: int = 10,
) -> dict[str, Any]:
    """污点修补（LaMa）：去除痘痘、路人、杂物；结果作为记录追加进记录链，可回滚。

    Args:
        image_id: 库内图像 ID（ingest 返回）。作用于当前态指针指向的产物。
        points: 提示点 [[x,y],...]，像素坐标。
        boxes: 提示框 [[x0,y0,x1,y1],...]。points/boxes 至少给一个。
        brush: 提示点半径（像素），1..64，默认 10。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit, mask_sha256}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return inpaint_cap.blemish_remove_op(image_id, points=points, boxes=boxes, brush=brush)


def main() -> None:
    log.info("pixretouch server starting (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
