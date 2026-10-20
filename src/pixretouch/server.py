"""常驻 MCP 服务入口（ADR-0003）。

stdio 传输；stdout 是协议信道，任何日志只走 stderr。
工具随能力工单逐个挂载，骨架期仅有 doctor。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import doctor as doctor_mod

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("pixretouch.server")

mcp = FastMCP("pixretouch")


@mcp.tool()
def doctor() -> dict[str, Any]:
    """环境健康自诊断：python/torch-MPS/核心库/磁盘余量/模型缓存/镜像源可达性，结构化报告。"""
    return doctor_mod.run_checks()


def main() -> None:
    log.info("pixretouch server starting (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
