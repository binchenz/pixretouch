"""命令行入口：doctor / prefetch（调试与运维用；正式交互走 MCP 工具）。"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    p = argparse.ArgumentParser(prog="pixretouch")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="环境健康自诊断（JSON）")
    sub.add_parser("prefetch", help="后台预取模型权重（注册表驱动）")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        from . import doctor

        print(json.dumps(doctor.run_checks(), ensure_ascii=False, indent=1))
    elif args.cmd == "prefetch":
        from . import prefetch

        print(json.dumps(prefetch.fetch_all(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
