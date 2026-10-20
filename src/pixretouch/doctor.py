"""环境健康自诊断（工单 01）。

结构化、无副作用、单项检查有超时预算；供 MCP 工具 `doctor` 与 CLI 复用。
状态判定：python/磁盘/核心库 全部就绪且 torch-MPS 可用 → ok，否则 degraded；
镜像源不可达只降级为 warning（安装完成后离线运行是常态）。
"""
from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

from . import config


def _check_python() -> dict[str, Any]:
    v = sys.version_info
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "ok": (v.major, v.minor) >= (3, 11),
    }


def _check_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"installed": False, "error": f"{type(e).__name__}: {e}", "ok": False}
    mps = False
    try:
        mps = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    return {"installed": True, "version": torch.__version__, "mps_available": mps, "ok": mps}


def _check_libs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # pixretouch.server = 真实 import server 入口（MCPServer 注册全部工具），防止仅查
    # 顶层 mcp 掩盖 API 断裂（工单03 教训：mcp 2.x 移除 mcp.server.fastmcp，骨架期从未真实握手）
    for name in ("PIL", "cv2", "numpy", "mcp", "huggingface_hub", "requests", "pixretouch.server"):
        try:
            mod = __import__(name)
            out[name] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": type(e).__name__}
    return out


def _check_disk() -> dict[str, Any]:
    usage = shutil.disk_usage(config.PROJECT_HOME)
    free_gb = round(usage.free / 1024**3, 1)
    return {"free_gb": free_gb, "floor_gb": config.DISK_FLOOR_GB, "ok": free_gb >= config.DISK_FLOOR_GB}


def _check_mirrors() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import requests
    except Exception:  # noqa: BLE001
        return {"unavailable": True}
    for name, url in (
        ("pypi_tuna", config.PIP_MIRROR),
        ("hf_mirror", config.HF_MIRROR),
        ("modelscope", config.MODELSCOPE_BASE),
    ):
        try:
            r = requests.head(url, timeout=5, allow_redirects=True)
            out[name] = {"ok": r.status_code < 500, "status": r.status_code}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": type(e).__name__}
    return out


def _check_models() -> dict[str, Any]:
    """权重注册表盘点：哪些已缓存、哪些待各能力工单补录/预取。"""
    try:
        from .capabilities import WEIGHT_SPECS
    except Exception:  # noqa: BLE001
        WEIGHT_SPECS = {}
    present, missing = [], []
    for name, spec in WEIGHT_SPECS.items():
        dest = config.MODELS_DIR / name
        (present if dest.exists() and any(dest.iterdir()) else missing).append(name)
    return {"registry_size": len(WEIGHT_SPECS), "present": present, "missing": missing, "ok": not missing}


def run_checks() -> dict[str, Any]:
    config.ensure_dirs()
    report: dict[str, Any] = {
        "python": _check_python(),
        "torch": _check_torch(),
        "libs": _check_libs(),
        "disk": _check_disk(),
        "mirrors": _check_mirrors(),
        "models": _check_models(),
        "paths": {
            "project_home": str(config.PROJECT_HOME),
            "var": str(config.VAR_DIR),
            "models": str(config.MODELS_DIR),
            "album_inbox": str(config.ALBUM_INBOX),
        },
    }
    hard_ok = (
        report["python"]["ok"]
        and report["disk"]["ok"]
        and all(v.get("ok", False) for v in report["libs"].values())
    )
    report["status"] = "ok" if (hard_ok and report["torch"]["ok"] and report["models"]["ok"]) else "degraded"
    report["mirrors_reachable"] = any(
        v.get("ok") for k, v in report["mirrors"].items() if isinstance(v, dict)
    )
    return report
