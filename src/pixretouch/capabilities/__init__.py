"""能力目录：注册表 + 权重规格。

每个能力工单（03 起）在此登记：
- CAPABILITIES[name]：渲染函数 + 参数 schema + 默认配方（数据驱动注册，见 GLOSSARY「能力目录」）
- WEIGHT_SPECS[name]：权重下载规格，供 doctor 盘点与 prefetch 后台预取

权重规格形如（已验证的真实示例见 capabilities/inpaint.py）：
    weight("lama", {"repo": "okaris/simple-lama", "file": "big-lama.pt",
                    "kind": "file",            # "file" 单文件（默认）| "repo" 整仓库
                    "modelscope": "iic/cv_fft_inpainting_lama"})  # 兜底源 id，可为 None
"""
from __future__ import annotations

from typing import Any, Callable

CAPABILITIES: dict[str, dict[str, Any]] = {}
WEIGHT_SPECS: dict[str, dict[str, Any]] = {}


def capability(name: str, **meta: Any) -> Callable:
    """注册一项能力：被装饰函数即渲染入口 fn(image, params, ctx) -> 产物路径。"""

    def deco(fn: Callable) -> Callable:
        CAPABILITIES[name] = {"fn": fn, **meta}
        return fn

    return deco


def weight(name: str, spec: dict[str, Any]) -> None:
    """登记权重下载规格（hf-mirror 优先，ModelScope 兜底）。"""
    WEIGHT_SPECS[name] = spec


def load_all() -> None:
    """导入本包全部能力模块，触发 @capability/weight 注册副作用。

    供 prefetch/doctor 盘点全量注册表（server.py 挂工具时显式 import 所用模块即可，
    运维路径用本函数）。注意：共享 checkout 下若某能力模块正在编辑中导致 import
    失败，本函数会抛错——这是有意的，不吞错。
    """
    import importlib
    import pkgutil

    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
