"""能力目录：注册表 + 权重规格。

每个能力工单（03 起）在此登记：
- CAPABILITIES[name]：渲染函数 + 参数 schema + 默认配方（数据驱动注册，见 GLOSSARY「能力目录」）
- WEIGHT_SPECS[name]：权重下载规格，供 doctor 盘点与 prefetch 后台预取

权重规格形如：
    weight("lama", {"repo": "smartywu/film-lama", "file": "model.pt",
                    "modelscope": None})   # modelscope 字段为兜底源 id，可为 None
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
