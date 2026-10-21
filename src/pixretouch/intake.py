"""入库工具（工单 13 首步提前落，lead 待命期派单 2026-10-21）。

ingest：单图只读入库（幂等：sha256 去重，duplicate 标记）；
ingest_album：扫 album-inbox/ 或指定目录逐图入库，**失败隔离**（一张坏图不阻塞
整本，进 failed 清单），确定性顺序（按路径排序）。

定位：非「能力」——不走渲染/不追加编辑记录，仅库级登记（根记录 capability='base'
由 data.ingest 建，当前态指针指向根）。输入域（Spec 一期）：JPEG/TIFF；其他扩展名
skipped 记录在案。腐坏文件（PIL 无法解码）结构化拒绝，不入库——防污染能力目录
下游（assess/聚类都假设库内是可解码图像）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config, data
from .engine import get_session

log = logging.getLogger("pixretouch.intake")

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}  # Spec 一期输入域


def _resolve(path: str) -> Path:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = config.PROJECT_HOME / p
    return p


def _decode_check(p: Path) -> None:
    """真实解码校验：腐坏文件抛 ValueError（结构化错误面）。"""
    from PIL import Image

    try:
        with Image.open(p) as im:
            im.load()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"图像不可解码（{type(e).__name__}）：{p}") from e


def ingest_op(path: str) -> dict[str, Any]:
    """单图入库：原图只读、幂等去重。返回 image_id 供全部后续工具寻址。"""
    conn, _cache = get_session()
    try:
        p = _resolve(path)
        if not p.is_file():
            raise ValueError(f"文件不存在：{p}")
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"一期输入域为 JPEG/TIFF，收到 {p.suffix!r}：{p}")
        _decode_check(p)
        r = data.ingest(conn, p)  # 原图只读登记；重复 sha256 → duplicate=True 同 image_id
        return {"status": "ok", "path": str(p), **r}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("ingest(%s) 失败", path)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def ingest_album_op(directory: str | None = None) -> dict[str, Any]:
    """扫目录（默认 album-inbox/，递归）逐图入库；失败隔离 + 幂等去重标记。

    返回 {status, directory, total, ingested:[{path,image_id,sha256,duplicate}],
    failed:[{path,error}], skipped:[{path,reason}]}；total = ingested+failed
    （skipped 是非图像候选，不计入）。
    """
    conn, _cache = get_session()
    d = _resolve(directory) if directory else config.ALBUM_INBOX
    if not d.is_dir():
        return {"status": "error", "error": f"目录不存在：{d}"}
    result: dict[str, Any] = {
        "status": "ok", "directory": str(d),
        "ingested": [], "failed": [], "skipped": [],
    }
    for f in sorted(d.rglob("*")):  # 排序保证确定性顺序（缓存/报告可复现）
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            result["skipped"].append({"path": str(f), "reason": f"非 JPEG/TIFF（{f.suffix or '无扩展名'}）"})
            continue
        try:
            _decode_check(f)
            r = data.ingest(conn, f)
            result["ingested"].append({
                "path": str(f), "image_id": r["image_id"],
                "sha256": r["sha256"], "duplicate": r["duplicate"],
            })
        except Exception as e:  # noqa: BLE001  失败隔离：单图异常不阻塞整本
            log.warning("ingest_album 跳过 %s：%s", f, e)
            result["failed"].append({"path": str(f), "error": f"{type(e).__name__}: {e}"})
    result["total"] = len(result["ingested"]) + len(result["failed"])
    return result
