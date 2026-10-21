"""相册级工具：verdict 覆写 / 联络表 / 导出（工单 13，lead 裁决 O1/O4/O5）。

定位：与 intake.py 同类的**数据/报告层工具**——非「能力」：不走渲染、不追加
记录链、不进内容寻址缓存（产物是报告与交付，不是图像状态）。

- set_verdicts（O1）：选片结论的**单一写入口**。verdict 持久化在
  images.meta['verdict'] = {value: keep|review|reject, source: report|human,
  reason?, ts}——不做 schema 迁移（SCHEMA_VERSION 不动）。事实源=selection_report
  （工单04，verdict ∈ keep/review/reject）；workflow S0 段在报告产出后以
  source="report" 调本工具持久化；用户自然语言推翻 = agent 解析后以
  source="human" 调本工具（spec 用户故事3）。fan-out（O3：keep→套用、review→
  套用+联络表打标、reject→跳过）/联络表/导出一律消费 meta 里的最终值。
  批量原子：整批成功或整批无痕（单事务）。
- contact_sheet（O4）：联络表（三交付物之二，HTML+PNG）——修图前后成对网格；
  verdict=review 打标（橙色框+REVIEW 标记）；reject 不入网格、单列清单。
  幂等：目录名 = 规范输入（图集+参数+各图当前/根输出哈希+verdict）sha256 前 12 位，
  同输入重生成直接返回既有目录（报告类内容寻址）。零新依赖（PIL 网格+极简 HTML）。
- export_album（O5）：导出目录（三交付物之三）——当前态产物按 {序号:03d}_{原 stem}{ext}
  规整命名复制进 var/exports/<album>/ + manifest.json（image_id/sha256/链路摘要/
  verdict）；默认跳过 verdict=reject；同名相册重导=覆盖刷新。
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, data
from .engine import get_session

log = logging.getLogger("pixretouch.album")

VERDICTS = ("keep", "review", "reject")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_meta(conn, image_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT meta FROM images WHERE id=?", (image_id,)).fetchone()
    if row is None:
        raise KeyError(f"图像 {image_id} 不存在")
    return json.loads(row["meta"] or "{}")


def _verdict_of(meta: dict[str, Any]) -> str | None:
    v = meta.get("verdict")
    return v.get("value") if isinstance(v, dict) else None


# ---------------------------------------------------------------- set_verdicts

def set_verdicts_op(items: list[dict], source: str = "human") -> dict[str, Any]:
    """批量写 images.meta['verdict']；单事务原子（一项非法整批无痕）。"""
    conn, _cache = get_session()
    try:
        if source not in ("report", "human"):
            raise ValueError(f"source 须为 'report' 或 'human'，收到 {source!r}")
        if not items:
            raise ValueError("items 不能为空")
        staged: list[tuple[int, str]] = []
        for it in items:
            image_id = int(it["image_id"])
            value = str(it["verdict"]).strip().lower()
            if value not in VERDICTS:
                raise ValueError(f"verdict 须为 keep/review/reject，收到 {it['verdict']!r}")
            meta = _get_meta(conn, image_id)
            entry: dict[str, Any] = {"value": value, "source": source, "ts": _now()}
            if it.get("reason"):
                entry["reason"] = str(it["reason"])
            meta["verdict"] = entry
            staged.append((json.dumps(meta, ensure_ascii=False, sort_keys=True), image_id))
        for meta_json, image_id in staged:
            conn.execute("UPDATE images SET meta=? WHERE id=?", (meta_json, image_id))
        conn.commit()
        return {"status": "ok", "count": len(staged),
                "updated": [{"image_id": i, "verdict": json.loads(m)["verdict"]["value"]} for m, i in staged]}
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        log.exception("set_verdicts 失败")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- contact_sheet

def _resolve_ids(conn, image_ids: list[int] | None) -> list[int]:
    if image_ids:
        return [int(i) for i in image_ids]
    rows = conn.execute("SELECT id FROM images ORDER BY id").fetchall()
    return [r["id"] for r in rows]


def _thumb(im, box: int) -> Any:
    """等比缩放到 box×box 内；RGBA 拼白底。"""
    from PIL import Image

    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((box, box))
    return im


def _pair_cell(before_p: Path, after_p: Path, caption: str, thumb: int, verdict: str | None) -> Any:
    """前后成对单元格图：before | after + 底栏 caption；review 橙框打标。"""
    from PIL import Image, ImageDraw

    with Image.open(before_p) as b, Image.open(after_p) as a:
        tb, ta = _thumb(b, thumb), _thumb(a, thumb)
    gap = 6
    cap_h = 22
    w = tb.width + gap + ta.width
    h = max(tb.height, ta.height) + cap_h
    cell = Image.new("RGB", (w + 8, h + 8), (250, 250, 250))
    cell.paste(tb, (4, 4 + (h - cap_h - tb.height) // 2))
    cell.paste(ta, (4 + tb.width + gap, 4 + (h - cap_h - ta.height) // 2))
    d = ImageDraw.Draw(cell)
    color = (230, 120, 0) if verdict == "review" else (60, 60, 60)
    d.text((6, h - cap_h + 6), caption[:60], fill=color)
    if verdict == "review":  # O3：review → 套用但联络表打标
        d.rectangle([1, 1, w + 6, h + 6], outline=(230, 120, 0), width=3)
    return cell


def contact_sheet_op(image_ids: list[int] | None = None, thumb: int = 256,
                     per_row: int = 6) -> dict[str, Any]:
    """联络表：sheet.png（网格）+ pairs/*.png（逐对）+ index.html。报告类内容寻址幂等。"""
    from PIL import Image

    conn, _cache = get_session()
    tmp_note: list[Path] = []
    try:
        thumb = max(64, min(int(thumb), 1024))
        per_row = max(1, min(int(per_row), 12))
        ids = _resolve_ids(conn, image_ids)
        if not ids:
            raise ValueError("库内无图像可出联络表")

        entries = []
        for iid in ids:
            meta = _get_meta(conn, iid)
            base = conn.execute(
                "SELECT output_path FROM records WHERE image_id=? AND parent_id IS NULL", (iid,)
            ).fetchone()
            cur = data.get_current(conn, iid)
            prow = conn.execute("SELECT path FROM images WHERE id=?", (iid,)).fetchone()
            entries.append({
                "image_id": iid, "verdict": _verdict_of(meta),
                "src": Path(prow["path"]), "before": Path(base["output_path"]),
                "after": Path(cur["output_path"]), "after_sha": cur["output_sha256"],
                "chain": [h["capability"] for h in data.history(conn, iid)],
            })

        canon = json.dumps({
            "ids": ids, "thumb": thumb, "per_row": per_row,
            "state": [{"i": e["image_id"], "a": e["after_sha"], "v": e["verdict"]} for e in entries],
        }, sort_keys=True, separators=(",", ":"))
        run_key = hashlib.sha256(canon.encode()).hexdigest()[:12]
        out_dir = config.VAR_DIR / "reports" / f"contact_{run_key}"
        sheet_png, index_html = out_dir / "sheet.png", out_dir / "index.html"
        stats = {
            "total": len(entries),
            "shown": sum(1 for e in entries if e["verdict"] != "reject"),
            "review_marked": sum(1 for e in entries if e["verdict"] == "review"),
            "rejected_listed": sum(1 for e in entries if e["verdict"] == "reject"),
        }
        if sheet_png.exists() and index_html.exists():
            return {"status": "ok", "cached": True, "sheet_png": str(sheet_png),
                    "index_html": str(index_html), "directory": str(out_dir), "stats": stats}

        out_dir.mkdir(parents=True, exist_ok=True)
        pairs_dir = out_dir / "pairs"
        pairs_dir.mkdir(exist_ok=True)

        shown = [e for e in entries if e["verdict"] != "reject"]
        cells = []
        for e in shown:
            caption = f"#{e['image_id']} {e['src'].stem} [{e['verdict'] or '-'}]"
            cell = _pair_cell(e["before"], e["after"], caption, thumb, e["verdict"])
            cell_path = pairs_dir / f"p{e['image_id']:04d}.png"
            cell.save(cell_path, "PNG")
            cells.append(cell)

        if cells:
            cw = max(c.width for c in cells)
            ch = max(c.height for c in cells)
            rows = (len(cells) + per_row - 1) // per_row
            sheet = Image.new("RGB", (per_row * (cw + 8) + 8, rows * (ch + 8) + 8), (40, 40, 40))
            for idx, c in enumerate(cells):
                r, col = divmod(idx, per_row)
                sheet.paste(c, (4 + col * (cw + 8), 4 + r * (ch + 8)))
            sheet.save(sheet_png, "PNG")
        else:
            Image.new("RGB", (64, 64), (40, 40, 40)).save(sheet_png, "PNG")

        esc = html.escape
        row_htmls = []
        for e in entries:
            # py3.11：f-string 表达式内不得含反斜杠 → cell 先行构建，属性用单引号
            if e["verdict"] != "reject":
                cell = "<img src='pairs/p%04d.png' height='160'>" % e["image_id"]
            else:
                cell = "(未处理)"
            row_htmls.append(
                f"<tr><td>{e['image_id']}</td><td>{esc(e['src'].name)}</td>"
                f"<td>{esc(e['verdict'] or '-')}</td><td>{esc(' → '.join(e['chain']))}</td>"
                f"<td>{cell}</td></tr>"
            )
        rows_html = "\n".join(row_htmls)
        index_html.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>contact sheet {run_key}</title></head><body>"
            f"<h1>PixRetouch 联络表 <small>{run_key}</small></h1>"
            f"<p>总计 {stats['total']} · 入网格 {stats['shown']} · review 打标 {stats['review_marked']}"
            f" · reject 单列 {stats['rejected_listed']} · 生成 {esc(_now())}</p>"
            "<p><img src='sheet.png' style='max-width:100%'></p>"
            "<table border='1' cellpadding='4'>"
            "<tr><th>id</th><th>文件</th><th>verdict</th><th>记录链</th><th>前 | 后</th></tr>"
            f"{rows_html}</table></body></html>",
            encoding="utf-8",
        )
        return {"status": "ok", "cached": False, "sheet_png": str(sheet_png),
                "index_html": str(index_html), "directory": str(out_dir), "stats": stats}
    except Exception as e:  # noqa: BLE001
        log.exception("contact_sheet 失败")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_note:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------- export_album

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")

_FMT_EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}


def _artifact_ext(p: Path) -> str:
    """产物扩展名：缓存产物是纯内容寻址哈希名（无扩展名）——按内容格式嗅探。"""
    from PIL import Image

    try:
        with Image.open(p) as im:
            ext = _FMT_EXT.get((im.format or "").upper())
    except Exception:  # noqa: BLE001
        ext = None
    return ext or p.suffix.lower() or ".bin"


def export_album_op(image_ids: list[int] | None = None, album_name: str | None = None) -> dict[str, Any]:
    """导出目录：当前态按 {i:03d}_{原stem}{ext} 复制 + manifest.json；默认跳过 reject。"""
    conn, _cache = get_session()
    try:
        ids = _resolve_ids(conn, image_ids)
        if not ids:
            raise ValueError("库内无图像可导出")
        album = str(album_name).strip() if album_name else f"album_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        if not album or "/" in album or "\\" in album or album.startswith("."):
            raise ValueError(f"album_name 非法：{album!r}")
        out_dir = config.EXPORTS_DIR / album
        out_dir.mkdir(parents=True, exist_ok=True)

        items, skipped_rejected = [], 0
        seq = 0
        for iid in ids:
            meta = _get_meta(conn, iid)
            verdict = _verdict_of(meta)
            if verdict == "reject":
                skipped_rejected += 1
                continue
            cur = data.get_current(conn, iid)
            src_p = Path(cur["output_path"])
            orig = Path(conn.execute("SELECT path FROM images WHERE id=?", (iid,)).fetchone()["path"])
            seq += 1
            stem = _SAFE_STEM.sub("_", orig.stem) or f"image_{iid}"
            dest = out_dir / f"{seq:03d}_{stem}{_artifact_ext(src_p)}"
            shutil.copy2(src_p, dest)
            items.append({
                "file": dest.name, "image_id": iid, "sha256": cur["output_sha256"],
                "verdict": verdict, "chain": [h["capability"] for h in data.history(conn, iid)],
                "source_path": str(orig),
            })
        manifest = {"album": album, "generated_at": _now(), "count": len(items),
                    "skipped_rejected": skipped_rejected, "items": items}
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"status": "ok", "directory": str(out_dir), "manifest_path": str(manifest_path),
                "count": len(items), "skipped_rejected": skipped_rejected}
    except Exception as e:  # noqa: BLE001
        log.exception("export_album 失败")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
