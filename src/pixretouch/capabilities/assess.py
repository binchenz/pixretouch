"""选片质检（工单 04 / issue #5）：单图质量维度 + 相册级选片报告。

分析型能力：读当前态输出（data.get_current），维度合并写入 images.meta['assess']，
人脸经 faces.detect → faces.persist_faces 落 faces 表；不追加记录链、不产出渲染产物
（记录链只登记编辑操作，见 ADR-0002；选片本身不是编辑）。

四个质量维度（工单 04 验收）：
- 清晰度（无参考）：拉普拉斯方差，大图统一缩放到 WORK_SIZE 工作尺度后计算
  （只缩不放；原生最长边 < MIN_NATIVE_LONGEST 直接判「分辨率不足」——
  上采样平滑会产生假低清晰度，语义也不是失焦）；
- 曝光：检出人脸时以脸部区域平均灰度为主信号（白背景/暗背景不淘汰素材），
  全局均值与高光溢出比例作辅助；
- 闭眼：faces.EAR（双眼均低于阈值 → 闭眼；单眼 → wink 复核）；
- 人脸可检出：检出数量 + 最大脸面积占比 + 最大脸绝对像素短边。

选片报告（GLOSSARY「选片」）：相册级 JSON，每张 keep/review/reject + 可读理由，
按 verdict+score 排序；落盘 var/reports/，是 workflow（工单 13）三交付物之一。

阈值是经验起点（skill 决策树工单 12 会引用），全部模块级常量，可按验收相册校准。
MCP stdio 纪律：本模块不 print；日志一律 logging→stderr。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .. import config, data
from ..engine import get_session
from . import faces as faces_mod

log = logging.getLogger("pixretouch.assess")

WORK_SIZE = 1024            # 清晰度/曝光统一工作尺度（最长边，只缩不放）
MIN_NATIVE_LONGEST = 512    # 原生最长边低于此 → 分辨率不足 reject（语义是像素不够，不是失焦）
SHARP_FLOOR = 60.0          # 拉普拉斯方差低于此 → 模糊 reject
SHARP_REVIEW = 110.0        # floor~review → 复核
EXPOSURE_LOW, EXPOSURE_HIGH = 45.0, 210.0          # 曝光硬界 → reject（有人脸时按脸部曝光判）
EXPOSURE_LOW_SOFT, EXPOSURE_HIGH_SOFT = 70.0, 195.0  # 软界 → review
CLIP_HI_FRACTION = 0.35     # 全局高光溢出比例超过此且脸部亦偏亮 → review（白背景合法高溢出，不单独淘汰）
FACE_MIN_AREA = 0.004       # 最大脸面积占比低于此 → review（人脸过小）
FACE_SIDE_REJECT = 60.0     # 最大脸短边（原生像素）低于此 → reject（分辨率不足，无修图价值）
FACE_SIDE_REVIEW = 110.0    # 低于此 → review（脸部像素偏少，磨皮/增强收益有限）

_VERDICT_SEVERITY = {"keep": 0, "review": 1, "reject": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pixel_dimensions(image_path: Path) -> dict[str, Any]:
    """清晰度（无参考）与曝光维度；统一工作尺度，返回可直接落库的 dict。"""
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"not a readable image: {image_path}")
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > WORK_SIZE:
        # 统一工作尺度（只缩不放，INTER_AREA 抗混叠）：大图之间拉普拉斯方差可比。
        # 小图不上采样——插值平滑会把清晰度打到假低位（320px 锐利裁切实测被打到
        # ~10）；「分辨率不足」语义由 MIN_NATIVE_LONGEST 与人脸绝对像素规则负责。
        scale = WORK_SIZE / longest
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "width": w,
        "height": h,
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "exposure_mean": round(float(gray.mean()), 1),
        "clip_hi": round(float((gray >= 250).mean()), 4),
        "clip_lo": round(float((gray <= 5).mean()), 4),
    }


def face_exposure(image_path: Path, bbox: Sequence[int]) -> float | None:
    """脸部区域（bbox 外扩 10%）平均灰度——人像选片的曝光主信号。

    白背景棚拍 / 暗调人像的全局均值会系统性偏离主体曝光，脸区均值才是
    「这个人拍准了没有」。
    """
    import cv2

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    H, W = gray.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    mx, my = int(w * 0.1), int(h * 0.1)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    if x1 <= x0 or y1 <= y0:
        return None
    return round(float(gray[y0:y1, x0:x1].mean()), 1)


def assess_image(
    conn: sqlite3.Connection,
    image_id: int,
    *,
    detect_faces: bool = True,
    max_faces: int = 8,
) -> dict[str, Any]:
    """单图选片质检：维度落 images.meta['assess']，人脸落 faces 表；返回维度 dict。

    detect_faces=True 时模型缺失/下载失败抛 RuntimeError（结构化，不静默降级——
    agent 必须知道人脸维度缺失而不是拿到假阴性）。
    """
    cur = data.get_current(conn, image_id)
    path = Path(cur["output_path"])
    dims = pixel_dimensions(path)

    result: dict[str, Any] = {**dims, "face_detection": detect_faces}
    if detect_faces:
        detected = faces_mod.detect(path, max_faces=max_faces)
        # 对账式落库（issue #17）：保留 face_id / cluster_id 归属，复检不摧毁按人寻址
        face_audit = faces_mod.persist_faces(
            conn, image_id, detected,
            canvas=(dims["width"], dims["height"]), image_path=path)
        img_area = max(1, dims["width"] * dims["height"])
        max_area = max((f.bbox[2] * f.bbox[3] for f in detected), default=0)
        max_face_side = max((min(f.bbox[2], f.bbox[3]) for f in detected), default=0)
        if detected:
            biggest = max(detected, key=lambda f: f.bbox[2] * f.bbox[3])
            fe = face_exposure(path, biggest.bbox)
            if fe is not None:
                result["face_exposure_mean"] = fe
        result.update({
            "face_count": len(detected),
            "max_face_area": round(max_area / img_area, 5),
            "max_face_side": max_face_side,
            "faces": [f.summary() for f in detected],
            "eyes_closed": any(f.eyes_closed() for f in detected),
            "wink": any(f.wink() for f in detected),
            "ear_threshold": faces_mod.EAR_CLOSED_THRESHOLD,
            # 归属迁移审计（issue #17）：消失/新增的脸不静默，agent 可据此判断
            "faces_audit": face_audit,
        })
    result["assessed_at"] = _now()

    row = conn.execute("SELECT meta FROM images WHERE id=?", (image_id,)).fetchone()
    if row is None:
        raise KeyError(f"image {image_id} not found")
    meta = json.loads(row["meta"] or "{}")
    meta["assess"] = result
    conn.execute(
        "UPDATE images SET meta=? WHERE id=?",
        (json.dumps(meta, ensure_ascii=False, sort_keys=True), image_id),
    )
    conn.commit()
    return {"image_id": image_id, "assess": result}


# ------------------------------------------------------------------ 选片报告
def _downgrade(current: str, new: str) -> str:
    return new if _VERDICT_SEVERITY[new] > _VERDICT_SEVERITY[current] else current


def _verdict(a: dict[str, Any]) -> tuple[str, list[str], float]:
    """由 assess 维度得出 取舍结论 + 可读理由 + 排序分（0~1，越高越优）。"""
    verdict = "keep"
    reasons: list[str] = []

    sharp = float(a.get("sharpness", 0.0))
    longest = max(int(a.get("width", 0)), int(a.get("height", 0)))
    if longest and longest < MIN_NATIVE_LONGEST:
        verdict = _downgrade(verdict, "reject")
        reasons.append(f"分辨率不足：原生最长边 {longest}px < {MIN_NATIVE_LONGEST}px，细节不足以支撑修图")
    elif sharp < SHARP_FLOOR:
        verdict = _downgrade(verdict, "reject")
        reasons.append(f"模糊：清晰度 {sharp:.0f} 低于下限 {SHARP_FLOOR:.0f}（无参考拉普拉斯方差，工作尺度 {WORK_SIZE}）")
    elif sharp < SHARP_REVIEW:
        verdict = _downgrade(verdict, "review")
        reasons.append(f"清晰度偏低（{sharp:.0f} < {SHARP_REVIEW:.0f}），建议复核")

    # 曝光主信号：检出人脸时按脸部区域均值（背景过/欠曝不淘汰素材），否则全局
    face_mean = a.get("face_exposure_mean")
    if face_mean is not None:
        mean, expo_scope = float(face_mean), "脸部"
    else:
        mean, expo_scope = float(a.get("exposure_mean", 128.0)), "全局"
    clip_hi = float(a.get("clip_hi", 0.0))
    if mean < EXPOSURE_LOW:
        verdict = _downgrade(verdict, "reject")
        reasons.append(f"曝光严重不足：{expo_scope}平均灰度 {mean:.0f} < {EXPOSURE_LOW:.0f}")
    elif mean > EXPOSURE_HIGH:
        verdict = _downgrade(verdict, "reject")
        reasons.append(f"曝光过度：{expo_scope}平均灰度 {mean:.0f} > {EXPOSURE_HIGH:.0f}")
    elif mean < EXPOSURE_LOW_SOFT or mean > EXPOSURE_HIGH_SOFT:
        verdict = _downgrade(verdict, "review")
        reasons.append(f"{expo_scope}曝光轻度偏离（平均灰度 {mean:.0f}），建议复核")
    if clip_hi > CLIP_HI_FRACTION and mean > EXPOSURE_HIGH_SOFT:
        verdict = _downgrade(verdict, "review")
        reasons.append(f"高光溢出：{clip_hi:.0%} 像素过曝且{expo_scope}偏亮（灰度 {mean:.0f}），建议复核")

    if a.get("face_detection"):
        count = int(a.get("face_count", 0))
        area = float(a.get("max_face_area", 0.0))
        side = float(a.get("max_face_side", 0.0))
        # 模糊/超低分辨率时 landmark 漂移，EAR 结论不可靠，不出眼部理由
        sharp_ok = sharp >= SHARP_FLOOR and longest >= MIN_NATIVE_LONGEST
        if count == 0:
            verdict = _downgrade(verdict, "reject")
            reasons.append("未检出人脸（人像相册选片不可用）")
        else:
            if sharp_ok and a.get("eyes_closed"):
                verdict = _downgrade(verdict, "reject")
                reasons.append(f"闭眼：至少一人双眼 EAR 低于阈值 {a.get('ear_threshold')}")
            elif sharp_ok and a.get("wink"):
                verdict = _downgrade(verdict, "review")
                reasons.append("疑似单眼闭合，建议复核")
            if side and side < FACE_SIDE_REJECT:
                verdict = _downgrade(verdict, "reject")
                reasons.append(f"分辨率不足：最大脸短边仅 {side:.0f}px（< {FACE_SIDE_REJECT:.0f}px），无修图价值")
            elif side and side < FACE_SIDE_REVIEW:
                verdict = _downgrade(verdict, "review")
                reasons.append(f"脸部像素偏少（短边 {side:.0f}px），修图收益有限，建议复核")
            elif area < FACE_MIN_AREA:
                verdict = _downgrade(verdict, "review")
                reasons.append(f"人脸过小（最大脸占画面 {area:.2%}），建议复核")

    sharp_score = min(sharp / SHARP_REVIEW, 1.5) / 1.5
    expo_score = 1.0 - min(abs(mean - 128.0) / 128.0, 1.0)
    if a.get("face_detection"):
        face_score = min(float(a.get("max_face_area", 0.0)) / 0.05, 1.0)
        score = 0.5 * sharp_score + 0.3 * expo_score + 0.2 * face_score
    else:
        score = 0.65 * sharp_score + 0.35 * expo_score
    if verdict == "reject":
        score *= 0.5

    # 票面口径「每张含取舍结论与可读理由」：keep 也给正向理由（供人审阅选片报告）
    if verdict == "keep" and not reasons:
        parts = [f"清晰度 {sharp:.0f}（≥{SHARP_REVIEW:.0f}）", f"{expo_scope}曝光正常（均值 {mean:.0f}）"]
        if a.get("face_detection") and count:
            parts.append("双眼睁开（EAR 高于阈值）")
            if side:
                parts.append(f"最大脸短边 {side:.0f}px 满足修图要求")
        reasons.append("可留：" + "；".join(parts))
    return verdict, reasons, score


def selection_report(
    conn: sqlite3.Connection,
    image_ids: Sequence[int] | None = None,
    *,
    reports_dir: Path | None = None,
    save: bool = True,
    detect_faces: bool = True,
) -> dict[str, Any]:
    """相册级选片报告：每张 取舍结论 + 可读理由；JSON 落盘并返回。

    image_ids=None → 全库。未质检的图自动补测；detect_faces 控制补测是否含
    人脸检出（False → 仅像素维度，供无模型场景与单元测试）。
    排序：keep → review → reject，同结论按 score 降序（选片主序）。
    """
    if image_ids is None:
        image_ids = [r["id"] for r in conn.execute("SELECT id FROM images ORDER BY id").fetchall()]

    entries: list[dict[str, Any]] = []
    for iid in image_ids:
        row = conn.execute("SELECT sha256, meta FROM images WHERE id=?", (iid,)).fetchone()
        if row is None:
            raise KeyError(f"image {iid} not found")
        meta = json.loads(row["meta"] or "{}")
        a = meta.get("assess")
        if a is None:
            a = assess_image(conn, iid, detect_faces=detect_faces)["assess"]
        verdict, reasons, score = _verdict(a)
        entries.append({
            "image_id": iid,
            "sha256": row["sha256"][:12],
            "verdict": verdict,
            "score": round(score, 4),
            "reasons": reasons,
            "assess": a,
        })

    entries.sort(key=lambda e: (_VERDICT_SEVERITY[e["verdict"]], -e["score"]))
    counts = {v: sum(1 for e in entries if e["verdict"] == v) for v in ("keep", "review", "reject")}
    report: dict[str, Any] = {
        "generated_at": _now(),
        "album": {"total": len(entries), **counts},
        "images": entries,
    }
    if save:
        rdir = Path(reports_dir) if reports_dir is not None else config.VAR_DIR / "reports"
        rdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = rdir / f"{stamp}-selection_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        report["report_path"] = str(out)
    return report


# ------------------------------------------------------------------ 工具实现（接缝）
# 沿用工单 03 确立的标准挂接模式（范本 capabilities/inpaint.py docstring）：
# server.py 只挂 @mcp.tool() 薄包装，逻辑全在本节 <tool>_op()；结构化错误不抛穿 MCP。
# 选片质检是分析型能力：不渲染、不追加记录链，故不进 CAPABILITIES 渲染注册表，
# 只登记 WEIGHT_SPECS（faces.py）；DSH 侧工具名 mcp__pixretouch__<tool>。

def assess_image_op(image_id: int) -> dict[str, Any]:
    """assess_image 工具实现：单图质量维度落 images.meta + 人脸落 faces 表。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **assess_image(conn, int(image_id))}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约），完整栈走 stderr
        log.exception("assess_image(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def detect_faces_op(image_id: int) -> dict[str, Any]:
    """detect_faces 工具实现：重检人脸+landmark 并对账写 faces 表（供聚类前刷新）。

    对账语义（issue #17）：命中的旧行原地更新 → cluster_id / embedding / face_id
    保留；结果附 faces_audit（updated/inserted/deleted_face_ids/links_inherited）。
    """
    conn, _cache = get_session()
    try:
        cur = data.get_current(conn, int(image_id))
        detected = faces_mod.detect(Path(cur["output_path"]))
        audit = faces_mod.persist_faces(
            conn, image_id, detected, image_path=Path(cur["output_path"]))
        return {"status": "ok", "image_id": image_id, "face_count": len(detected),
                "faces": [f.summary() for f in detected], "faces_audit": audit}
    except Exception as e:  # noqa: BLE001
        log.exception("detect_faces(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def selection_report_op(image_ids: list[int] | None = None) -> dict[str, Any]:
    """selection_report 工具实现：相册级选片报告（未质检自动补测，JSON 落 var/reports/）。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **selection_report(conn, image_ids)}
    except Exception as e:  # noqa: BLE001
        log.exception("selection_report(image_ids=%s) 失败", image_ids)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
