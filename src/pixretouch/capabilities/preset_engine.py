"""预设方案事务化编排（工单 11 / issue #12）。

apply_preset = 配方展开为执行计划 → **逐段渲染只入内容寻址缓存（零库写）** →
全部成功后单事务一次性追加全部记录 + 移动当前态指针。中途任何异常 →
零记录残留、当前态不动（整体生效或整体无痕，lead 约束 2026-10-21）；
重跑时段段缓存命中，第二次极快（内容寻址缓存 = 天然断点续作）。

单图事务：批量粒度归 workflow fan-out（工单 13，每图一次工具调用，天然隔离
超时与失败）——DSH toolCallTimeoutMs=600s 预算内单图 ≤5 步 × 秒级实测安全。

按人路由（票面「聚类默认方案在批量应用时自动按人路由」）：
- 图级应用（cluster_id=None）：person 域步骤按人物聚类分组展开（确定性顺序：
  cluster_id 升序，未聚类组最后）；聚类绑定了 default_suit_id → 该组用**绑定
  方案的 person 域步骤**（绑定方案的 image 域步骤不参与路由——全图调色不可
  按人碎切），未绑定/未聚类 → 请求方案的 person 域步骤。
- 指定 cluster_id：单聚类覆盖应用（Spec 用户故事 8「这张里的孩子用更轻的
  磨皮」），只执行该聚类的 person 域步骤，image 域步骤跳过。
- 链上顺序（确定性）：全部 person 段（组间按聚类序、组内按配方序）→ image 段
  （按配方序）。种子方案 person 先于 image 的编排意图由此保持。

渲染接缝：直接消费能力目录的 @capability 渲染函数 fn(image, params, ctx)——
「渲染函数不碰库、仅凭 params+payload 可重建」的工单 03 纪律在此是硬依赖；
mask/payload 由本引擎按步骤预构建（与单能力 op 同一构建代码，保证
mask_sha/缓存键与直接调用单能力时一致）。

straight 基准：零步骤配方 → 零渲染零记录，输出哈希恒等于输入当前态
（工单 11 验收锚点：方案引擎自身不引入任何画质损失）。

无人脸图的两分法（dev-a 语义裁决 2026-10-21）：
- 图级应用（cluster_id=None）零人脸 → person 域整体跳过 + image 域照常执行 +
  返回 person_skipped 审计字段。套系语义 =「可应用的都应用」：natural 的
  轻暖调对风景照依然成立；真实相册必含风景/静物，整单失败是坏语义。
- 显式 cluster_id 寻址而该聚类在本图无脸 → 维持结构化报错（ValueError）：
  单聚类覆盖是「特判这个人」的显式意图，静默跳过会掩盖寻址错误。
- 单能力直调（smooth_skin 等）无脸仍报错「人像能力无目标」——点名磨皮而
  无脸可磨，报错是诚实语义；「部分适用」只属于套系。

MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config, data, presets
from ..engine import EngineContext, get_context, get_session, png_sha256
from . import CAPABILITIES, load_all
from . import face_enhance as fe_infra
from . import faces as faces_infra
from . import skin as skin_infra

log = logging.getLogger("pixretouch.capabilities.preset_engine")

CAPABILITY_NAME = "apply_preset"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _Step:
    capability: str
    params: dict[str, Any]
    cluster_id: int | None
    mask_sha: str | None
    mask_img: Any | None            # PIL "L"（双登记固化用）
    payload: dict[str, Any] = field(default_factory=dict)
    routed_suit: str | None = None  # 审计：该步实际来自哪套方案（路由结果）


# ------------------------------------------------------------------ 计划构建
def _person_step(step: dict, cid: int | None, rows: list[dict], size: tuple[int, int],
                 img_bgr: Any, suit_name: str) -> _Step:
    """person 域步骤 → 带 mask/payload/face_boxes 的可渲染步骤。"""
    import numpy as np
    from PIL import Image

    cap = step["capability"]
    landmarks = [r["landmark"] for r in rows]
    boxes = [list(r["bbox"]) for r in rows]
    params = {**step.get("params", {}), "cluster_id": cid, "face_boxes": boxes}

    if cap == "face_enhance":
        masks = fe_infra._face_masks(size, landmarks)
        combined = np.maximum.reduce(masks)
        params.setdefault("fidelity", fe_infra.W_FIDELITY)
        payload = {"faces": [(tuple(r["bbox"]), r["landmark"]) for r in rows], "masks": masks}
    else:
        kind = skin_infra._MASK_KIND[cap]
        mask = skin_infra.build_face_mask(kind, size, landmarks, img_bgr=img_bgr)
        combined = mask
        payload = {"mask": mask}
    mask_img = Image.fromarray((combined * 255.0).round().astype(np.uint8))
    return _Step(capability=cap, params=params, cluster_id=cid,
                 mask_sha=png_sha256(mask_img), mask_img=mask_img,
                 payload=payload, routed_suit=suit_name)


def _prepare_step(capability: str, params: dict, cache: data.ContentCache) -> tuple[dict, dict]:
    """工具级配方参数 → 渲染级规范形（素材规则，模式 docstring 纪律 7）。

    配方以工具级友好形式存储（{"lut": "bw-classic"} 或 {"lut_path": "..."}），
    引擎负责下降：LUT（内置名或外部 .cube）→ {"name", "sha256"}（内容 sha 进
    params/缓存键，路径不进键——issue #16 修复：lut_path 曾被静默丢弃）、
    素材 cache.put("material:"+sha) 固化、payload 预解析省算（重建路径 =
    ctx.cache.path_for(sha)，与 color_grade_op 行为逐位一致）。
    person 域能力现无素材类参数，直通。
    """
    params = dict(params)
    payload: dict[str, Any] = {}
    if capability == "color_grade":
        from . import color as color_cap

        lut = params.get("lut")
        lut_path = params.pop("lut_path", None)   # 路径本身永不进键（素材规则）
        if lut is not None and lut_path is not None:
            raise ValueError("lut（内置名）与 lut_path（外部文件）二选一")
        if lut_path is not None:
            # 外部 .cube 按素材规则真支持（issue #16：不允许静默丢弃）：
            # 内容 sha256 进 params/缓存键、素材 cache.put 固化、payload 预解析
            p = Path(str(lut_path)).expanduser()
            if not p.is_absolute():
                p = config.PROJECT_HOME / p
            if not p.exists():
                raise ValueError(f"lut_path 不存在：{p}")
            sha = data.file_sha256(p)
            cache.put(f"material:{sha}", p)
            payload["lut"] = color_cap.parse_cube(p)   # fail-fast：格式错误在计划期结构化抛出
            params["lut"] = {"name": None, "sha256": sha}
        elif isinstance(lut, str):
            src = color_cap.builtin_lut_path(lut)
            ref = {"name": lut, "sha256": data.file_sha256(src)}
            cache.put(f"material:{ref['sha256']}", src)
            payload["lut"] = color_cap.parse_cube(src)
            params["lut"] = ref
        elif isinstance(lut, dict) and lut.get("sha256"):
            src = cache.path_for(lut["sha256"])
            if src.exists():
                payload["lut"] = color_cap.parse_cube(src)
    return params, payload


def build_plan(conn: sqlite3.Connection, cache: data.ContentCache, image_id: int,
               suit: dict[str, Any], cluster_id: int | None) -> tuple[list[_Step], str | None]:
    """配方 → 确定性执行计划（person 段按聚类路由展开 → image 段）。

    返回 (plan, person_skipped)；person_skipped 非 None = 图级应用零人脸，
    person 域已按两分法跳过（见模块 docstring）。
    """
    import cv2
    import numpy as np

    steps = suit.get("steps", [])
    person_steps = [s for s in steps if s.get("scope") == "person"]
    image_steps = [s for s in steps if s.get("scope", "image") == "image"]

    cur = data.get_current(conn, image_id)
    plan: list[_Step] = []
    person_skipped: str | None = None
    rows_all: list[dict] = []
    if person_steps:
        if cluster_id is not None:
            # 显式寻址：无脸/聚类无交集 = 寻址错误，响亮失败（target_faces 抛 ValueError）
            rows_all = skin_infra.target_faces(conn, image_id, cluster_id)
        else:
            # 图级语义：零人脸 → person 域跳过 + 审计（套系 =「可应用的都应用」）
            rows_all = faces_infra.load_faces(conn, image_id)
            if not rows_all:
                detected = faces_infra.detect(Path(cur["output_path"]))
                # 对账式持久化（issue #17）：复用既有行，保留聚类归属
                faces_infra.persist_faces(
                    conn, image_id, detected, image_path=Path(cur["output_path"]))
                rows_all = faces_infra.load_faces(conn, image_id)
            if not rows_all:
                person_skipped = "未检出人脸——person 域步骤跳过（image 域照常执行）"
    if person_steps and person_skipped is None:
        img = cv2.imread(str(cur["output_path"]), cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            img = cv2.cvtColor(np.asarray(Image.open(cur["output_path"]).convert("RGB")),
                               cv2.COLOR_RGB2BGR)
        size = (img.shape[1], img.shape[0])

        # 分组：指定 cluster_id → 单组；否则按 cluster_id 分组（升序，未聚类最后）
        groups: list[tuple[int | None, list[dict]]] = []
        if cluster_id is not None:
            groups = [(int(cluster_id), rows_all)]
        else:
            by_cid: dict[int | None, list[dict]] = {}
            for r in rows_all:
                by_cid.setdefault(r["cluster_id"], []).append(r)
            ordered = sorted(by_cid.keys(), key=lambda c: (c is None, c if c is not None else 0))
            groups = [(cid, by_cid[cid]) for cid in ordered]

        for cid, rows in groups:
            # 路由：聚类绑定 default_suit_id → 用绑定方案的 person 域步骤
            eff_steps, eff_name = person_steps, suit["name"]
            if cid is not None:
                crow = conn.execute("SELECT default_suit_id FROM clusters WHERE id=?", (cid,)).fetchone()
                if crow is not None and crow["default_suit_id"] is not None:
                    bound = presets.resolve_suit(conn, crow["default_suit_id"])
                    eff_steps = [s for s in bound.get("steps", []) if s.get("scope") == "person"]
                    eff_name = f"{bound['name']}@v{bound['version']}"
            for st in eff_steps:
                plan.append(_person_step(st, cid, rows, size, img, eff_name))

    if image_steps and cluster_id is None:
        for st in image_steps:
            params, extra = _prepare_step(st["capability"], st.get("params", {}), cache)
            plan.append(_Step(capability=st["capability"], params=params,
                              cluster_id=None, mask_sha=None, mask_img=None,
                              payload=extra, routed_suit=suit["name"]))
    return plan, person_skipped


# ------------------------------------------------------------------ 事务化执行
def run_plan(conn: sqlite3.Connection, cache: data.ContentCache, image_id: int,
             plan: list[_Step]) -> dict[str, Any]:
    """两阶段事务：① 逐段渲染全部入缓存（零库写）② 单事务追加全部记录+移指针。

    阶段①任何异常 → 无库副作用（缓存条目无害且为下次重跑的加速材料）；
    阶段②异常 → rollback，链与指针不动。
    """
    cur = data.get_current(conn, image_id)
    input_sha = cur["output_sha256"]
    input_path = Path(cur["output_path"])
    staged: list[dict[str, Any]] = []
    tmp_files: list[Path] = []
    cache_hits = 0
    try:
        # ---- 阶段①：渲染进缓存（链位置敏感的键：上一段输出哈希 = 下一段输入哈希）
        for st in plan:
            key = cache.key(input_sha, st.capability, st.params, st.mask_sha)
            hit = cache.get(key)
            if hit is not None:
                out_sha, out_path = data.file_sha256(hit), hit
                cache_hits += 1
            else:
                fn = CAPABILITIES[st.capability]["fn"]
                ctx = get_context()
                ctx.cache = cache
                ctx.payload.update(st.payload)
                out_tmp = Path(fn(input_path, st.params, ctx))
                tmp_files.append(out_tmp)
                out_sha, out_path = cache.put(key, out_tmp)
            staged.append({
                "capability": st.capability,
                "params": json.dumps(st.params, sort_keys=True, ensure_ascii=False),
                "output_sha256": out_sha,
                "output_path": str(out_path),
                "cluster_id": st.cluster_id,
                "mask_sha256": st.mask_sha,
            })
            input_sha, input_path = out_sha, out_path

        # ---- 阶段②：单事务一次性追加全部记录 + 移动当前态指针
        parent_id = cur["id"]
        record_ids: list[int] = []
        for rec in staged:
            row = conn.execute(
                "INSERT INTO records(image_id, parent_id, capability, params, output_sha256,"
                " output_path, cluster_id, mask_sha256, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (image_id, parent_id, rec["capability"], rec["params"], rec["output_sha256"],
                 rec["output_path"], rec["cluster_id"], rec["mask_sha256"], _now()),
            )
            parent_id = row.lastrowid
            record_ids.append(parent_id)
        if record_ids:
            conn.execute("UPDATE images SET current_record_id=? WHERE id=?", (parent_id, image_id))
        conn.commit()
    except Exception:
        conn.rollback()   # 阶段②中途失败：链与指针完全恢复（阶段①本就零库写）
        raise
    finally:
        for p in tmp_files:
            p.unlink(missing_ok=True)

    return {
        "steps": len(staged),
        "cache_hits": cache_hits,
        "record_ids": record_ids,
        "output_sha256": input_sha,
        "output_path": str(input_path),
    }


# ------------------------------------------------------------------ 工具实现（接缝）
def apply_preset_op(image_id: int, suit: int | str, cluster_id: int | None = None) -> dict[str, Any]:
    """apply_preset 工具实现（server.py 只做薄包装）。

    suit：方案名（取最新版）或 preset_suits.id；cluster_id：单聚类覆盖应用。
    """
    conn, cache = get_session()
    mask_tmps: list[Path] = []
    try:
        load_all()   # 全量能力注册表（渲染接缝按能力名派发，幂等）
        presets.ensure_seeds(conn)
        suit_row = presets.resolve_suit(conn, suit)
        plan, person_skipped = build_plan(conn, cache, int(image_id), suit_row, cluster_id)
        result = run_plan(conn, cache, int(image_id), plan)

        # mask 双登记（成功路径）：与单能力 op 同规则 cache.put("mask:"+sha)
        mask_paths: dict[str, str] = {}
        ctx = get_context()
        for st in plan:
            if st.mask_img is not None and st.mask_sha not in mask_paths:
                tmp = ctx.scratch_file("mask", ".png")
                st.mask_img.save(tmp, "PNG")
                _, mpath = cache.put(f"mask:{st.mask_sha}", tmp)
                mask_paths[st.mask_sha] = str(mpath)
                mask_tmps.append(tmp)

        out = {
            "status": "ok",
            "suit": f"{suit_row['name']}@v{suit_row['version']}",
            "suit_id": suit_row["id"],
            "cluster_id": cluster_id,
            "routing": sorted({st.routed_suit for st in plan if st.routed_suit}),
            "mask_paths": mask_paths,
            **result,
        }
        if person_skipped is not None:
            out["person_skipped"] = person_skipped   # 审计：图级零人脸，person 域跳过
        return out
    except Exception as e:  # noqa: BLE001  结构化错误返回；事务性保证零记录残留
        log.exception("apply_preset(image_id=%s, suit=%s) 失败", image_id, suit)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in mask_tmps:
            p.unlink(missing_ok=True)
