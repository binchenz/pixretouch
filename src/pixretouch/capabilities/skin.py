"""皮肤磨皮与牙眼 — 工单 06：按人寻址的三项人像能力。

能力（GLOSSARY「按人寻址」：参数以人物聚类为单位定向应用）：
- smooth_skin：保留纹理的频率分离磨皮——低频均匀（肤色/瑕疵）、高频纹理保留
  （毛孔/细节），皮肤 mask 内按 strength 混合；
- whiten_teeth：牙齿美白（牙 mask ∩ 亮低饱和像素，HSV 提亮去黄，有上限）；
- brighten_eyes：亮眼（眼 mask 内暗像素优先提亮，巩膜增白、虹膜温和，有上限）。

face-parsing 实现选型（白盒复刻，记 issue #7）：不引入 BiSeNet 类分割模型，
以工单 04 的 mediapipe 478 点 landmark 拓扑直接派生皮肤/牙齿/眼部 mask
（FACE_OVAL 减去眼/眉/唇得皮肤；唇内缩 ∩ 亮低饱和得牙齿；眼轮廓外扩得眼部）。
理由：零新增权重与供给链面；每脸自有几何 → 按人寻址天然精确；确定性可缓存。
mask 是渲染实际消费的权重图 → 走 mask 通道双登记（纪律：png_sha256 落
records.mask_sha256 + cache.put("mask:"+sha) 固化 + 结果返回 mask_path）。

按人寻址与缓存键语义（dev-a 对齐 2026-10-21）：
- cluster_id 进 params（参与缓存键）并落 records.cluster_id 专列（可审计）；
- params.face_boxes（op 自动填充）记录目标脸 bbox：渲染函数「仅凭 params 重建
  mask」时重跑检测（同图像内容 ⇒ 检测确定性 ⇒ bbox 精确匹配还原同一脸子集），
  不依赖库状态，满足 ADR-0002 键语义；
- 聚类状态变化 → 目标脸集合变化 → mask sha 变化 → 键自然不同，无假命中。

记录链模式（工单 03 范本）：@capability 注册渲染函数 + op 薄接缝 + data.apply；
RGBA 链态纪律 8：只调 RGB、alpha 直通、产物保持 RGBA（PNG 无损）。
mediapipe 推理经 faces.detect（容器化：专职 owner 线程 + 超时 + 代际自愈 +
重试一次契约），本模块不直接持有任何模型句柄。
MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from .. import data
from ..engine import EngineContext, get_context, get_session, png_sha256
from . import capability
from . import faces as faces_infra

log = logging.getLogger("pixretouch.capabilities.skin")

# ------------------------------------------------------------------ 拓扑常量
# mediapipe FaceMesh 478 点拓扑的规范索引集（公开拓扑事实，白盒使用）。
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153,
                     145, 144, 163, 7]           # 图像左侧眼
LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373,
                    374, 380, 381, 382]          # 图像右侧眼
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
              267, 0, 37, 39, 40, 185]

MASK_KINDS = ("skin", "teeth", "eyes")


# ------------------------------------------------------------------ mask 构建
def _pts(landmarks: Sequence[tuple[float, float]], idxs: Sequence[int]):
    import numpy as np

    return np.array([[landmarks[i][0], landmarks[i][1]] for i in idxs], dtype=np.int32)


def build_face_mask(
    kind: str,
    size: tuple[int, int],
    faces_landmarks: Sequence[Sequence[tuple[float, float]]],
    img_bgr: Any | None = None,
) -> Any:
    """kind ∈ {skin, teeth, eyes}：多脸合成权重 mask（float32 (h,w) ∈[0,1]）。

    每脸以自有尺度（脸宽）确定膨胀/羽化半径——分辨率无关；多脸取逐像素 max。
    teeth 需要像素信息（亮低饱和选择），传 img_bgr；闭嘴时 mask 近空 →
    渲染近恒等（记录照常追加，语义诚实）。
    """
    import cv2
    import numpy as np

    if kind not in MASK_KINDS:
        raise ValueError(f"未知 mask 类型 {kind!r}（{MASK_KINDS}）")
    w, h = size
    canvas = np.zeros((h, w), dtype=np.float32)
    for landmarks in faces_landmarks:
        pts_l = list(landmarks)
        face_w = float(np.ptp([p[0] for p in pts_l]))
        face_w = max(face_w, 8.0)

        if kind == "skin":
            region = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region, [_pts(pts_l, FACE_OVAL)], 255)
            exclude = np.zeros((h, w), dtype=np.uint8)
            for idxs in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR, RIGHT_BROW, LEFT_BROW, LIPS_OUTER):
                cv2.fillPoly(exclude, [_pts(pts_l, idxs)], 255)
            k = max(3, int(face_w * 0.04) | 1)
            exclude = cv2.dilate(exclude, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            region = cv2.subtract(region, exclude)
            sigma = max(1.0, face_w * 0.02)
            m = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (0, 0), sigma)

        elif kind == "eyes":
            region = np.zeros((h, w), dtype=np.uint8)
            for idxs in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
                poly = _pts(pts_l, idxs)
                cv2.fillPoly(region, [poly], 255)
            k = max(3, int(face_w * 0.03) | 1)
            region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            m = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (0, 0), max(1.0, face_w * 0.012))

        else:  # teeth
            lips = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(lips, [_pts(pts_l, LIPS_OUTER)], 255)
            kh = max(3, int(face_w * 0.045) | 1)
            inner = cv2.erode(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh)))
            if img_bgr is None:
                raise ValueError("teeth mask 需要 img_bgr（亮低饱和像素选择）")
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # HSV 通道序：[...,1]=S 饱和度，[...,2]=V 明度（勿倒置——牙=亮且低饱和）
            s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
            bright = ((v > 150) & (s < 100)).astype(np.uint8) * 255
            teeth = cv2.bitwise_and(inner, bright)
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 牙列在照片里是细条带：先闭运算连接条带，再开运算去孤立散点
            # （实测 Pexels 1181686：只做开运算会把 2008px 的牙区打碎成 10×11 碎点）
            teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE, k3)
            teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN, k3)
            m = cv2.GaussianBlur(teeth.astype(np.float32) / 255.0, (0, 0), max(0.8, face_w * 0.008))

        canvas = np.maximum(canvas, m)
    return np.clip(canvas, 0.0, 1.0)


def _landmarks_of(rows: Sequence[dict]) -> list[list[tuple[float, float]]]:
    return [r["landmark"] for r in rows if r.get("landmark")]


def target_faces(conn, image_id: int, cluster_id: int | None) -> list[dict]:
    """本图目标人脸行（faces 表）；无行时自动重检出（工单 04 基建，agent 友好）。

    cluster_id 给定时按人物聚类过滤（按人寻址）；过滤后为空 → ValueError
    （结构化：先确认聚类/检测状态，静默空操作会掩盖寻址错误）。
    """
    rows = faces_infra.load_faces(conn, image_id)
    if not rows:
        cur = data.get_current(conn, image_id)
        detected = faces_infra.detect(Path(cur["output_path"]))
        faces_infra.persist_faces(conn, image_id, detected)
        rows = faces_infra.load_faces(conn, image_id)
    if cluster_id is not None:
        rows = [r for r in rows if r["cluster_id"] == int(cluster_id)]
        if not rows:
            raise ValueError(
                f"聚类 {cluster_id} 在图像 {image_id} 上没有检出人脸"
                f"（先 cluster_faces / detect_faces 确认状态）")
    if not rows:
        raise ValueError(f"图像 {image_id} 未检出人脸，人像能力无目标（可先 detect_faces）")
    return rows


def _rebuild_mask(image_path: Path, params: dict, kind: str) -> Any:
    """渲染函数无 payload 时的 mask 重建：仅凭 params（face_boxes）+ 图像内容。

    同图像内容 ⇒ faces.detect 确定性 ⇒ bbox 精确匹配还原目标脸子集 ⇒
    mask 与 op 构建一致（ADR-0002 键语义：仅凭 params 重建结果一致）。
    """
    import numpy as np
    from PIL import Image

    detected = faces_infra.detect(Path(image_path))
    wanted = {tuple(b) for b in params.get("face_boxes", [])}
    rows = [{"landmark": f.landmarks} for f in detected if tuple(f.bbox) in wanted]
    if not rows:
        raise RuntimeError("mask 重建失败：params.face_boxes 与检测结果不匹配")
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return build_face_mask(kind, (rgb.shape[1], rgb.shape[0]),
                           _landmarks_of(rows), img_bgr=bgr)


# ------------------------------------------------------------------ 渲染函数
def _open_rgb_alpha(image: Path):
    """纪律 8：RGBA 链态取 alpha 直通，只处理 RGB。"""
    from PIL import Image

    with Image.open(image) as im:
        alpha_ch = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = im.convert("RGB")
    return rgb, alpha_ch


def _save_out(ctx: EngineContext, prefix: str, out_u8: Any, alpha_ch: Any) -> Path:
    from PIL import Image

    img_out = Image.fromarray(out_u8)
    if alpha_ch is not None:
        img_out.putalpha(alpha_ch)
        dst = ctx.scratch_file(prefix, ".png")
        img_out.save(dst, "PNG")     # RGBA 态保持无损（纪律 8）
    else:
        dst = ctx.scratch_file(prefix, ".jpg")
        img_out.save(dst, "JPEG", quality=95, subsampling=0)  # 确定性编码
    return dst


def _mask_of(ctx: EngineContext, params: dict, kind: str, image_path: Path) -> Any:
    m = ctx.payload.get("mask") if ctx is not None else None
    if m is None:  # 纪律：仅凭 params 可重建
        m = _rebuild_mask(image_path, params, kind)
    return m


@capability(
    "smooth_skin",
    params_schema={
        "strength": "磨皮强度 0..1（低频均匀化混合比例），默认 0.6",
        "cluster_id": "人物聚类 ID（按人寻址）；None=全图检出人脸",
        "face_boxes": "目标脸 bbox 列表（op 自动填充，参与缓存键，mask 重建依据）",
    },
    defaults={"strength": 0.6},
    weight=None,
)
def render_smooth_skin(image: Path, params: dict, ctx: EngineContext) -> Path:
    """频率分离磨皮：low=Gauss(img)；smooth_low=Gauss(low)；out=img+(smooth_low-low)*mask*strength。

    高频层（img-low，纹理/毛孔）逐像素不变——「保留纹理」是构造性质，非调参结果。
    高斯 σ 随目标脸宽自适应（分辨率无关）。
    """
    import cv2
    import numpy as np

    rgb, alpha_ch = _open_rgb_alpha(image)
    img_f = np.asarray(rgb, dtype=np.float32)
    mask = _mask_of(ctx, params, "skin", image)

    boxes = params.get("face_boxes") or []
    face_w = max((b[2] for b in boxes), default=0) or min(img_f.shape[:2]) * 0.25
    s_low = max(1.5, float(face_w) * 0.045)
    s_smooth = max(3.0, float(face_w) * 0.10)
    low = cv2.GaussianBlur(img_f, (0, 0), s_low)
    smooth_low = cv2.GaussianBlur(low, (0, 0), s_smooth)
    strength = float(params.get("strength", 0.6))
    delta = (smooth_low - low) * (mask * strength)[..., None]
    out = np.clip(img_f + delta, 0, 255).round().astype(np.uint8)
    return _save_out(ctx, "smooth_skin", out, alpha_ch)


@capability(
    "whiten_teeth",
    params_schema={
        "amount": "美白强度 0..1（提亮+去黄，有上限防假牙感），默认 0.5",
        "cluster_id": "人物聚类 ID（按人寻址）；None=全图检出人脸",
        "face_boxes": "目标脸 bbox 列表（op 自动填充）",
    },
    defaults={"amount": 0.5},
    weight=None,
)
def render_whiten_teeth(image: Path, params: dict, ctx: EngineContext) -> Path:
    """牙 mask（唇内缩 ∩ 亮低饱和）内 HSV 提亮去黄：V += a*45*(1-V/255)，S *= 1-0.45a。"""
    import cv2
    import numpy as np

    rgb, alpha_ch = _open_rgb_alpha(image)
    img_f = np.asarray(rgb, dtype=np.float32)
    bgr_f = np.ascontiguousarray(img_f[..., ::-1])
    mask = _mask_of(ctx, params, "teeth", image)

    hsv = cv2.cvtColor(bgr_f.astype(np.float32), cv2.COLOR_BGR2HSV)
    amount = float(params.get("amount", 0.5))
    v = hsv[..., 2]
    s = hsv[..., 1]
    v2 = v + amount * 45.0 * (1.0 - v / 255.0) * mask
    s2 = s * (1.0 - 0.45 * amount * mask)
    hsv2 = np.dstack([hsv[..., 0], np.clip(s2, 0, 255), np.clip(v2, 0, 255)])
    bgr_out = cv2.cvtColor(hsv2.astype(np.float32), cv2.COLOR_HSV2BGR)
    out = np.clip(bgr_out[..., ::-1], 0, 255).round().astype(np.uint8)
    return _save_out(ctx, "whiten_teeth", out, alpha_ch)


@capability(
    "brighten_eyes",
    params_schema={
        "amount": "亮眼强度 0..1（暗像素优先提亮，有上限防死鱼眼），默认 0.35",
        "cluster_id": "人物聚类 ID（按人寻址）；None=全图检出人脸",
        "face_boxes": "目标脸 bbox 列表（op 自动填充）",
    },
    defaults={"amount": 0.35},
    weight=None,
)
def render_brighten_eyes(image: Path, params: dict, ctx: EngineContext) -> Path:
    """眼 mask 内暗像素优先提亮（增益 ∝ (255-V)/255）：巩膜增白、虹膜瞳孔温和。"""
    import cv2
    import numpy as np

    rgb, alpha_ch = _open_rgb_alpha(image)
    img_f = np.asarray(rgb, dtype=np.float32)
    bgr_f = np.ascontiguousarray(img_f[..., ::-1])
    mask = _mask_of(ctx, params, "eyes", image)

    hsv = cv2.cvtColor(bgr_f.astype(np.float32), cv2.COLOR_BGR2HSV)
    amount = float(params.get("amount", 0.35))
    v = hsv[..., 2]
    v2 = v + amount * 40.0 * ((255.0 - v) / 255.0) * mask
    hsv2 = np.dstack([hsv[..., 0], hsv[..., 1], np.clip(v2, 0, 255)])
    bgr_out = cv2.cvtColor(hsv2.astype(np.float32), cv2.COLOR_HSV2BGR)
    out = np.clip(bgr_out[..., ::-1], 0, 255).round().astype(np.uint8)
    return _save_out(ctx, "brighten_eyes", out, alpha_ch)


# ------------------------------------------------------------------ 工具实现（接缝）
_RENDERERS = {
    "smooth_skin": render_smooth_skin,
    "whiten_teeth": render_whiten_teeth,
    "brighten_eyes": render_brighten_eyes,
}
_MASK_KIND = {"smooth_skin": "skin", "whiten_teeth": "teeth", "brighten_eyes": "eyes"}
_LEVEL_PARAM = {"smooth_skin": "strength", "whiten_teeth": "amount", "brighten_eyes": "amount"}


def _validate_level(name: str, value: Any) -> float:
    v = float(value)
    if not (0.0 < v <= 1.0):
        raise ValueError(f"{name} 须在 (0,1]，收到 {value!r}")
    return round(v, 4)


def _face_region_op(image_id: int, cap_name: str, level: Any, cluster_id: int | None) -> dict[str, Any]:
    """三项能力的共享 op 骨架（工单 03 模式）：目标脸 → mask 双登记 → data.apply。"""
    import numpy as np
    from PIL import Image

    level_name = _LEVEL_PARAM[cap_name]
    conn, cache = get_session()
    tmp_out: list[Path] = []
    mask_tmp: Path | None = None
    try:
        level_v = _validate_level(level_name, level)
        cid = None if cluster_id is None else int(cluster_id)
        cur = data.get_current(conn, image_id)   # 图像不存在 → KeyError → 结构化错误
        rows = target_faces(conn, image_id, cid)

        with Image.open(cur["output_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        size = (rgb.shape[1], rgb.shape[0])
        mask_f = build_face_mask(_MASK_KIND[cap_name], size, _landmarks_of(rows), img_bgr=bgr)
        mask_img = Image.fromarray((mask_f * 255.0).round().astype(np.uint8))
        m_sha = png_sha256(mask_img)

        params: dict[str, Any] = {
            level_name: level_v,
            "cluster_id": cid,
            "face_boxes": [list(r["bbox"]) for r in rows],   # 自动填充，参与缓存键
        }
        ctx = get_context()
        ctx.cache = cache
        ctx.payload["mask"] = mask_f
        renderer = _RENDERERS[cap_name]

        def _render(in_path: Path, prm: dict) -> Path:
            out = renderer(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, cap_name, params, _render,
                            cluster_id=cid, mask_sha=m_sha)
        mask_tmp = ctx.scratch_file("mask", ".png")
        mask_img.save(mask_tmp, "PNG")
        _, mask_path = cache.put(f"mask:{m_sha}", mask_tmp)
        return {"status": "ok", "mask_sha256": m_sha, "mask_path": str(mask_path), **result}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("%s(image_id=%s, cluster_id=%s) 失败", cap_name, image_id, cluster_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
        if mask_tmp is not None:
            mask_tmp.unlink(missing_ok=True)


def smooth_skin_op(image_id: int, cluster_id: int | None = None, strength: float = 0.6) -> dict[str, Any]:
    """smooth_skin 工具实现（server.py 只做薄包装）。"""
    return _face_region_op(image_id, "smooth_skin", strength, cluster_id)


def whiten_teeth_op(image_id: int, cluster_id: int | None = None, amount: float = 0.5) -> dict[str, Any]:
    """whiten_teeth 工具实现（server.py 只做薄包装）。"""
    return _face_region_op(image_id, "whiten_teeth", amount, cluster_id)


def brighten_eyes_op(image_id: int, cluster_id: int | None = None, amount: float = 0.35) -> dict[str, Any]:
    """brighten_eyes 工具实现（server.py 只做薄包装）。"""
    return _face_region_op(image_id, "brighten_eyes", amount, cluster_id)
