"""人脸增强 — 工单 07：CodeFormer 输出按强度与原图保守混合。

能力：低清/模糊/老照片人脸的修复增强（CodeFormer ONNX），输出与原图按
strength 混合——**默认 0.3 保守混合，杜绝「换脸感」/恐怖谷**（Spec 实现决定：
默认混合强度 ≤0.3）。支持 cluster_id 按人寻址（工单 05 人物聚类 + 工单 06
skin.target_faces 复用）。

模型与供给链（哈希钉死纪律，白盒复刻边界）：
- codeformer.onnx（376,821,950 B）sha256=91e7e881…f57a，主源 hf-mirror
  bluefoxcreation/Codeformer-ONNX（单文件 ONNX，pytorch 2.0 导出，opset 11，
  输入 x[batch,3,512,512] float + w 标量 double=保真度）。
- ModelScope 勘察结论（2026-10-21）：budaoshou/CodeFormer-ONNX 为**不同转换版本**
  （376,951,650 B，哈希不符）——哈希钉死策略拒绝混源，兜底登记 None；hf-mirror
  失败时报结构化错误并提示重试（不做静默换模型）。
- 无官方 ONNX 发行（官方发 .pth，加载需 basicsr/gfpgan 依赖地狱，弃）；单源
  + 哈希钉死 + 功能实证（模糊脸清晰度 2.5→120.4、身份相关 0.969；干净脸
  294→296 近恒等）——沿 LaMa（工单 03）先例，证据记 issue #8。
- ORT 锁 CPUExecutionProvider（纪律 9 标准形：模块级单例 + 双检锁；确定性，
  与工单 08 对齐）。

对齐与混合（白盒：GFPGAN 公开 512 模板常数，Apache-2.0 仓库）：
- 每目标脸：5 点 kps（工单 05 clusters.kps_from_landmarks，源自 478 点 landmark）
  → skimage SimilarTransform 对齐 512×512 → CodeFormer（w=0.7 保真优先，
  第二层保守）→ 逆变换回原画布 → 与该脸椭圆羽化 mask × strength 混合。
- mask 口径（lead 裁决 2026-10-21，测试判读基准）：face_oval 羽化权重图，
  **mask=0 区域严格不变**（断言 <JPEG 噪声级）；mask>0.5 核心区做变化断言；
  羽化过渡带（0<mask≤0.5）为合法变化区，不做断言。羽化尾在合成前截断到 0
  （<0.01 → 0），保证 mask=0 处数学恒等。

记录链模式（工单 03 范本）：@capability 注册 + op 薄接缝 + data.apply
（cluster_id 专列 + params.face_boxes 仅凭 params 重建）+ mask 双登记。
RGBA 纪律 8：只调 RGB、alpha 直通、产物 PNG 无损。
MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Sequence

from .. import data
from ..engine import EngineContext, get_context, get_session, png_sha256
from . import capability, weight
from . import clusters as clusters_infra
from . import faces as faces_infra
from . import skin as skin_infra

log = logging.getLogger("pixretouch.capabilities.face_enhance")

WEIGHT_NAME = "codeformer"
CAPABILITY_NAME = "face_enhance"

weight(WEIGHT_NAME, {
    "repo": "bluefoxcreation/Codeformer-ONNX",   # hf-mirror 主源（哈希钉死）
    "file": "codeformer.onnx",
    "kind": "file",
    # ModelScope 勘察（2026-10-21）：budaoshou/CodeFormer-ONNX 为不同转换版本
    # （字节数/哈希均不符），哈希钉死拒绝混源 → 无兜底，失败报结构化错误。
    "modelscope": None,
    "sha256": "91e7e881c5001fea4a535e8f96eaeaa672d30c963a678a3e27f0429a6620f57a",
})

DEFAULT_STRENGTH = 0.3      # 票面保守上限：默认混合强度 ≤0.3（杜绝换脸感）
W_FIDELITY = 0.7            # CodeFormer 保真度（0=质量优先强生成，1=保真优先；保守取高保真）
ENHANCE_SIZE = 512

# GFPGAN 公开对齐模板（512×512，5 点：图像左眼/右眼/鼻尖/左嘴角/右嘴角）
FACE_TEMPLATE_512 = (
    (192.98138, 239.94708), (318.90277, 240.1936), (256.63416, 314.01935),
    (201.26117, 371.41043), (313.08905, 371.15118),
)


# ------------------------------------------------------------------ ORT 单例（纪律 9）
_SESSION: dict[str, Any] = {}
_SESSION_LOCK = threading.Lock()


def ort_session():
    """CodeFormer ORT 会话（模块级单例 + 双检锁 + CPU EP 锁定）。"""
    s = _SESSION.get("sess")
    if s is None:
        with _SESSION_LOCK:
            s = _SESSION.get("sess")
            if s is None:
                import onnxruntime

                p = faces_infra.model_path(WEIGHT_NAME)
                so = onnxruntime.SessionOptions()
                so.log_severity_level = 3
                s = _SESSION["sess"] = onnxruntime.InferenceSession(
                    str(p), sess_options=so, providers=["CPUExecutionProvider"])
                log.info("CodeFormer 常驻（CPU EP）：%s", p)
    return s


def reset_session() -> None:
    """释放 ORT 会话（测试钩子）。"""
    _SESSION.clear()


# ------------------------------------------------------------------ 对齐与推理
def _similarity(src, dst):
    """skimage SimilarTransform（0.26 from_estimate 新 API，兼容旧 estimate）。"""
    import numpy as np
    import skimage.transform as skt

    ST = skt.SimilarityTransform
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    try:
        return ST.from_estimate(src=src, dst=dst)
    except TypeError:
        t = ST()
        t.estimate(src, dst)   # 旧 API（Deprecated 但可用）
        return t


def align_face(img_bgr: Any, kps) -> tuple[Any, Any]:
    """5 点 kps → 512×512 对齐脸 + SimilarTransform（.params[:2] 为 warpAffine M）。"""
    import cv2
    import numpy as np

    t = _similarity(kps, np.array(FACE_TEMPLATE_512, dtype=np.float64))
    aligned = cv2.warpAffine(img_bgr, t.params[:2], (ENHANCE_SIZE, ENHANCE_SIZE), borderValue=0)
    return aligned, t


def enhance_aligned(aligned_bgr: Any, fidelity: float = W_FIDELITY) -> Any:
    """512 对齐脸 → CodeFormer 增强 → uint8 BGR。确定性（CPU EP）。"""
    import cv2
    import numpy as np

    sess = ort_session()
    rgb = np.ascontiguousarray(aligned_bgr[..., ::-1]).astype(np.float32) / 255.0
    x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
    y = sess.run(["y"], {sess.get_inputs()[0].name: x,
                         sess.get_inputs()[1].name: np.array(float(fidelity), dtype=np.float64)})[0]
    out = np.clip((y[0].transpose(1, 2, 0) + 1.0) / 2.0 * 255.0, 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(out[..., ::-1])


def enhance_face_in_canvas(img_bgr: Any, landmarks, fidelity: float) -> Any:
    """单脸增强并逆变换回原画布尺寸（画布外/对齐空区为 0，由 mask 合成兜底）。"""
    import cv2
    import numpy as np

    kps = clusters_infra.kps_from_landmarks(landmarks)
    aligned, t = align_face(img_bgr, kps)
    out512 = enhance_aligned(aligned, fidelity)
    h, w = img_bgr.shape[:2]
    inv_m = t.inverse.params[:2]
    return cv2.warpAffine(out512, inv_m, (w, h), borderValue=0)


# ------------------------------------------------------------------ 渲染函数
def _faces_from_params(image_path: Path, params: dict) -> list:
    """仅凭 params（face_boxes）重建目标脸（检测确定性 ⇒ bbox 精确匹配）。"""
    detected = faces_infra.detect(Path(image_path))
    wanted = {tuple(b) for b in params.get("face_boxes", [])}
    rows = [f for f in detected if tuple(f.bbox) in wanted]
    if not rows:
        raise RuntimeError("face_enhance 重建失败：params.face_boxes 与检测结果不匹配")
    return rows


def _face_masks(size, faces_lms, feather_eps: float = 0.01):
    """每脸 face_oval 羽化 mask（尾部截断保证 mask=0 区严格恒等）。"""
    import numpy as np

    masks = []
    for lm in faces_lms:
        m = skin_infra.build_face_mask("face_oval", size, [lm])
        m = np.where(m < feather_eps, 0.0, m)
        masks.append(m.astype(np.float32))
    return masks


@capability(
    CAPABILITY_NAME,
    params_schema={
        "strength": "混合强度 (0,1]，默认 0.3（保守上限，杜绝换脸感）",
        "fidelity": "CodeFormer 保真度 w [0,1]，默认 0.7（高保真，弱生成）",
        "cluster_id": "人物聚类 ID（按人寻址）；None=全图检出人脸",
        "face_boxes": "目标脸 bbox 列表（op 自动填充，参与缓存键，mask/对齐重建依据）",
    },
    defaults={"strength": DEFAULT_STRENGTH, "fidelity": W_FIDELITY},
    weight=WEIGHT_NAME,
)
def render_face_enhance(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染：逐脸 对齐→CodeFormer→逆变换→椭圆羽化 mask×strength 混合（纪律 8 RGBA 直通）。"""
    import numpy as np
    from PIL import Image

    with Image.open(image) as im:
        alpha_ch = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = im.convert("RGB")
    img_f = np.asarray(rgb, dtype=np.float32)
    bgr = np.ascontiguousarray(img_f[..., ::-1])
    h, w = img_f.shape[:2]

    faces_data = ctx.payload.get("faces") if ctx is not None else None
    if faces_data is None:  # 纪律：仅凭 params 可重建
        faces_data = [(f.bbox, f.landmarks) for f in _faces_from_params(image, params)]
    masks = ctx.payload.get("masks") if ctx is not None else None
    if masks is None:
        masks = _face_masks((w, h), [lm for _bbox, lm in faces_data])

    strength = float(params.get("strength", DEFAULT_STRENGTH))
    fidelity = float(params.get("fidelity", W_FIDELITY))
    acc = np.zeros_like(img_f)
    for (bbox, lm), m in zip(faces_data, masks):
        enhanced_full = enhance_face_in_canvas(bgr, lm, fidelity)   # BGR uint8 画布尺寸
        diff = enhanced_full.astype(np.float32)[..., ::-1] - img_f  # → RGB delta
        acc += diff * (m * strength)[..., None]
    out = np.clip(img_f + acc, 0, 255).round().astype(np.uint8)

    img_out = Image.fromarray(out)
    if alpha_ch is not None:
        img_out.putalpha(alpha_ch)
        dst = ctx.scratch_file("face_enhance", ".png")
        img_out.save(dst, "PNG")     # RGBA 态保持无损（纪律 8）
    else:
        dst = ctx.scratch_file("face_enhance", ".jpg")
        img_out.save(dst, "JPEG", quality=95, subsampling=0)   # 确定性编码
    return dst


# ------------------------------------------------------------------ 工具实现（接缝）
def face_enhance_op(image_id: int, cluster_id: int | None = None,
                    strength: float = DEFAULT_STRENGTH) -> dict[str, Any]:
    """face_enhance 工具实现（server.py 只做薄包装）。"""
    import numpy as np
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    mask_tmp: Path | None = None
    try:
        s = float(strength)
        if not (0.0 < s <= 1.0):
            raise ValueError(f"strength 须在 (0,1]，收到 {strength!r}")
        cid = None if cluster_id is None else int(cluster_id)
        cur = data.get_current(conn, image_id)   # 图像不存在 → KeyError → 结构化错误
        rows = skin_infra.target_faces(conn, image_id, cid)

        with Image.open(cur["output_path"]) as im:
            size = im.convert("RGB").size        # (w, h)
        faces_data = [(tuple(r["bbox"]), r["landmark"]) for r in rows]
        masks = _face_masks(size, [lm for _b, lm in faces_data])
        combined = np.maximum.reduce(masks) if masks else np.zeros((size[1], size[0]), np.float32)
        mask_img = Image.fromarray((combined * 255.0).round().astype(np.uint8))
        m_sha = png_sha256(mask_img)

        params: dict[str, Any] = {
            "strength": round(s, 4),
            "fidelity": W_FIDELITY,
            "cluster_id": cid,
            "face_boxes": [list(b) for b, _lm in faces_data],
        }
        ctx = get_context()
        ctx.cache = cache
        ctx.payload["faces"] = faces_data
        ctx.payload["masks"] = masks

        def _render(in_path: Path, prm: dict) -> Path:
            out = render_face_enhance(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, CAPABILITY_NAME, params, _render,
                            cluster_id=cid, mask_sha=m_sha)
        mask_tmp = ctx.scratch_file("mask", ".png")
        mask_img.save(mask_tmp, "PNG")
        _, mask_path = cache.put(f"mask:{m_sha}", mask_tmp)
        return {"status": "ok", "mask_sha256": m_sha, "mask_path": str(mask_path), **result}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("face_enhance(image_id=%s, cluster_id=%s) 失败", image_id, cluster_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
        if mask_tmp is not None:
            mask_tmp.unlink(missing_ok=True)
