"""人脸检测 + landmark 基建（工单 04 / issue #5）。

分析型能力：读图像文件产出人脸框 / 478 点 landmark / EAR，持久化到 faces 表；
不产出渲染产物、不追加记录链（记录链只登记编辑操作，见 ADR-0002）。
消费方：人物聚类（工单 05，faces.embedding 列已预留）、磨皮与牙眼 / 人脸增强
（工单 06/07，按人寻址）。

选型结论（工单 04 评估，已记 issue #5）：
- mediapipe 1.1.1：py3.11/arm64 可装、与 numpy 2.4.6 共存；仅 Tasks API
  （legacy solutions 已在 1.x 移除），wheel 零内置模型。本单采用
  FaceDetector（bbox + 检出分）+ FaceLandmarker（478 点，含眼睑轮廓 → EAR）。
- insightface 2.1：py3.11/arm64 可装（临时 venv 实测，onnxruntime 1.31.0
  arm64 wheel，CoreML EP 可用），留作工单 05 ArcFace 嵌入主路线；本单不引入。

供给链（白盒复刻边界 + 工单 01 供给链决定）：
- 不走 mediapipe 自带下载器；模型统一落 models/<name>/（gitignore）。
- 两个模型文件均已与 Google 官方发行（storage.googleapis.com/mediapipe-models）
  逐字节 sha256 核验一致；官方哈希钉死在 WEIGHT_SPECS，每次下载后校验，不符即拒。
- 下载三级：本地缓存 → hf-mirror（huggingface_hub）→ ModelScope（requests），
  全部失败抛 RuntimeError（结构化消息，MCP 工具层原样传播）。

MCP stdio 纪律：本模块不 print；日志一律 logging→stderr。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .. import config
from . import weight

log = logging.getLogger("pixretouch.faces")

# ------------------------------------------------------------------ 权重登记
# sha256 = Google 官方发行文件哈希（2026-10-21 核验），下载后逐字节校验。
weight("face_landmarker", {
    "repo": "Skulleton12/mediapipe-face-landmarker",   # hf-mirror 主源（哈希已核验）
    "file": "face_landmarker.task",
    "modelscope": "yangchunwanwusheng/mediapipe-vision-models",  # 兜底（哈希已核验）
    "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
})
weight("face_detector", {
    # HF 侧无官方 blaze_face_short_range.tflite 干净镜像（2026-10-21 勘察）；
    # prefetch 的 HF 尝试会如实失败并报 modelscope_fallback_id，本模块
    # model_path() 的 ModelScope 级负责实际供给（哈希已核验）。
    "repo": "Composable/blaze_face_short_range",
    "file": "blaze_face_short_range.tflite",
    "modelscope": "Composable/blaze_face_short_range",
    "sha256": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
})

# MediaPipe FaceLandmarker 478 点拓扑中的眼睑轮廓（EAR 用 6 点/眼）：
# p1 外眼角、p2/p3 上睑、p4 内眼角、p5/p6 下睑。
RIGHT_EYE = (33, 160, 158, 133, 153, 144)    # 图像左侧眼（主体右眼）
LEFT_EYE = (362, 385, 387, 263, 373, 380)    # 图像右侧眼（主体左眼）

EAR_CLOSED_THRESHOLD = 0.15   # 双眼 EAR 均低于此 → 闭眼。
# 验收 fixture 实测校准（issue #5）：睁眼正例 0.19~0.37、闭眼负例 0.015~0.022，
# 0.15 两侧余量 ≥0.04；skill 决策树（工单 12）引用此经验起点。

# 检出置信度：3 人合影 fixture（Apollo 11 老照片，胶片颗粒）第三脸检出分仅 0.26，
# 0.5/0.3 均漏检；0.2 + presence 0.35 可检出且全部 fixture 无误检。detector 低阈值
# 带来的重复框由 IoU 配对自然丢弃（faces 计数以 landmarker 为准）。
DETECTION_CONFIDENCE = 0.2
FACE_PRESENCE_CONFIDENCE = 0.35


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def model_path(name: str, models_dir: Path | None = None) -> Path:
    """确保 WEIGHT_SPECS[name] 的模型文件在本地并通过哈希校验，返回路径。

    三级供给：本地缓存 → hf-mirror（HF_ENDPOINT）→ ModelScope resolve URL。
    任何一级下载成功但哈希不符都视为污染，删除并继续下一级；全部失败抛
    RuntimeError。models_dir 可注入（测试用），默认 config.MODELS_DIR。
    """
    from . import WEIGHT_SPECS

    spec = WEIGHT_SPECS[name]
    want = spec["sha256"]
    root = Path(models_dir if models_dir is not None else config.MODELS_DIR) / name
    dest = root / spec["file"]
    if dest.exists():
        if _sha256(dest) == want:
            return dest
        log.warning("model %s hash mismatch on disk, re-fetching", name)
        dest.unlink()
    root.mkdir(parents=True, exist_ok=True)

    # 1) hf-mirror 主源
    try:
        from huggingface_hub import hf_hub_download

        got = Path(hf_hub_download(spec["repo"], spec["file"], local_dir=root,
                                   endpoint=config.HF_MIRROR))
        if _sha256(got) == want:
            if got != dest:
                got.replace(dest)
            log.info("model %s fetched via hf-mirror", name)
            return dest
        got.unlink(missing_ok=True)
        log.warning("model %s from hf-mirror failed sha256 check", name)
    except Exception as e:  # noqa: BLE001
        log.warning("hf-mirror fetch failed for %s: %s: %s", name, type(e).__name__, e)

    # 2) ModelScope 兜底
    ms_id = spec.get("modelscope")
    if ms_id:
        try:
            import requests

            url = f"{config.MODELSCOPE_BASE}/models/{ms_id}/resolve/master/{spec['file']}"
            tmp = root / (spec["file"] + ".part")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if _sha256(tmp) == want:
                tmp.replace(dest)
                log.info("model %s fetched via ModelScope fallback", name)
                return dest
            tmp.unlink(missing_ok=True)
            log.warning("model %s from ModelScope failed sha256 check", name)
        except Exception as e:  # noqa: BLE001
            log.warning("ModelScope fetch failed for %s: %s: %s", name, type(e).__name__, e)

    raise RuntimeError(
        f"model '{name}' ({spec['file']}, sha256={want}) unavailable: "
        f"hf-mirror and ModelScope both failed; run doctor then retry"
    )


# ------------------------------------------------------------------ 推理单例
_model_cache: dict[str, Any] = {}


def _mp_image(image_path: Path):
    """读图为 mediapipe SRGB Image；返回 (mp_image, width, height)。"""
    import mediapipe as mp
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr)), arr.shape[1], arr.shape[0]


def _detector():
    if "detector" not in _model_cache:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path("face_detector"))),
            min_detection_confidence=DETECTION_CONFIDENCE,
        )
        _model_cache["detector"] = vision.FaceDetector.create_from_options(opts)
    return _model_cache["detector"]


def _landmarker(num_faces: int):
    key = f"landmarker:{num_faces}"
    if key not in _model_cache:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path("face_landmarker"))),
            num_faces=num_faces,
            min_face_detection_confidence=DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_PRESENCE_CONFIDENCE,
        )
        _model_cache[key] = vision.FaceLandmarker.create_from_options(opts)
    return _model_cache[key]


def reset_models() -> None:
    """释放已驻留的检测器/landmarker（测试钩子；ADR-0003 常驻进程正常不调用）。"""
    for obj in _model_cache.values():
        try:
            obj.close()
        except Exception:  # noqa: BLE001
            pass
    _model_cache.clear()


# ------------------------------------------------------------------ 检测
@dataclass
class Face:
    """单张人脸的检测结果（像素坐标）。

    bbox: (x, y, w, h)；quality: FaceDetector 检出分（仅 landmarker 命中时 None）；
    landmarks: 478 个 (x, y)；ear_left/ear_right: 眼睑纵横比。
    """

    bbox: tuple[int, int, int, int]
    quality: float | None
    landmarks: list[tuple[float, float]]
    ear_left: float
    ear_right: float

    def eyes_closed(self, threshold: float = EAR_CLOSED_THRESHOLD) -> bool:
        return self.ear_left < threshold and self.ear_right < threshold

    def wink(self, threshold: float = EAR_CLOSED_THRESHOLD) -> bool:
        return (self.ear_left < threshold) != (self.ear_right < threshold)

    def summary(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "quality": self.quality,
            "ear_left": self.ear_left,
            "ear_right": self.ear_right,
            "eyes_closed": self.eyes_closed(),
            "wink": self.wink(),
        }


def eye_aspect_ratio(six_points: Sequence[tuple[float, float]]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|)。

    p1/p4 为内外眼角，p2/p3 上睑、p5/p6 下睑（顺序对应 RIGHT_EYE/LEFT_EYE）。
    睁眼典型 0.25~0.40，闭眼 < 0.15；阈值见 EAR_CLOSED_THRESHOLD。
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = six_points

    def d(ax: float, ay: float, bx: float, by: float) -> float:
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    horiz = d(x1, y1, x4, y4)
    if horiz <= 1e-9:
        return 0.0
    return (d(x2, y2, x6, y6) + d(x3, y3, x5, y5)) / (2.0 * horiz)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect(image_path: Path, max_faces: int = 8) -> list[Face]:
    """检测人脸 + 478 点 landmark；detector 框与 landmarker 点按 IoU 配对。

    landmarker 独有命中（detector 漏检）时 bbox 取 landmark 包络、quality=None。
    模型缺失/下载失败抛 RuntimeError（结构化，工具层原样传播）。
    """
    image_path = Path(image_path)
    mp_image, w, h = _mp_image(image_path)

    det_result = _detector().detect(mp_image)
    det_boxes: list[tuple[tuple[int, int, int, int], float | None]] = []
    for d in det_result.detections:
        bb = d.bounding_box
        score = d.categories[0].score if d.categories else None
        det_boxes.append(((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)), score))

    lm_result = _landmarker(max_faces).detect(mp_image)
    faces: list[Face] = []
    used: set[int] = set()
    for landmarks in lm_result.face_landmarks:
        pts = [(p.x * w, p.y * h) for p in landmarks]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        derived = (x0, y0, min(w, int(max(xs))) - x0, min(h, int(max(ys))) - y0)

        best, best_iou = -1, 0.0
        for i, (box, _score) in enumerate(det_boxes):
            if i in used:
                continue
            iou = _iou(derived, box)
            if iou > best_iou:
                best, best_iou = i, iou
        if best >= 0 and best_iou > 0.3:
            used.add(best)
            bbox, quality = det_boxes[best]
        else:
            bbox, quality = derived, None

        faces.append(Face(
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            quality=round(float(quality), 4) if quality is not None else None,
            landmarks=[(round(x, 2), round(y, 2)) for x, y in pts],
            ear_left=round(eye_aspect_ratio([pts[i] for i in LEFT_EYE]), 4),
            ear_right=round(eye_aspect_ratio([pts[i] for i in RIGHT_EYE]), 4),
        ))
    return faces


# ------------------------------------------------------------------ 持久化
def persist_faces(conn: sqlite3.Connection, image_id: int, faces: Sequence[Face]) -> int:
    """覆盖式写入 faces 表（DELETE + INSERT），返回行数。

    landmark 以扁平 JSON 存 [x0,y0,x1,y1,...]（478 点 → 956 个数）。
    注意（工单 05 消费前必读）：重检出会清空本图旧行及其 cluster_id 关联；
    聚类落地后如需保留关联，由聚类工单负责合并策略。
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM faces WHERE image_id=?", (image_id,))
    for f in faces:
        conn.execute(
            "INSERT INTO faces(image_id, bbox, landmark, quality, detected_at) VALUES(?,?,?,?,?)",
            (image_id, json.dumps(list(f.bbox)),
             json.dumps([c for p in f.landmarks for c in p]),
             f.quality, now),
        )
    conn.commit()
    return len(faces)


def load_faces(conn: sqlite3.Connection, image_id: int) -> list[dict[str, Any]]:
    """读回 faces 表行（bbox/landmark 解析回结构），供聚类与按人寻址消费。"""
    rows = conn.execute(
        "SELECT id, cluster_id, bbox, landmark, quality, detected_at FROM faces WHERE image_id=? ORDER BY id",
        (image_id,),
    ).fetchall()
    out = []
    for r in rows:
        landmark = json.loads(r["landmark"]) if r["landmark"] else []
        out.append({
            "face_id": r["id"], "image_id": image_id, "cluster_id": r["cluster_id"],
            "bbox": tuple(json.loads(r["bbox"])), "quality": r["quality"],
            "landmark": [(landmark[i], landmark[i + 1]) for i in range(0, len(landmark), 2)],
            "detected_at": r["detected_at"],
        })
    return out
