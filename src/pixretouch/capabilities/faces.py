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

句柄生命周期：mediapipe Tasks 句柄不直接暴露给调用线程——收容于专职推理线程
（串行执行、调用带超时、超时即代际回收自愈）；实证依据与纪律 9 例外说明见
「推理容器」节与 issue #5。

MCP stdio 纪律：本模块不 print；日志一律 logging→stderr。
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import sqlite3
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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


class InferenceTimeout(RuntimeError):
    """推理容器超时（当代句柄已毒化，下次调用自愈重建）。

    产品契约（lead 裁决 2026-10-21，issue #5 / 任务板 task-10、task-11）：
    调用路径遇本异常应**重试一次**（新代际）再向上报真错；detect() 已内建
    该契约，上层消费方（assess/聚类/后续能力）无需重复处理。
    """
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

    # 2) ModelScope 兜底（modelscope_file 支持兜底源内不同布局，如嵌套 models/buffalo_l/）
    ms_id = spec.get("modelscope")
    if ms_id:
        try:
            import requests

            ms_file = spec.get("modelscope_file", spec["file"])
            url = f"{config.MODELSCOPE_BASE}/models/{ms_id}/resolve/master/{ms_file}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / (dest.name + ".part")
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


# ------------------------------------------------------------------ 推理容器
# 句柄生命周期纪律（纪律 9 对齐 + 实证例外，lead 预批准 2026-10-21，issue #5）：
#
# 纪律 9 标准形 = 模块级常驻单例 + 双检锁创建 + 无线程本地副本（ORT/torch 域沿用）。
# mediapipe Tasks 例外的实证依据：其 Python 绑定 SerialDispatcher 把 C 调用提交到
# 内部单线程 executor 后做**无超时** future.result() 等待；issue #5 复现的 A 类挂起
# 原生栈（var/qa/devb_A_hang_native_sample.txt）显示：调用线程永久停在 Python 锁
# 等待（future waiter），而两个 executor 工作线程空闲、队列为空、mediapipe C++ 池
# （drishti×16 + fiber×2）全部空闲——完成事件在绑定层内部丢失，Python 侧不可修复，
# 且非实例泄漏（实例数恒为 2）。
#
# 容器化方案（确定性 + 自愈）：
# - 全部 mediapipe 推理经队列串行到**专职 owner 线程**；句柄只在当代 owner 线程内
#   创建与使用（纪律 9「无跨线程复用」的最强形式：单使用者=单 owner 线程）；
# - 调用方带超时等待：LOAD_TIMEOUT_S（首载）与 INFER_TIMEOUT_S（单图推理）分设；
# - 超时 = 当代失联 → 毒化弃置：owner 线程为 daemon、只持有 mediapipe 句柄与纯数据
#   任务（**从不触碰 engine 会话/DB 连接/文件句柄**——弃置的是原生图资源与该线程
#   本身，Python 侧无泄漏面）；下次调用自动重建下一代；
# - 效果：把「server 永久挂起」降级为「一次结构化超时错误 + 自愈」。
# 正常路径句柄仍常驻（ADR-0003），无每调用建-关开销。

LOAD_TIMEOUT_S = 90.0     # 首次调用（import mediapipe + 建句柄 + 首推理，实测 ~5s，裕度 >15x）
INFER_TIMEOUT_S = 30.0    # 单图推理（fixture 实测 <0.5s，裕度 >60x）


class _InferenceGeneration:
    """一代推理容器：专职 owner 线程 + 该线程私有的句柄表（队列串行，无锁推理）。"""

    def __init__(self, gen_id: int):
        self.gen_id = gen_id
        self.handles: dict[str, Any] = {}   # 仅 owner 线程读写；warm 是外部只读视图
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name=f"pixretouch-face-infer-{gen_id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:                  # 优雅收场（reset_models 测试钩子）
                for h in list(self.handles.values()):
                    try:
                        h.close()
                    except Exception:        # noqa: BLE001
                        pass
                self.handles.clear()
                return
            fn, args, fut = job
            if not fut.set_running_or_notify_cancel():
                continue                     # 调用方已放弃（理论上不发生，防御）
            try:
                fut.set_result(fn(self.handles, *args))
            except BaseException as e:       # noqa: BLE001  异常随 future 回传调用方
                fut.set_exception(e)

    @property
    def warm(self) -> bool:
        return bool(self.handles)

    def submit(self, fn: Callable, *args: Any) -> Future:
        fut: Future = Future()
        self._q.put((fn, args, fut))
        return fut

    def request_shutdown(self) -> None:
        self._q.put(None)

    def thread_join(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)


class _FaceInference:
    """模块级单例（纪律 9 话语体系）：代际计数器 + 当代引用 + 创建锁。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen: _InferenceGeneration | None = None
        self.gen_counter = 0

    def call(self, fn: Callable, *args: Any) -> Any:
        """串行提交到当代 owner 线程；超时毒化当代并抛 RuntimeError（结构化，可重试）。"""
        with self._lock:
            if self._gen is None:
                self.gen_counter += 1
                self._gen = _InferenceGeneration(self.gen_counter)
            gen = self._gen
        timeout = INFER_TIMEOUT_S if gen.warm else LOAD_TIMEOUT_S
        fut = gen.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            with self._lock:
                if self._gen is gen:         # 只毒化当代，防重复回收
                    self._gen = None
            log.error("mediapipe 推理超时（>%.0fs, gen=%s）：当代句柄已弃置，下次调用自动重建",
                      timeout, gen.gen_id)
            raise InferenceTimeout(
                f"mediapipe inference timeout (>{timeout:.0f}s, gen={gen.gen_id}): "
                f"疑似绑定层丢失唤醒（SerialDispatcher future 永不完成，见 issue #5）；"
                f"当代句柄已弃置，重试本调用即自动重建"
            ) from None

    def shutdown(self, timeout: float = 15.0) -> None:
        """优雅关闭当代并同步等待 owner 退出（测试钩子；ADR-0003 常驻进程正常不调用）。

        同步 join 杜绝解释器退出期 mediapipe 句柄 GC __del__ 在 concurrent.futures
        全局 shutdown 之后触发 close 的噪声（Exception ignored: cannot schedule
        new futures after shutdown）。正常路径 owner 秒级退出；仅当代健康时调用，
        卡死代际走毒化路径而非本函数。
        """
        with self._lock:
            gen, self._gen = self._gen, None
        if gen is not None:
            gen.request_shutdown()
            gen.thread_join(timeout)


_INFERENCE = _FaceInference()


def reset_models() -> None:
    """优雅关闭当代推理容器（测试钩子；常驻进程正常不调用，ADR-0003）。"""
    _INFERENCE.shutdown()


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


def _detect_job(handles: dict[str, Any], rgb_arr, max_faces: int, det_model: str, lmk_model: str) -> list["Face"]:
    """owner 线程作业：句柄懒建（仅本线程创建/使用）→ 推理 → 转纯数据 Face 列表。

    跨线程只传纯数据（numpy 数组进、dataclass 出）；C 支撑的 result 对象不出本线程。
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    det = handles.get("detector")
    if det is None:
        det_opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=det_model),
            min_detection_confidence=DETECTION_CONFIDENCE,
        )
        det = handles["detector"] = vision.FaceDetector.create_from_options(det_opts)
    lkey = f"landmarker:{max_faces}"
    lmk = handles.get(lkey)
    if lmk is None:
        lmk_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=lmk_model),
            num_faces=max_faces,
            min_face_detection_confidence=DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_PRESENCE_CONFIDENCE,
        )
        lmk = handles[lkey] = vision.FaceLandmarker.create_from_options(lmk_opts)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr)
    h, w = rgb_arr.shape[0], rgb_arr.shape[1]

    det_boxes: list[tuple[tuple[int, int, int, int], float | None]] = []
    for d in det.detect(mp_image).detections:
        bb = d.bounding_box
        score = d.categories[0].score if d.categories else None
        det_boxes.append(((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)), score))

    faces: list[Face] = []
    used: set[int] = set()
    for landmarks in lmk.detect(mp_image).face_landmarks:
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


def detect(image_path: Path, max_faces: int = 8) -> list[Face]:
    """检测人脸 + 478 点 landmark；detector 框与 landmarker 点按 IoU 配对。

    推理经专职 owner 线程串行执行，调用带超时、超时毒化当代并自愈（见推理容器节）。
    landmarker 独有命中（detector 漏检）时 bbox 取 landmark 包络、quality=None。
    模型缺失/下载失败/推理超时抛 RuntimeError（结构化，工具层原样传播）。
    """
    import numpy as np
    from PIL import Image

    image_path = Path(image_path)
    with Image.open(image_path) as im:
        rgb_arr = np.ascontiguousarray(np.asarray(im.convert("RGB")))
    # 权重供给（含潜在网络下载）在调用方线程完成，不占推理超时预算
    det_model = str(model_path("face_detector"))
    lmk_model = str(model_path("face_landmarker"))
    try:
        return _INFERENCE.call(_detect_job, rgb_arr, max_faces, det_model, lmk_model)
    except InferenceTimeout:
        # 产品契约（issue #5）：超时 → 新代际重试一次；再超时才向上抛（当代又已毒化自愈）
        log.warning("mediapipe 推理超时，按产品契约以新代际重试一次")
        return _INFERENCE.call(_detect_job, rgb_arr, max_faces, det_model, lmk_model)


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
