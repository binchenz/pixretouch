"""人物聚类与修正工具（工单 05 / issue #6）。

消费工单 04 的 faces 表：ArcFace(w600k_r50) 嵌入回填 faces.embedding →
余弦阈值增量聚类 → clusters 行（name / person_type / default_suit_id）；
四个修正工具 list/rename/merge/assign_face 供人工收敛（自然语言驱动由
skill 层负责，本模块只提供确定性工具面）。人物聚类是按人寻址的寻址单位
（GLOSSARY），本模块产出的 cluster_id 即工单 06/07 的寻址键。

person_type 自动初判（可被 rename_cluster 覆盖，工单要求「自动初判可覆盖」）：
- genderage.onnx 输出 (gender, age)：age < CHILD_AGE_MAX → "child"；
  否则按 gender argmax → "female"/"male"（索引映射经 fixture 实测标定，
  证据记 issue #6）。
- 已知局限（lead 裁决留档）：genderage 的年龄回归误差较大（老龄/儿童边界尤甚），
  child 判定只是初判信号；性别为二值分类，非二元性别语义之外的表达不支持——
  人工覆盖（rename_cluster）永远优先。

供给链（沿用 faces.model_path 三级供给，哈希钉死）：
- w600k_r50.onnx sha256=4c06341c…e43（HF 三独立源逐字节一致：DavidHoa/buffalo_l、
  vjump21848/buffalo_l_unzip、public-data/insightface；174,383,860 B）；
- genderage.onnx sha256=4fde69b1…2fb（同三源一致；1,322,532 B）；
- insightface 自带下载器永不触发：ArcFaceONNX/Attribute 以显式本地路径 +
  预建 session 构造（其 get_model/download 路径不参与）。

确定性纪律（与工单 08 ORT 约束对齐）：
- onnxruntime 显式锁 CPUExecutionProvider——同一人脸跨进程/跨机器嵌入字节一致，
  聚类可复现（CoreML EP 输出字节可能不同，弃用；CPU 单脸 ~百 ms，相册规模足够）。
- 嵌入 = float32 512 维 L2 归一化，存 faces.embedding（2048 B blob）；
  余弦相似度 = 归一化向量点积。

增量聚类算法（确定性，单遍）：按 (image_id, face_id) 升序遍历未分配人脸；
与现有全部聚类质心（成员归一化嵌入的均值再归一化）算余弦；最优 ≥ COSINE_THRESHOLD
→ 入该聚类并增量更新质心；否则新建聚类。O(F×C)，相册规模毫秒级。
重跑安全：已分配人脸不动（人工修正不被覆盖），只处理 cluster_id IS NULL 的行。

MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .. import config, data
from ..engine import get_session
from . import faces as faces_infra
from . import weight

log = logging.getLogger("pixretouch.clusters")

# ------------------------------------------------------------------ 权重登记
weight("arcface_w600k_r50", {
    "repo": "DavidHoa/buffalo_l",          # hf-mirror 主源（三源交叉哈希一致）
    "file": "w600k_r50.onnx",
    "modelscope": "muse/insightface_model_buffalo_l",
    "modelscope_file": "models/buffalo_l/w600k_r50.onnx",   # 兜底源内嵌套布局
    "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
})
weight("genderage", {
    "repo": "DavidHoa/buffalo_l",
    "file": "genderage.onnx",
    "modelscope": "muse/insightface_model_buffalo_l",
    "modelscope_file": "models/buffalo_l/genderage.onnx",
    "sha256": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
})

# 阈值经 fixture 校准（同一人分布 vs 不同人分布的间隔，证据见 issue #6）
COSINE_THRESHOLD = 0.42
CHILD_AGE_MAX = 13          # genderage 年龄初判低于此 → "child"
PERSON_TYPES = ("female", "male", "child", "unknown")

# MediaPipe 478 点 → ArcFace 对齐所需 5 点（norm_crop 的 arcface_dst 模板序：
# 图像左眼、图像右眼、鼻尖、图像左嘴角、图像右嘴角）
NOSE_TIP = 1
MOUTH_CORNER_IMAGE_LEFT = 61
MOUTH_CORNER_IMAGE_RIGHT = 291


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ 模型单例
# 纪律 9 标准形（insightface 底层为 ORT，dev-a 确认无 SerialDispatcher 型风险面，
# 无需 faces.py 式容器化例外）：模块级常驻单例 + 双检锁创建 + 推理不在加载锁内
# + 无线程本地副本。
_model_cache: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _cpu_session(path: Path):
    """预建 CPU-only ORT session（确定性纪律；注入模型类，绕开其默认 EP 路由）。"""
    import onnxruntime

    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    return onnxruntime.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def rec_model():
    """ArcFace w600k_r50 识别模型（懒加载常驻，ADR-0003；纪律 9 双检锁）。"""
    m = _model_cache.get("rec")
    if m is None:
        with _MODEL_LOCK:
            m = _model_cache.get("rec")
            if m is None:
                from insightface.model_zoo import ArcFaceONNX

                p = faces_infra.model_path("arcface_w600k_r50")
                m = _model_cache["rec"] = ArcFaceONNX(model_file=str(p), session=_cpu_session(p))
                log.info("ArcFace 常驻（CPU EP）：%s", p)
    return m


def attr_model():
    """genderage 属性模型（懒加载常驻；纪律 9 双检锁）。"""
    m = _model_cache.get("attr")
    if m is None:
        with _MODEL_LOCK:
            m = _model_cache.get("attr")
            if m is None:
                from insightface.model_zoo import Attribute

                p = faces_infra.model_path("genderage")
                m = _model_cache["attr"] = Attribute(model_file=str(p), session=_cpu_session(p))
                log.info("genderage 常驻（CPU EP）：%s", p)
    return m


def reset_models() -> None:
    """释放常驻模型（测试钩子）。"""
    _model_cache.clear()


# ------------------------------------------------------------------ 嵌入
class _FaceShim(dict):
    """Attribute.get 的最小 face 契约：.bbox 属性 + dict 项写入（gender/age）。"""


def kps_from_landmarks(landmarks: Sequence[tuple[float, float]]):
    """478 点 mediapipe landmark → ArcFace 对齐 5 点（np.float32 (5,2)）。"""
    import numpy as np

    pts = list(landmarks)
    if len(pts) < 478:
        raise ValueError(f"landmark 点数不足 478：{len(pts)}")

    def mean_idx(idxs):
        xs = sum(pts[i][0] for i in idxs) / len(idxs)
        ys = sum(pts[i][1] for i in idxs) / len(idxs)
        return [xs, ys]

    eye_image_left = mean_idx(faces_infra.RIGHT_EYE)    # RIGHT_EYE = 图像左侧眼
    eye_image_right = mean_idx(faces_infra.LEFT_EYE)
    kps = [
        eye_image_left,
        eye_image_right,
        [pts[NOSE_TIP][0], pts[NOSE_TIP][1]],
        [pts[MOUTH_CORNER_IMAGE_LEFT][0], pts[MOUTH_CORNER_IMAGE_LEFT][1]],
        [pts[MOUTH_CORNER_IMAGE_RIGHT][0], pts[MOUTH_CORNER_IMAGE_RIGHT][1]],
    ]
    return np.asarray(kps, dtype=np.float32)


def embed_face(image_bgr, landmarks) -> Any:
    """单脸 ArcFace 嵌入：norm_crop(112) → get_feat → L2 归一化 float32 (512,)。"""
    import numpy as np
    from insightface.utils import face_align

    rec = rec_model()
    kps = kps_from_landmarks(landmarks)
    aimg = face_align.norm_crop(image_bgr, landmark=kps, image_size=rec.input_size[0])
    emb = rec.get_feat(aimg).flatten().astype(np.float32)
    norm = float(np.linalg.norm(emb))
    if norm <= 0:
        raise RuntimeError("ArcFace 嵌入为零向量（landmark 退化？）")
    return emb / norm


def gender_age(image_bgr, bbox_xywh) -> tuple[int, int]:
    """genderage 初判：返回 (gender_argmax, age)。bbox 为 (x,y,w,h)。"""
    x, y, w, h = (float(v) for v in bbox_xywh)
    shim = _FaceShim()
    shim.bbox = [x, y, x + w, y + h]
    gender, age = attr_model().get(image_bgr, shim)
    return int(gender), int(age)


def person_type_from(gender: int, age: int) -> str:
    """(gender, age) → person_type 初判。gender 索引映射经 fixture 实测：1=male, 0=female。"""
    if age < CHILD_AGE_MAX:
        return "child"
    return "male" if gender == 1 else "female"


# ------------------------------------------------------------------ 嵌入回填
def _iter_faces(conn, image_ids: Sequence[int] | None, only_missing_emb: bool):
    sql = ("SELECT f.id, f.image_id, f.bbox, f.landmark, f.cluster_id, f.embedding IS NOT NULL AS has_emb"
           " FROM faces f")
    args: list[Any] = []
    if image_ids is not None:
        sql += f" WHERE f.image_id IN ({','.join('?' * len(image_ids))})"
        args.extend(image_ids)
    sql += " ORDER BY f.image_id, f.id"
    for r in conn.execute(sql, args).fetchall():
        if only_missing_emb and r["has_emb"]:
            continue
        yield r


def backfill_embeddings(conn: sqlite3.Connection, image_ids: Sequence[int] | None = None) -> int:
    """为缺失嵌入的 faces 行计算并落库 ArcFace 嵌入；返回新回填行数。

    图像按 image_id 分组只读一次（当前态指针产物）；landmark 缺失/损坏的行跳过
    并计数告警（不中断整批——批量失败隔离精神）。
    """
    import cv2
    import numpy as np

    rows = list(_iter_faces(conn, image_ids, only_missing_emb=True))
    if not rows:
        return 0
    filled = 0
    skipped = 0
    img_cache: dict[int, Any] = {}
    for r in rows:
        iid = r["image_id"]
        if iid not in img_cache:
            cur = data.get_current(conn, iid)
            img_cache[iid] = cv2.imread(str(Path(cur["output_path"])), cv2.IMREAD_COLOR)
        img = img_cache[iid]
        if img is None or not r["landmark"]:
            skipped += 1
            continue
        try:
            flat = json.loads(r["landmark"])
            landmarks = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            emb = embed_face(img, landmarks)
        except Exception as e:  # noqa: BLE001  单脸失败隔离
            skipped += 1
            log.warning("face %s 嵌入失败：%s: %s", r["id"], type(e).__name__, e)
            continue
        conn.execute("UPDATE faces SET embedding=? WHERE id=?",
                     (emb.astype(np.float32).tobytes(), r["id"]))
        filled += 1
    conn.commit()
    if skipped:
        log.warning("backfill: %s 脸跳过（landmark 缺失或嵌入失败）", skipped)
    return filled


# ------------------------------------------------------------------ 增量聚类
def _centroid(embeddings) -> Any:
    import numpy as np

    stack = np.stack(embeddings)
    c = stack.mean(axis=0)
    n = float(np.linalg.norm(c))
    return c / n if n > 0 else c


def _load_embeddings(conn, cluster_id: int) -> list:
    import numpy as np

    out = []
    for r in conn.execute("SELECT embedding FROM faces WHERE cluster_id=? AND embedding IS NOT NULL",
                          (cluster_id,)).fetchall():
        out.append(np.frombuffer(r["embedding"], dtype=np.float32))
    return out


def cluster_faces(
    conn: sqlite3.Connection,
    image_ids: Sequence[int] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """嵌入回填 + 余弦阈值增量聚类（确定性单遍）。已分配人脸不动（人工修正优先）。

    person_type 初判在新建聚类时经 genderage 计算；模型不可用时降级 "unknown"
    并在结果中注明（初判可覆盖，聚类本体不因属性模型缺失而失败）。
    """
    import cv2
    import numpy as np

    threshold = COSINE_THRESHOLD if threshold is None else float(threshold)
    backfilled = backfill_embeddings(conn, image_ids)

    clusters: dict[int, dict[str, Any]] = {}
    for r in conn.execute("SELECT id, name, person_type, default_suit_id FROM clusters").fetchall():
        clusters[r["id"]] = dict(r)
    centroids: dict[int, Any] = {}
    for cid in clusters:
        embs = _load_embeddings(conn, cid)
        if embs:
            centroids[cid] = _centroid(embs)

    members: dict[int, list] = {cid: _load_embeddings(conn, cid) for cid in clusters}

    where = " WHERE f.cluster_id IS NULL AND f.embedding IS NOT NULL"
    args: list[Any] = []
    if image_ids is not None:
        where += f" AND f.image_id IN ({','.join('?' * len(image_ids))})"
        args.extend(image_ids)
    pending = conn.execute(
        "SELECT f.id, f.image_id, f.bbox, f.embedding FROM faces f" + where + " ORDER BY f.image_id, f.id",
        args,
    ).fetchall()

    img_cache: dict[int, Any] = {}
    attr_degraded: str | None = None
    assigned = created = 0
    for r in pending:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        best_cid, best_sim = None, -1.0
        for cid, cen in centroids.items():
            sim = float(np.dot(emb, cen))
            if sim > best_sim:
                best_cid, best_sim = cid, sim
        if best_cid is not None and best_sim >= threshold:
            conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (best_cid, r["id"]))
            members[best_cid].append(emb)
            centroids[best_cid] = _centroid(members[best_cid])
            assigned += 1
            continue

        # 新建聚类：person_type 自动初判（genderage；失败降级 unknown 不中断）
        ptype = "unknown"
        if attr_degraded is None:
            try:
                iid = r["image_id"]
                if iid not in img_cache:
                    cur = data.get_current(conn, iid)
                    img_cache[iid] = cv2.imread(str(Path(cur["output_path"])), cv2.IMREAD_COLOR)
                if img_cache[iid] is not None:
                    g, age = gender_age(img_cache[iid], json.loads(r["bbox"]))
                    ptype = person_type_from(g, age)
            except Exception as e:  # noqa: BLE001
                attr_degraded = f"{type(e).__name__}: {e}"
                log.warning("genderage 初判失败，person_type 降级 unknown：%s", attr_degraded)
        cur = conn.execute(
            "INSERT INTO clusters(person_type, created_at) VALUES(?,?)", (ptype, _now()))
        cid = cur.lastrowid
        clusters[cid] = {"id": cid, "name": None, "person_type": ptype, "default_suit_id": None}
        conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (cid, r["id"]))
        members[cid] = [emb]
        centroids[cid] = _centroid(members[cid])
        created += 1

    conn.commit()
    result = {
        "backfilled": backfilled,
        "assigned": assigned,
        "created": created,
        "threshold": threshold,
        "clusters": len(clusters),
    }
    if attr_degraded:
        result["person_type_degraded"] = attr_degraded
    return result


# ------------------------------------------------------------------ 修正工具
def list_clusters(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT c.id, c.name, c.person_type, c.default_suit_id, c.created_at,"
        "  (SELECT COUNT(*) FROM faces f WHERE f.cluster_id=c.id) AS face_count,"
        "  (SELECT GROUP_CONCAT(DISTINCT f.image_id) FROM faces f WHERE f.cluster_id=c.id) AS image_ids"
        " FROM clusters c ORDER BY c.id"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["image_ids"] = [int(v) for v in d["image_ids"].split(",")] if d["image_ids"] else []
        # F8（#13 盲测摩擦项）：每聚类 face 明细（face_id/image_id/bbox/quality）——
        # 聚类正确性核验（不必肉眼数人头）、按位置兜底命名（左/中/右）、#15 聚类
        # 审计的第一手数据。additive 兼容：既有键与语义不变。
        face_rows = conn.execute(
            "SELECT id, image_id, bbox, quality FROM faces WHERE cluster_id=? ORDER BY image_id, id",
            (r["id"],),
        ).fetchall()
        d["faces"] = [{"face_id": f["id"], "image_id": f["image_id"],
                       "bbox": json.loads(f["bbox"]), "quality": f["quality"]}
                      for f in face_rows]
        out.append(d)
    return out


def rename_cluster(
    conn: sqlite3.Connection, cluster_id: int, name: str | None = None, person_type: str | None = None,
) -> dict[str, Any]:
    if conn.execute("SELECT id FROM clusters WHERE id=?", (cluster_id,)).fetchone() is None:
        raise KeyError(f"cluster {cluster_id} not found")
    if name is None and person_type is None:
        raise ValueError("name 与 person_type 至少给一个")
    if person_type is not None and person_type not in PERSON_TYPES:
        raise ValueError(f"person_type 须为 {PERSON_TYPES} 之一，收到 {person_type!r}")
    if name is not None:
        conn.execute("UPDATE clusters SET name=? WHERE id=?", (str(name), cluster_id))
    if person_type is not None:
        conn.execute("UPDATE clusters SET person_type=? WHERE id=?", (person_type, cluster_id))
    conn.commit()
    row = conn.execute("SELECT id, name, person_type, default_suit_id FROM clusters WHERE id=?",
                       (cluster_id,)).fetchone()
    return dict(row)


def merge_clusters(
    conn: sqlite3.Connection, source_ids: Sequence[int], target_id: int,
) -> dict[str, Any]:
    """把 source 聚类并入 target：faces 重指、source 行删除。

    target.default_suit_id 为空时继承 source 中第一个非空默认方案（预设方案绑定
    不因合并丢失，工单 11 消费）。
    """
    sources = sorted({int(s) for s in source_ids} - {int(target_id)})
    if not sources:
        raise ValueError("source_ids 为空或与 target 相同")
    for cid in [target_id, *sources]:
        if conn.execute("SELECT id FROM clusters WHERE id=?", (cid,)).fetchone() is None:
            raise KeyError(f"cluster {cid} not found")
    target = conn.execute("SELECT default_suit_id FROM clusters WHERE id=?", (target_id,)).fetchone()
    inherited = None
    if target["default_suit_id"] is None:
        for s in sources:
            row = conn.execute("SELECT default_suit_id FROM clusters WHERE id=?", (s,)).fetchone()
            if row["default_suit_id"] is not None:
                inherited = row["default_suit_id"]
                conn.execute("UPDATE clusters SET default_suit_id=? WHERE id=?", (inherited, target_id))
                break
    qs = ",".join("?" * len(sources))
    moved = conn.execute(f"UPDATE faces SET cluster_id=? WHERE cluster_id IN ({qs})",
                         [target_id, *sources]).rowcount
    conn.execute(f"DELETE FROM clusters WHERE id IN ({qs})", sources)
    conn.commit()
    return {"target_id": target_id, "merged_sources": sources, "faces_moved": moved,
            "default_suit_inherited": inherited}


def assign_face(conn: sqlite3.Connection, face_id: int, cluster_id: int | None) -> dict[str, Any]:
    """把单张人脸指派到聚类（cluster_id=None → 移出聚类待重分配）。"""
    if conn.execute("SELECT id FROM faces WHERE id=?", (face_id,)).fetchone() is None:
        raise KeyError(f"face {face_id} not found")
    if cluster_id is not None and \
            conn.execute("SELECT id FROM clusters WHERE id=?", (cluster_id,)).fetchone() is None:
        raise KeyError(f"cluster {cluster_id} not found")
    conn.execute("UPDATE faces SET cluster_id=? WHERE id=?", (cluster_id, face_id))
    conn.commit()
    return {"face_id": face_id, "cluster_id": cluster_id}


# ------------------------------------------------------------------ 工具实现（接缝）
# 沿用工单 03 模式：server.py 薄包装，逻辑在本节 <tool>_op()；结构化错误不抛穿。

def cluster_faces_op(image_ids: list[int] | None = None) -> dict[str, Any]:
    """cluster_faces 工具实现：嵌入回填 + 增量聚类（幂等，重跑只处理未分配脸）。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **cluster_faces(conn, image_ids)}
    except Exception as e:  # noqa: BLE001
        log.exception("cluster_faces(image_ids=%s) 失败", image_ids)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def list_clusters_op() -> dict[str, Any]:
    """list_clusters 工具实现。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", "clusters": list_clusters(conn)}
    except Exception as e:  # noqa: BLE001
        log.exception("list_clusters 失败")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def rename_cluster_op(cluster_id: int, name: str | None = None, person_type: str | None = None) -> dict[str, Any]:
    """rename_cluster 工具实现：命名 + person_type 人工覆盖。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **rename_cluster(conn, int(cluster_id), name=name, person_type=person_type)}
    except Exception as e:  # noqa: BLE001
        log.exception("rename_cluster(cluster_id=%s) 失败", cluster_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def merge_clusters_op(source_ids: list[int], target_id: int) -> dict[str, Any]:
    """merge_clusters 工具实现：合并被拆散的同一人。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **merge_clusters(conn, source_ids, int(target_id))}
    except Exception as e:  # noqa: BLE001
        log.exception("merge_clusters(%s -> %s) 失败", source_ids, target_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def assign_face_op(face_id: int, cluster_id: int | None = None) -> dict[str, Any]:
    """assign_face 工具实现：把误归的脸移入/移出聚类。"""
    conn, _cache = get_session()
    try:
        return {"status": "ok", **assign_face(conn, int(face_id), None if cluster_id is None else int(cluster_id))}
    except Exception as e:  # noqa: BLE001
        log.exception("assign_face(face_id=%s, cluster_id=%s) 失败", face_id, cluster_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
