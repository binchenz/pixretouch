"""人物聚类测试（工单 05 / issue #6 验收项）。

验收映射：
- 嵌入聚类跑通（合影与单人照同一人同聚类）→ test_same_person_across_images
- 不同人不串（lead 反例要求：portrait-01/02 两人 + 合影内三人）→ 同上 + test_different_persons_separate
- 四工具可用 → test_four_correction_tools
- 聚类行含 name/person_type/default_suit_id + person_type 初判可覆盖 → test_cluster_row_fields_and_override
- 嵌入持久化与幂等 → test_embeddings_persisted_and_idempotent
- op 结构化错误 → test_cluster_op_structured_errors（无模型依赖）
- kps 派生/person_type 规则 → 单元级（无模型依赖）

纪律（铁律 8 + 工单 03 范本）：主接缝 = op 函数；isolated_home 隔离库/缓存，
MODELS_DIR 共享仓库真实权重（零网络）；单人照用「从已提交 group-01 fixture 现场
裁切」合成——不依赖 gitignored 的 var/qa 文件；权重缺失 skipif；绝不污染 var/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixretouch import data
from pixretouch.capabilities import assess as A
from pixretouch.capabilities import clusters as C
from pixretouch.capabilities import faces as F

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "mini-album"

MODELS_READY = all(
    (REPO / "models" / sub / f).exists()
    for sub, f in (
        ("face_landmarker", "face_landmarker.task"),
        ("face_detector", "blaze_face_short_range.tflite"),
        ("arcface_w600k_r50", "w600k_r50.onnx"),
        ("genderage", "genderage.onnx"),
    )
)
needs_models = pytest.mark.skipif(not MODELS_READY, reason="人脸/嵌入模型权重缺失（先跑 prefetch）")
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "group-01-three-frontal.jpg").exists(), reason="mini-album fixture 缺失"
)


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "MODELS_DIR", REPO / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()
    F.reset_models()
    C.reset_models()


def _ingest(path: Path):
    from pixretouch import engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, path)


def _face_clusters(conn, image_id: int) -> dict[int, int | None]:
    rows = conn.execute("SELECT id, cluster_id FROM faces WHERE image_id=? ORDER BY id",
                        (image_id,)).fetchall()
    return {r["id"]: r["cluster_id"] for r in rows}


def _scenario(tmp_path):
    """group-01（3 脸）+ 每张脸现场合成裁切「单人照」+ portrait-01/02 → 全部 assess。

    返回 (conn, group_image_id, {parent_face_id: crop_image_id}, p1_id, p2_id)。
    """
    import cv2

    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    gid = ing["image_id"]
    assert A.assess_image_op(gid)["status"] == "ok"

    img = cv2.imread(str(Path(data.get_current(conn, gid)["output_path"])))
    H, W = img.shape[:2]
    crop_ids: dict[int, int] = {}
    for i, r in enumerate(F.load_faces(conn, gid)):
        x, y, w, h = r["bbox"]
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * 1.8
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
        p = tmp_path / f"synth-solo-{i}.jpg"
        cv2.imwrite(str(p), img[y0:y1, x0:x1])
        c2, i2 = _ingest(p)
        assert A.assess_image_op(i2["image_id"])["status"] == "ok"
        crop_ids[r["face_id"]] = i2["image_id"]

    _, p1 = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    assert A.assess_image_op(p1["image_id"])["status"] == "ok"
    _, p2 = _ingest(FIXTURES / "portrait-02-frontal-clear.jpg")
    assert A.assess_image_op(p2["image_id"])["status"] == "ok"
    return conn, gid, crop_ids, p1["image_id"], p2["image_id"]


# ------------------------------------------------------------------ 单元（无模型）
def test_kps_derivation_and_person_type_rules():
    """478 点 → ArcFace 5 点 kps 的索引映射；person_type 初判规则。"""
    import numpy as np

    lm = [(float(i), float(i * 2)) for i in range(478)]
    kps = C.kps_from_landmarks(lm)
    assert kps.shape == (5, 2) and kps.dtype == np.float32
    # 鼻尖 = 索引 1；嘴角 = 61 / 291
    assert list(kps[2]) == [1.0, 2.0]
    assert list(kps[3]) == [61.0, 122.0]
    assert list(kps[4]) == [291.0, 582.0]
    # 眼中心 = 对应眼睑 6 点均值
    re = np.mean([lm[i] for i in F.RIGHT_EYE], axis=0)
    assert np.allclose(kps[0], re, atol=1e-4)

    assert C.person_type_from(1, 30) == "male"
    assert C.person_type_from(0, 30) == "female"
    assert C.person_type_from(1, 8) == "child"
    assert C.person_type_from(0, 12) == "child"
    assert C.person_type_from(0, 13) == "female"   # 边界：13 岁不再判童

    with pytest.raises(ValueError):
        C.kps_from_landmarks([(0.0, 0.0)] * 100)  # 点数不足


def test_cluster_op_structured_errors(isolated_home):
    """空库聚类 ok；坏 id → 结构化错误不抛穿。"""
    r = C.cluster_faces_op()
    assert r["status"] == "ok" and r["clusters"] == 0
    r = C.list_clusters_op()
    assert r["status"] == "ok" and r["clusters"] == []

    assert C.rename_cluster_op(999999, name="x")["status"] == "error"
    assert C.merge_clusters_op([999998], 999999)["status"] == "error"
    assert C.assign_face_op(999999, None)["status"] == "error"


# ------------------------------------------------------------------ 集成（需模型 + fixture）
@needs_models
@needs_fixtures
def test_same_person_across_images(isolated_home):
    """验收核心：合影中的脸与其「单人照」（现场合成裁切）归入同一人物聚类。"""
    conn, gid, crop_ids, p1, p2 = _scenario(isolated_home)

    res = C.cluster_faces_op()
    assert res["status"] == "ok", res
    # 场景共 8 脸：group-01×3 + 合成单人裁切×3 + portrait-01×1 + portrait-02×1
    assert res["backfilled"] == 8 and res["created"] + res["assigned"] == 8

    group_faces = _face_clusters(conn, gid)
    assert len(group_faces) == 3
    for parent_fid, crop_img in crop_ids.items():
        crop_faces = _face_clusters(conn, crop_img)
        assert len(crop_faces) == 1, "单人裁切应恰检出 1 脸"
        crop_cid = next(iter(crop_faces.values()))
        assert crop_cid == group_faces[parent_fid], \
            f"同一人跨图必须同聚类：parent face {parent_fid} vs crop image {crop_img}"


@needs_models
@needs_fixtures
def test_different_persons_separate(isolated_home):
    """反例（lead 要求）：portrait-01 与 portrait-02 是两个人 → 不同聚类；
    合影内三人互不串；portrait 与合影人员互不串。"""
    conn, gid, crop_ids, p1, p2 = _scenario(isolated_home)
    assert C.cluster_faces_op()["status"] == "ok"

    c_p1 = next(iter(_face_clusters(conn, p1).values()))
    c_p2 = next(iter(_face_clusters(conn, p2).values()))
    crew = set(_face_clusters(conn, gid).values())
    assert c_p1 != c_p2, "两个不同的人（portrait-01/02）必须不同聚类"
    assert len(crew) == 3, "合影三人必须三个聚类（互不串）"
    assert c_p1 not in crew and c_p2 not in crew, "portrait 与合影人员不得串聚类"

    listing = C.list_clusters_op()["clusters"]
    assert len(listing) == 5
    by_id = {cl["id"]: cl for cl in listing}
    for cid in crew:
        assert by_id[cid]["face_count"] == 2
    assert by_id[c_p1]["face_count"] == 1 and by_id[c_p2]["face_count"] == 1


@needs_models
@needs_fixtures
def test_embeddings_persisted_and_idempotent(isolated_home):
    """嵌入落 faces.embedding（512 维 float32 L2 归一化 = 2048B）；重跑不新建聚类。"""
    import numpy as np

    conn, gid, crop_ids, p1, p2 = _scenario(isolated_home)
    r1 = C.cluster_faces_op()
    assert r1["status"] == "ok"
    created_first = r1["created"]

    rows = conn.execute("SELECT embedding FROM faces").fetchall()
    assert len(rows) == 8
    for r in rows:
        assert r["embedding"] is not None and len(r["embedding"]) == 2048
        v = np.frombuffer(r["embedding"], dtype=np.float32)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4, "嵌入必须 L2 归一化"

    r2 = C.cluster_faces_op()
    assert r2["status"] == "ok"
    assert r2["created"] == 0 and r2["assigned"] == 0 and r2["backfilled"] == 0, \
        "重跑幂等：无未分配脸，不新建聚类"
    assert C.list_clusters_op()["clusters"] and len(C.list_clusters_op()["clusters"]) == created_first


@needs_models
@needs_fixtures
def test_cluster_row_fields_and_override(isolated_home):
    """聚类行字段齐备；person_type 自动初判生效且可人工覆盖（工单验收）。"""
    conn, gid, crop_ids, p1, p2 = _scenario(isolated_home)
    assert C.cluster_faces_op()["status"] == "ok"

    listing = C.list_clusters_op()["clusters"]
    for cl in listing:
        assert set(cl) >= {"id", "name", "person_type", "default_suit_id", "face_count", "image_ids"}
        assert cl["person_type"] in C.PERSON_TYPES
        assert cl["default_suit_id"] is None          # 预设方案工单 11 才绑定
        # F8：faces 明细与 face_count 一致，字段可供聚类核验/按位置兜底命名
        assert len(cl["faces"]) == cl["face_count"]
        for fd in cl["faces"]:
            assert {"face_id", "image_id", "bbox", "quality"} <= set(fd)
            assert len(fd["bbox"]) == 4 and all(isinstance(v, int) for v in fd["bbox"])
    # genderage 初判生效： Apollo 11 机组（男性成人）应判 male，而非 unknown
    crew_cids = set(_face_clusters(conn, gid).values())
    crew_types = {cl["person_type"] for cl in listing if cl["id"] in crew_cids}
    assert "male" in crew_types, f"genderage 初判应识别机组为 male，实际 {crew_types}"

    # 人工覆盖：命名 + person_type 改写并持久化
    cid = sorted(crew_cids)[0]
    r = C.rename_cluster_op(cid, name="机长", person_type="child")
    assert r["status"] == "ok" and r["name"] == "机长" and r["person_type"] == "child"
    row = conn.execute("SELECT name, person_type FROM clusters WHERE id=?", (cid,)).fetchone()
    assert row["name"] == "机长" and row["person_type"] == "child", "覆盖必须落库"
    # 非法 person_type → 结构化错误
    bad = C.rename_cluster_op(cid, person_type="alien")
    assert bad["status"] == "error" and "ValueError" in bad["error"]


@needs_models
@needs_fixtures
def test_four_correction_tools(isolated_home):
    """四个修正工具：list / rename / merge / assign_face 全链路行为。"""
    conn, gid, crop_ids, p1, p2 = _scenario(isolated_home)
    assert C.cluster_faces_op()["status"] == "ok"

    # list
    listing = C.list_clusters_op()
    assert listing["status"] == "ok" and len(listing["clusters"]) == 5

    # rename（命名聚类，模拟「新娘」式按人寻址前置）
    c_p1 = next(iter(_face_clusters(conn, p1).values()))
    r = C.rename_cluster_op(c_p1, name="主角")
    assert r["status"] == "ok" and r["name"] == "主角"

    # assign_face：把 portrait-02 的脸指到「主角」聚类（人工改判），再指回 None
    p2_face_id = next(iter(_face_clusters(conn, p2)))
    r = C.assign_face_op(p2_face_id, c_p1)
    assert r["status"] == "ok" and r["cluster_id"] == c_p1
    assert _face_clusters(conn, p2)[p2_face_id] == c_p1
    r = C.assign_face_op(p2_face_id, None)
    assert r["status"] == "ok" and _face_clusters(conn, p2)[p2_face_id] is None

    # merge：把 portrait-02 的聚类并入机组某聚类——先把它重新指派回自己的原聚类
    # （上一步已置 None，重新聚类会新建一个代管聚类；直接建场景：用两个机组聚类演示合并）
    crew_cids = sorted(set(_face_clusters(conn, gid).values()))
    src, dst = crew_cids[0], crew_cids[1]
    # default_suit 继承：源聚类预置方案绑定（工单 11 语义的前置字段）
    conn.execute("UPDATE clusters SET default_suit_id=77 WHERE id=?", (src,))
    conn.commit()
    before_faces = conn.execute("SELECT COUNT(*) c FROM faces WHERE cluster_id=?", (src,)).fetchone()["c"]
    r = C.merge_clusters_op([src], dst)
    assert r["status"] == "ok" and r["faces_moved"] == before_faces
    assert r["default_suit_inherited"] == 77, "target 无绑定时必须继承源的默认方案"
    row = conn.execute("SELECT default_suit_id FROM clusters WHERE id=?", (dst,)).fetchone()
    assert row["default_suit_id"] == 77
    assert conn.execute("SELECT COUNT(*) c FROM clusters WHERE id=?", (src,)).fetchone()["c"] == 0, \
        "源聚类行必须删除"
    assert conn.execute("SELECT COUNT(*) c FROM faces WHERE cluster_id=?", (src,)).fetchone()["c"] == 0

    # 合并后 list 计数 -1；重跑聚类不会复活已删聚类（人工修正优先：已分配脸不动）
    assert len(C.list_clusters_op()["clusters"]) == 4
    r = C.cluster_faces_op()
    assert r["status"] == "ok" and r["created"] <= 1  # 至多为 assign None 的那张脸新建
    # 幂等护栏：merge 不存在的聚类 → 结构化错误
    assert C.merge_clusters_op([999999], dst)["status"] == "error"
