"""预设方案引擎测试（工单 11 / issue #12 验收项）。

验收映射：
- 5 套种子入库含版本号            → test_seeds_and_listing
- straight 输出与输入内容一致      → test_straight_identity（零步骤恒等基准）
- 事务性（中途失败不残留半套）     → test_transactional_zero_residue（注入 boom 能力，
                                     断言链/指针/当前态零变化 + 重跑缓存命中）
- 聚类绑定后批量自动按人寻址路由   → test_cluster_routing（A 绑 rescue、请求 bw-classic：
                                     A 走 rescue person 段，B/C 走请求方案，image 段随请求方案）
- save_preset 自建可立即消费       → test_save_preset_immediately_consumable
- 单聚类覆盖应用（cluster_id）     → test_single_cluster_override
- 结构化错误                       → test_validation_errors

无模型测试面：straight/自建 image 域方案/事务性（color_grade+boom）均不触人脸模型；
需模型的仅 natural 全链与聚类路由两测（skipif 门）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pixretouch import data, presets
from pixretouch.capabilities import assess as A
from pixretouch.capabilities import clusters as C
from pixretouch.capabilities import faces as F
from pixretouch.capabilities import preset_engine as PE

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "mini-album"

MODELS_READY = all(
    (REPO / "models" / sub / f).exists()
    for sub, f in (
        ("face_landmarker", "face_landmarker.task"),
        ("face_detector", "blaze_face_short_range.tflite"),
        ("codeformer", "codeformer.onnx"),
    )
)
needs_models = pytest.mark.skipif(not MODELS_READY, reason="人脸/增强模型权重缺失")
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "portrait-01-frontal-clear.jpg").exists(), reason="mini-album fixture 缺失"
)


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


def _ingest(path: Path):
    from pixretouch import engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, path)


def _caps(conn, image_id):
    return [h["capability"] for h in data.history(conn, image_id)]


# ------------------------------------------------------------------ 种子与列表（验收项 1）
def test_seeds_and_listing(isolated_home):
    r = presets.list_presets_op()   # 触发 ensure_seeds
    assert r["status"] == "ok"
    by_name = {p["name"]: p for p in r["presets"]}
    assert set(by_name) >= {"natural", "editorial", "bw-classic", "rescue", "straight"}
    for name in ("natural", "editorial", "bw-classic", "rescue", "straight"):
        p = by_name[name]
        assert p["version"] == 1 and p["is_builtin"] is True
        assert p["recipe_version"] == 1
    assert by_name["straight"]["steps"] == [], "straight 必须是零步骤基准"
    assert len(by_name["natural"]["steps"]) == 3
    # 幂等：重复调用不重复入库
    again = presets.list_presets_op()
    assert len(again["presets"]) == len(r["presets"])


# ------------------------------------------------------------------ straight 恒等基准（验收项 4）
@needs_fixtures
def test_straight_identity(isolated_home):
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    before = data.get_current(conn, img)

    r = PE.apply_preset_op(img, "straight")
    assert r["status"] == "ok", r
    assert r["steps"] == 0 and r["record_ids"] == []
    assert r["output_sha256"] == before["output_sha256"] == ing["sha256"], \
        "straight 输出内容必须恒等于输入"
    assert _caps(conn, img) == ["base"], "零步骤 = 零记录（链不动、指针不动）"


# ------------------------------------------------------------------ 自建即消费（验收项 3）
@needs_fixtures
def test_save_preset_immediately_consumable(isolated_home):
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]

    saved = presets.save_preset_op("my-matte", [
        {"capability": "color_grade", "params": {"temperature": -8, "contrast": 10}, "scope": "image"},
    ])
    assert saved["status"] == "ok" and saved["name"] == "my-matte" and saved["version"] == 1

    r = PE.apply_preset_op(img, "my-matte")
    assert r["status"] == "ok" and r["steps"] == 1
    assert _caps(conn, img) == ["base", "color_grade"]
    assert data.get_current(conn, img)["output_sha256"] != ing["sha256"]

    # 升版：同名再存 → version 2，apply 默认取最新版
    v2 = presets.save_preset_op("my-matte", [
        {"capability": "color_grade", "params": {"temperature": 12}, "scope": "image"},
    ])
    assert v2["version"] == 2
    r2 = PE.apply_preset_op(img, "my-matte")
    assert r2["status"] == "ok" and "v2" in r2["suit"]


# ------------------------------------------------------------------ 事务性（验收项 2）
@needs_fixtures
def test_transactional_zero_residue(isolated_home, monkeypatch):
    """中途失败零记录残留：链/指针/当前态全不变；修复后重跑段段缓存命中。"""
    from pixretouch import capabilities

    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    root_sha = ing["sha256"]

    def boom_fn(image, params, ctx):
        raise RuntimeError("boom: 模拟第 2 步渲染失败")

    monkeypatch.setitem(capabilities.CAPABILITIES, "boom", {"fn": boom_fn})
    saved = presets.save_preset_op("broken-suit", [
        {"capability": "color_grade", "params": {"contrast": 12}, "scope": "image"},
        {"capability": "boom", "params": {}, "scope": "image"},
    ])
    assert saved["status"] == "ok"

    r = PE.apply_preset_op(img, "broken-suit")
    assert r["status"] == "error" and "boom" in r["error"]
    # 整体无痕：链只有 base、指针在根、当前态 = 原图哈希
    assert _caps(conn, img) == ["base"]
    cur = data.get_current(conn, img)
    assert cur["output_sha256"] == root_sha
    total = conn.execute("SELECT COUNT(*) c FROM records WHERE image_id=?", (img,)).fetchone()["c"]
    assert total == 1, "失败路径零记录残留（不许半套）"

    # 修复重跑：去掉 boom 的同参第一步 → 渲染全缓存命中（lead 设计：重跑极快）
    fixed = presets.save_preset_op("broken-suit", [
        {"capability": "color_grade", "params": {"contrast": 12}, "scope": "image"},
    ])
    assert fixed["version"] == 2
    r2 = PE.apply_preset_op(img, "broken-suit")
    assert r2["status"] == "ok" and r2["steps"] == 1
    assert r2["cache_hits"] == 1, "第一段渲染已在缓存（事务阶段①的产物），重跑必须命中"
    assert _caps(conn, img) == ["base", "color_grade"]


# ------------------------------------------------------------------ 结构化错误（无模型）
def test_validation_errors(isolated_home):
    assert PE.apply_preset_op(1, "no-such-suit")["status"] == "error"       # 方案不存在
    assert PE.apply_preset_op(999999, "straight")["status"] == "error"      # 图像不存在
    r = presets.save_preset_op("bad", [{"capability": "no_such_cap", "params": {}, "scope": "image"}])
    assert r["status"] == "error" and "ValueError" in r["error"]
    r = presets.save_preset_op("bad2", [{"capability": "color_grade", "params": {}, "scope": "person"}])
    assert r["status"] == "error" and "person" in r["error"]                # person 域白名单
    r = presets.bind_cluster_suit_op(999999, None)
    assert r["status"] == "error" and "KeyError" in r["error"]


# ------------------------------------------------------------------ natural 全链（需模型）
@needs_models
@needs_fixtures
def test_natural_full_chain(isolated_home):
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]

    r = PE.apply_preset_op(img, "natural")
    assert r["status"] == "ok", r
    assert r["steps"] == 3 and len(r["record_ids"]) == 3 and r["suit"] == "natural@v1"
    # 链上顺序：person 段（配方序）→ image 段
    assert _caps(conn, img) == ["base", "smooth_skin", "brighten_eyes", "color_grade"]
    # person 记录：未聚类 → cluster_id None，但 face_boxes 可审计
    rows = conn.execute(
        "SELECT capability, cluster_id, params, mask_sha256 FROM records WHERE image_id=? ORDER BY id",
        (img,)).fetchall()
    for row in rows[1:3]:
        assert row["cluster_id"] is None
        assert '"face_boxes"' in row["params"]
        assert len(row["mask_sha256"]) == 64
    assert rows[3]["cluster_id"] is None and rows[3]["mask_sha256"] is None
    # mask 双登记：两个 person 步骤各一份固化文件
    assert len(r["mask_paths"]) == 2
    for p in r["mask_paths"].values():
        assert Path(p).exists()
    assert data.get_current(conn, img)["output_sha256"] != ing["sha256"]


# ------------------------------------------------------------------ 按人路由（验收项 3 核心）
@needs_models
@needs_fixtures
def test_cluster_routing(isolated_home):
    """A 聚类绑 rescue、请求 bw-classic：A 走 rescue person 段（face_enhance 特征步），
    B/C 走请求方案 person 段，image 段（bw LUT）随请求方案全图一次。"""
    import cv2
    import numpy as np
    from PIL import Image

    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    assert C.cluster_faces_op()["status"] == "ok"
    clusters = C.list_clusters_op()["clusters"]
    assert len(clusters) == 3
    cid_a = clusters[0]["id"]

    rescue_id = next(p["id"] for p in presets.list_presets_op()["presets"] if p["name"] == "rescue")
    b = presets.bind_cluster_suit_op(cid_a, rescue_id)
    assert b["status"] == "ok" and b["default_suit_id"] == rescue_id

    r = PE.apply_preset_op(img, "bw-classic")
    assert r["status"] == "ok", r
    # 计划：A(rescue person×3) + B(smooth) + C(smooth) + image(bw color) = 6 段
    assert r["steps"] == 6, r
    caps = _caps(conn, img)
    assert caps[0] == "base"
    assert caps.count("face_enhance") == 1, "只有绑定了 rescue 的 A 聚类走 face_enhance"
    assert caps[-1] == "color_grade", "image 段收尾"
    assert caps[1:-1].count("smooth_skin") == 3, "A(rescue)+B+C 各一次磨皮"

    rows = conn.execute(
        "SELECT capability, cluster_id FROM records WHERE image_id=? ORDER BY id", (img,)).fetchall()
    enhance_rows = [x for x in rows if x["capability"] == "face_enhance"]
    assert enhance_rows and enhance_rows[0]["cluster_id"] == cid_a, "face_enhance 必须记在 A 聚类名下"
    assert any(s.startswith("rescue") for s in r["routing"]), r["routing"]

    # 功能侧证：bw LUT 生效 → 输出饱和度坍塌
    with Image.open(r["output_path"]) as im:
        out = np.asarray(im.convert("RGB"))
    with Image.open(FIXTURES / "group-01-three-frontal.jpg") as im:
        src = np.asarray(im.convert("RGB"))
    s_out = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 1].mean()
    s_src = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)[..., 1].mean()
    assert s_out < 0.3 * s_src, f"bw-classic 应坍塌饱和度: {s_src:.1f} -> {s_out:.1f}"


# ------------------------------------------------------------------ 单聚类覆盖（Spec 用户故事 8）
@needs_models
@needs_fixtures
def test_single_cluster_override(isolated_home):
    """cluster_id 指定 = 只执行该聚类 person 段、跳过 image 段（个案特判语义）。"""
    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    assert C.cluster_faces_op()["status"] == "ok"
    cid = C.list_clusters_op()["clusters"][1]["id"]

    r = PE.apply_preset_op(img, "natural", cluster_id=cid)
    assert r["status"] == "ok", r
    caps = _caps(conn, img)
    assert caps == ["base", "smooth_skin", "brighten_eyes"], "image 域步骤必须跳过"
    rows = conn.execute("SELECT cluster_id FROM records WHERE image_id=? AND id>1 ORDER BY id",
                        (img,)).fetchall()
    assert all(x["cluster_id"] == cid for x in rows)


# ------------------------------------------------------------------ 无人脸图两分法（dev-a 语义裁决）
@needs_models
def test_person_steps_skipped_on_faceless_image(isolated_home):
    """图级方案套零人脸图（合成风景）：person 域跳过 + person_skipped 审计字段，
    image 域照常执行（套系 =「可应用的都应用」，不整单失败）。"""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:480, 0:640]
    arr = np.dstack([
        (xx / 640 * 200).astype(np.uint8),
        (yy / 480 * 160).astype(np.uint8),
        rng.integers(60, 90, (480, 640)).astype(np.uint8),
    ])
    p = isolated_home / "landscape.jpg"
    Image.fromarray(arr).save(p, quality=92)
    conn, ing = _ingest(p)
    img = ing["image_id"]

    r = PE.apply_preset_op(img, "natural")
    assert r["status"] == "ok", r
    assert r["steps"] == 1, "natural=2 person 段+1 image 段：零人脸只剩 image 段"
    assert "未检出人脸" in r.get("person_skipped", ""), r.get("person_skipped")
    assert _caps(conn, img) == ["base", "color_grade"]
    assert r["mask_paths"] == {}


@needs_models
@needs_fixtures
def test_cluster_override_addressing_error(isolated_home):
    """显式 cluster_id 寻址到本图无脸的聚类 → 响亮失败（寻址错误不静默跳过），零残留。"""
    conn, ing1 = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    p1 = ing1["image_id"]
    assert A.assess_image_op(p1)["status"] == "ok"
    assert C.cluster_faces_op([p1])["status"] == "ok"
    cid = C.list_clusters_op()["clusters"][0]["id"]

    _c2, ing2 = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    g = ing2["image_id"]
    assert A.assess_image_op(g)["status"] == "ok"   # group 有脸，但均不属于 cid

    r = PE.apply_preset_op(g, "natural", cluster_id=cid)
    assert r["status"] == "error" and "ValueError" in r["error"]
    assert _caps(conn, g) == ["base"], "寻址失败必须零痕迹（事务性）"
