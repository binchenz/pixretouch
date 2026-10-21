"""人脸增强测试（工单 07 / issue #8 验收项）。

验收映射：
- 强度参数控制混合比例且默认 ≤0.3   → test_default_strength_conservative + 单调性断言
- 按人寻址生效（cluster_id）         → test_cluster_addressing_isolation
- 人脸区域清晰度提升、人脸外区域不变 → test_enhance_blurred_face_region
  （mask 口径，lead 裁决：mask=0 严格不变区 <JPEG 噪声级；mask>0.5 核心区做变化
   断言；羽化过渡带 0<mask≤0.5 为合法变化区不断言）
- 权重经镜像源下载并缓存             → needs_models 门（models/codeformer/）+ 供给链记录于 issue #8
- 缓存/RGBA/结构化错误（工单 03/08 模式）→ test_cache_semantics / test_rgba_passthrough / test_validation_errors_structured

纪律：主接缝 = op 函数；isolated_home 隔离；权重缺失 skipif；属性式断言。
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from pixretouch import data
from pixretouch.capabilities import assess as A
from pixretouch.capabilities import clusters as C
from pixretouch.capabilities import face_enhance as FE
from pixretouch.capabilities import faces as F

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
needs_models = pytest.mark.skipif(not MODELS_READY, reason="人脸/增强模型权重缺失（先跑 prefetch）")
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "waste-blur-01.jpg").exists(), reason="mini-album fixture 缺失"
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


def _arr(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _mask_arr(mask_path: str):
    import numpy as np
    from PIL import Image

    with Image.open(mask_path) as im:
        return np.asarray(im.convert("L")).astype(np.float32) / 255.0


def _core_sharpness(arr, core_mask) -> float:
    """核心区（mask>0.5 的 bbox 内、且 mask>0.5 像素）拉普拉斯方差。"""
    import cv2
    import numpy as np

    ys, xs = np.nonzero(core_mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    gray = cv2.cvtColor(arr[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float64)
    sub = core_mask[y0:y1, x0:x1] > 0.5
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[sub].var())


# ------------------------------------------------------------------ 默认保守（验收项 1）
def test_default_strength_conservative():
    """票面：默认混合强度 ≤0.3（杜绝换脸感）；op 签名默认与常量一致。"""
    assert FE.DEFAULT_STRENGTH <= 0.3
    sig = inspect.signature(FE.face_enhance_op)
    assert sig.parameters["strength"].default == FE.DEFAULT_STRENGTH
    assert sig.parameters["cluster_id"].default is None
    assert 0.0 < FE.W_FIDELITY <= 1.0


def test_validation_errors_structured(isolated_home):
    r = FE.face_enhance_op(999999)
    assert r["status"] == "error" and "KeyError" in r["error"]
    for bad in (0.0, 1.5, -0.2):
        r = FE.face_enhance_op(1, strength=bad)
        assert r["status"] == "error" and "ValueError" in r["error"], bad


# ------------------------------------------------------------------ 模糊脸功能（验收项 3）
@needs_models
@needs_fixtures
def test_enhance_blurred_face_region(isolated_home):
    """模糊脸（waste-blur-01，σ7 高斯废片）：核心区清晰度显著提升、mask=0 区严格不变、
    强度单调（0.3 < 1.0 的核心区变化量）。"""
    import numpy as np

    conn, ing = _ingest(FIXTURES / "waste-blur-01.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"          # 人脸落库
    orig = _arr(data.get_current(conn, img)["output_path"])

    strong = FE.face_enhance_op(img, strength=1.0)
    assert strong["status"] == "ok", strong
    assert Path(strong["output_path"]).read_bytes()[:8] != b"" and len(strong["mask_sha256"]) == 64
    hist = data.history(conn, img)
    assert [h["capability"] for h in hist] == ["base", "face_enhance"]
    row = conn.execute("SELECT cluster_id, params FROM records WHERE id=?",
                       (strong["record_id"],)).fetchone()
    assert row["cluster_id"] is None and '"face_boxes"' in row["params"]

    mask = _mask_arr(strong["mask_path"])
    out = _arr(strong["output_path"])

    # ① mask=0 严格不变区（lead 裁决口径）：仅 JPEG 重编码噪声
    zero = mask == 0
    assert zero.any()
    d_zero = float(np.abs(out[zero].astype(np.float32) - orig[zero].astype(np.float32)).mean())
    assert d_zero < 6.0, f"mask=0 区必须严格不变（噪声级）: {d_zero:.2f}"

    # ② 核心区清晰度提升（CodeFormer 修复模糊脸的功能正断言）
    sharp_in = _core_sharpness(orig, mask)
    sharp_out = _core_sharpness(out, mask)
    assert sharp_out >= 3.0 * max(sharp_in, 1e-6), f"核心区清晰度应显著提升: {sharp_in:.1f} -> {sharp_out:.1f}"
    assert sharp_out - sharp_in >= 20.0

    # ③ 强度单调：回滚后弱强度的核心区变化量 < 强强度
    d_strong = float(np.abs(out[mask > 0.5].astype(np.float32) - orig[mask > 0.5].astype(np.float32)).mean())
    data.rollback_to(conn, img, hist[0]["id"])
    weak = FE.face_enhance_op(img, strength=0.3)
    assert weak["status"] == "ok" and weak["cache_hit"] is False
    out_w = _arr(weak["output_path"])
    d_weak = float(np.abs(out_w[mask > 0.5].astype(np.float32) - orig[mask > 0.5].astype(np.float32)).mean())
    assert d_weak < 0.6 * d_strong, f"强度应控制混合比例: weak={d_weak:.2f} strong={d_strong:.2f}"


# ------------------------------------------------------------------ 按人寻址 + 区域隔离（验收项 2）
@needs_models
@needs_fixtures
def test_cluster_addressing_isolation(isolated_home):
    """group-01：只对聚类 A 增强——A 核心区变化、B 核心区不变（mask 口径）；
    records.cluster_id 落专列。"""
    import numpy as np

    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    assert C.cluster_faces_op()["status"] == "ok"
    clusters = C.list_clusters_op()["clusters"]
    assert len(clusters) == 3
    cid_a, cid_b = clusters[0]["id"], clusters[1]["id"]

    orig = _arr(data.get_current(conn, img)["output_path"])
    r = FE.face_enhance_op(img, cluster_id=cid_a, strength=1.0)
    assert r["status"] == "ok"
    row = conn.execute("SELECT cluster_id FROM records WHERE id=?", (r["record_id"],)).fetchone()
    assert row["cluster_id"] == cid_a, "按人寻址必须落 records.cluster_id 专列"

    out = _arr(r["output_path"])
    mask_a = _mask_arr(r["mask_path"])
    d_a = float(np.abs(out[mask_a > 0.5].astype(np.float32) - orig[mask_a > 0.5].astype(np.float32)).mean())
    assert d_a > 2.0, f"A 聚类核心区应变化: {d_a:.2f}"

    # B 聚类脸的椭圆区（未被寻址）必须近似不变——用 B 脸 bbox 内缩区做几何判据
    face_b = [f for f in F.load_faces(conn, img) if f["cluster_id"] == cid_b][0]
    x, y, w, h = face_b["bbox"]
    ix, iy = x + int(w * 0.3), y + int(h * 0.3)
    iw, ih = int(w * 0.4), int(h * 0.4)
    d_b = float(np.abs(out[iy:iy + ih, ix:ix + iw].astype(np.float32)
                       - orig[iy:iy + ih, ix:ix + iw].astype(np.float32)).mean())
    assert d_b < 6.0, f"B 聚类（未寻址）应近似不变: {d_b:.2f}"
    assert d_a > max(2.0, 2 * d_b)
    # mask=0 严格不变（全图口径）
    zero = mask_a == 0
    d_zero = float(np.abs(out[zero].astype(np.float32) - orig[zero].astype(np.float32)).mean())
    assert d_zero < 6.0


# ------------------------------------------------------------------ 缓存/RGBA（工单 03/08 模式）
@needs_models
@needs_fixtures
def test_cache_semantics(isolated_home):
    """缓存键链位置敏感：不回滚同参必 miss；回滚同参命中且哈希一致。"""
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"

    first = FE.face_enhance_op(img, strength=0.3)
    assert first["status"] == "ok" and first["cache_hit"] is False
    second = FE.face_enhance_op(img, strength=0.3)
    assert second["cache_hit"] is False, "不回滚连续同参必 miss（链位置敏感）"
    data.rollback_to(conn, img, data.history(conn, img)[0]["id"])
    hit = FE.face_enhance_op(img, strength=0.3)
    assert hit["cache_hit"] is True and hit["output_sha256"] == first["output_sha256"]


@needs_models
@needs_fixtures
def test_rgba_passthrough(isolated_home):
    """RGBA 链态（cutout → face_enhance）：alpha 逐字节直通、产物 PNG 无损（纪律 8）。"""
    import numpy as np
    from PIL import Image

    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"

    from pixretouch.capabilities import cutout as CO

    co = CO.cutout_op(img)
    if co["status"] != "ok":
        pytest.skip(f"cutout 不可用（{co.get('error')}），RGBA 断言跳过")
    with Image.open(co["output_path"]) as im:
        assert im.mode == "RGBA"
        alpha_before = np.asarray(im.getchannel("A"))

    fe = FE.face_enhance_op(img, strength=0.5)
    assert fe["status"] == "ok"
    assert Path(fe["output_path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(fe["output_path"]) as im:
        assert im.mode == "RGBA"
        alpha_after = np.asarray(im.getchannel("A"))
    assert np.array_equal(alpha_before, alpha_after), "alpha 必须逐字节直通（纪律 8）"
