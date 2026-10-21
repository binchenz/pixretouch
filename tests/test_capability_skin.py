"""皮肤磨皮与牙眼测试（工单 06 / issue #7 验收项）。

验收映射：
- face-parsing 产出皮肤/牙齿/眼部 mask 并参与渲染 → test_three_ops_mask_dual_registration
- 频率分离（低频均匀、高频纹理保留，fixture 可观测）→ test_freq_separation_texture_preserved
- 三工具接受 cluster_id，按人寻址生效 → test_cluster_addressing_isolation
- 同图不同聚类不同参数互不串扰（区域隔离断言）→ test_cluster_addressing_isolation
- 记录链/缓存/RGBA 纪律（工单 03/08 模式）→ test_cache_hit_and_rgba_passthrough
- 牙眼行为（闭嘴优雅近恒等 / 亮眼可观测）→ test_eyes_brighten_teeth_graceful
- 结构化错误 → test_validation_errors_structured

纪律（铁律 8 + 工单 03 范本）：主接缝 = op 函数；isolated_home 隔离库/缓存/临时产物，
MODELS_DIR 共享仓库真实权重（零网络）；权重缺失 skipif；属性式断言不绑模型字节；
区域隔离用「变化区 > max(绝对阈值, 2×未变区)」相对判据（JPEG 重编码噪声鲁棒）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pixretouch import data
from pixretouch.capabilities import assess as A
from pixretouch.capabilities import clusters as C
from pixretouch.capabilities import faces as F
from pixretouch.capabilities import skin as S

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "mini-album"

MODELS_READY = (
    (REPO / "models" / "face_landmarker" / "face_landmarker.task").exists()
    and (REPO / "models" / "face_detector" / "blaze_face_short_range.tflite").exists()
)
needs_models = pytest.mark.skipif(not MODELS_READY, reason="人脸模型权重缺失（先跑 prefetch）")
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "portrait-01-frontal-clear.jpg").exists(), reason="mini-album fixture 缺失"
)


# ------------------------------------------------------------------ fixtures/helpers
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


def _arr(path) -> "np.ndarray":  # noqa: F821
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _inner(box, frac: float = 0.6):
    """脸框中心 frac 比例区域（避开边缘羽化带，取纯皮肤/五官核心）。"""
    x, y, w, h = box
    iw, ih = int(w * frac), int(h * frac)
    return (x + (w - iw) // 2, y + (h - ih) // 2, iw, ih)


def _region_mean_absdiff(a, b, box) -> float:
    import numpy as np

    x, y, w, h = box
    d = np.abs(a[y:y + h, x:x + w].astype(float) - b[y:y + h, x:x + w].astype(float))
    return float(d.mean())


def _hf_energy(arr, box) -> float:
    """区域内高频能量：mean|img − Gauss(img, σ=2)|（纹理度量）。"""
    import cv2
    import numpy as np

    x, y, w, h = box
    reg = arr[y:y + h, x:x + w].astype(np.float32)
    hf = reg - cv2.GaussianBlur(reg, (0, 0), 2.0)
    return float(abs(hf).mean())


def _lf_mean(arr, box) -> "np.ndarray":  # noqa: F821
    import cv2
    import numpy as np

    x, y, w, h = box
    reg = arr[y:y + h, x:x + w].astype(np.float32)
    return cv2.GaussianBlur(reg, (0, 0), 8.0)


def _face_box(conn, image_id: int, idx: int = 0):
    rows = F.load_faces(conn, image_id)
    return rows[idx]["bbox"]


# ------------------------------------------------------------------ 结构化错误（无模型）
def test_validation_errors_structured(isolated_home):
    r = S.smooth_skin_op(999999)
    assert r["status"] == "error" and "KeyError" in r["error"]
    # 参数校验先于图像访问：无需真图/模型
    for op, kw in ((S.smooth_skin_op, {"strength": 0.0}),
                   (S.smooth_skin_op, {"strength": 1.5}),
                   (S.whiten_teeth_op, {"amount": -0.1}),
                   (S.brighten_eyes_op, {"amount": 2.0})):
        r = op(1, **kw)
        assert r["status"] == "error" and "ValueError" in r["error"], (op, kw, r)


# ------------------------------------------------------------------ mask 双登记（验收项 1）
@needs_models
@needs_fixtures
def test_three_ops_mask_dual_registration(isolated_home):
    """三工具各产出对应 mask（皮肤/牙/眼），走 mask 通道双登记并落记录链。"""
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]

    r1 = S.smooth_skin_op(img, strength=0.6)
    r2 = S.whiten_teeth_op(img, amount=0.5)
    r3 = S.brighten_eyes_op(img, amount=0.35)
    for r in (r1, r2, r3):
        assert r["status"] == "ok", r
        assert len(r["mask_sha256"]) == 64
        mp = Path(r["mask_path"])
        assert mp.exists() and mp.stat().st_size > 0
        # 缓存产物按哈希命名无扩展名——验内容格式：PNG 魔数
        assert mp.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "mask 固化产物必须是 PNG"
        assert r["cache_hit"] is False

    # 记录链：base → smooth_skin → whiten_teeth → brighten_eyes
    hist = data.history(conn, img)
    assert [h["capability"] for h in hist] == ["base", "smooth_skin", "whiten_teeth", "brighten_eyes"]
    # mask sha 落库且与返回一致；三种 mask 互不相同（几何不同）
    for r, rec in zip((r1, r2, r3), hist[1:]):
        row = conn.execute("SELECT mask_sha256 FROM records WHERE id=?", (rec["id"],)).fetchone()
        assert row["mask_sha256"] == r["mask_sha256"]
    assert len({r1["mask_sha256"], r2["mask_sha256"], r3["mask_sha256"]}) == 3


# ------------------------------------------------------------------ 频率分离（验收项 2）
@needs_models
@needs_fixtures
def test_freq_separation_texture_preserved(isolated_home):
    """磨皮 = 低频均匀 + 高频保留：皮肤区高频能量占比高、低频显著变化；
    脸外区域近似不变；强度单调。"""
    import numpy as np

    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"          # 人脸落库
    box = _inner(_face_box(conn, img), 0.6)
    orig = _arr(data.get_current(conn, img)["output_path"])

    strong = S.smooth_skin_op(img, strength=0.95)
    assert strong["status"] == "ok"
    out_strong = _arr(strong["output_path"])

    # ① 高频纹理保留：磨皮区高频能量 ≥ 原始的 70%
    hf_ratio = _hf_energy(out_strong, box) / max(_hf_energy(orig, box), 1e-6)
    assert hf_ratio >= 0.70, f"高频纹理未保留: ratio={hf_ratio:.3f}"
    # ② 低频均匀化：磨皮区低频层发生可观测变化
    lf_diff = float(np.abs(_lf_mean(out_strong, box) - _lf_mean(orig, box)).mean())
    assert lf_diff >= 1.0, f"低频层未变化: {lf_diff:.3f}"
    # ③ 脸外区域近似不变（仅 JPEG 重编码噪声）
    x, y, w, h = _face_box(conn, img)
    far = (0, 0, w, min(60, orig.shape[0]))     # 图像顶部条带（portrait-01 顶部为暗背景）
    assert _region_mean_absdiff(orig, out_strong, far) < 6.0, "脸外区域被改动"

    # ④ 强度单调：回滚到根后弱强度的低频变化 < 强强度
    data.rollback_to(conn, img, data.history(conn, img)[0]["id"])
    weak = S.smooth_skin_op(img, strength=0.3)
    assert weak["status"] == "ok" and weak["cache_hit"] is False
    out_weak = _arr(weak["output_path"])
    lf_weak = float(np.abs(_lf_mean(out_weak, box) - _lf_mean(orig, box)).mean())
    assert lf_weak < lf_diff, f"强度不单调: weak={lf_weak:.3f} strong={lf_diff:.3f}"


# ------------------------------------------------------------------ 按人寻址 + 区域隔离（验收项 3/4）
@needs_models
@needs_fixtures
def test_cluster_addressing_isolation(isolated_home):
    """cluster_id 按人寻址 + 同图不同聚类互不串扰（区域隔离断言）。"""
    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    assert C.cluster_faces_op()["status"] == "ok"

    clusters = C.list_clusters_op()["clusters"]
    assert len(clusters) == 3
    cid_a, cid_b = clusters[0]["id"], clusters[1]["id"]

    def box_of(cid):
        row = conn.execute("SELECT bbox FROM faces WHERE image_id=? AND cluster_id=?",
                           (img, cid)).fetchone()
        import json
        return _inner(tuple(json.loads(row["bbox"])), 0.6)

    box_a, box_b = box_of(cid_a), box_of(cid_b)
    orig = _arr(data.get_current(conn, img)["output_path"])

    # 第一次：只对聚类 A 磨皮（强参数放大可观测性）
    r1 = S.smooth_skin_op(img, cluster_id=cid_a, strength=0.95)
    assert r1["status"] == "ok"
    out1 = _arr(r1["output_path"])
    rec1 = conn.execute("SELECT cluster_id FROM records WHERE id=?", (r1["record_id"],)).fetchone()
    assert rec1["cluster_id"] == cid_a, "按人寻址必须落 records.cluster_id 专列"

    d_a1 = _region_mean_absdiff(orig, out1, box_a)
    d_b1 = _region_mean_absdiff(orig, out1, box_b)
    assert d_a1 > max(4.0, 2 * d_b1), f"A 区应显著变化: dA={d_a1:.2f} dB={d_b1:.2f}"
    assert d_b1 < 6.0, f"B 区（未寻址）应近似不变: {d_b1:.2f}"

    # 第二次：只对聚类 B 磨皮 —— A 区在两次产物之间必须近似不变（互不串扰）
    r2 = S.smooth_skin_op(img, cluster_id=cid_b, strength=0.95)
    assert r2["status"] == "ok"
    out2 = _arr(r2["output_path"])
    rec2 = conn.execute("SELECT cluster_id FROM records WHERE id=?", (r2["record_id"],)).fetchone()
    assert rec2["cluster_id"] == cid_b

    d_b2 = _region_mean_absdiff(out1, out2, box_b)
    d_a2 = _region_mean_absdiff(out1, out2, box_a)
    assert d_b2 > max(4.0, 2 * d_a2), f"B 区应显著变化: dB={d_b2:.2f} dA={d_a2:.2f}"
    assert d_a2 < 6.0, f"A 区（本轮未寻址）应近似不变: {d_a2:.2f}"

    # 未指定 cluster_id → 全图人脸（第三张脸也生效）：链上第三次调用
    r3 = S.smooth_skin_op(img, strength=0.95)
    assert r3["status"] == "ok"
    row = conn.execute("SELECT cluster_id, params FROM records WHERE id=?", (r3["record_id"],)).fetchone()
    assert row["cluster_id"] is None
    assert '"face_boxes"' in row["params"], "params 必须记录目标脸（可审计 + mask 重建依据）"


# ------------------------------------------------------------------ 记录链/缓存/RGBA 纪律
@needs_models
@needs_fixtures
def test_cache_hit_and_rgba_passthrough(isolated_home):
    """缓存键链位置敏感（回滚后同参命中）；RGBA 链态 alpha 直通、产物 PNG 无损。"""
    import numpy as np
    from PIL import Image

    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"

    first = S.smooth_skin_op(img, strength=0.5)
    assert first["status"] == "ok" and first["cache_hit"] is False
    again_norollback = S.smooth_skin_op(img, strength=0.5)
    assert again_norollback["cache_hit"] is False, "不回滚连续同参必 miss（链位置敏感）"
    data.rollback_to(conn, img, data.history(conn, img)[0]["id"])
    hit = S.smooth_skin_op(img, strength=0.5)
    assert hit["cache_hit"] is True and hit["output_sha256"] == first["output_sha256"]

    # RGBA：cutout（工单 08 能力）→ 链态 RGBA → 磨皮 alpha 直通
    data.rollback_to(conn, img, data.history(conn, img)[0]["id"])
    from pixretouch.capabilities import cutout as CO

    co = CO.cutout_op(img)
    if co["status"] != "ok":
        pytest.skip(f"cutout 不可用（{co.get('error')}），RGBA 断言跳过")
    with Image.open(co["output_path"]) as im:
        assert im.mode == "RGBA"
        alpha_before = np.asarray(im.getchannel("A"))

    sm = S.smooth_skin_op(img, strength=0.8)
    assert sm["status"] == "ok"
    # 缓存产物按哈希命名无扩展名——验内容格式：RGBA 链态产物必须 PNG 无损
    assert Path(sm["output_path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(sm["output_path"]) as im:
        assert im.mode == "RGBA"
        alpha_after = np.asarray(im.getchannel("A"))
    assert np.array_equal(alpha_before, alpha_after), "alpha 必须逐字节直通（纪律 8）"


# ------------------------------------------------------------------ 牙眼行为
@needs_models
@needs_fixtures
def test_eyes_brighten_teeth_graceful(isolated_home):
    """亮眼可观测（眼区提亮）；闭嘴照牙齿美白优雅近恒等（mask 近空，记录照常追加）。"""
    import numpy as np

    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    face = F.load_faces(conn, img)[0]
    lm = face["landmark"]
    eye_pts = [lm[i] for i in list(F.RIGHT_EYE) + list(F.LEFT_EYE)]
    exs = [p[0] for p in eye_pts]
    eys = [p[1] for p in eye_pts]
    pad = int((max(exs) - min(exs)) * 0.15)
    eye_box = (max(0, int(min(exs)) - pad), max(0, int(min(eys)) - pad),
               int(max(exs) - min(exs)) + 2 * pad, int(max(eys) - min(eys)) + 2 * pad)

    orig = _arr(data.get_current(conn, img)["output_path"])
    r = S.brighten_eyes_op(img, amount=0.6)
    assert r["status"] == "ok"
    out = _arr(r["output_path"])
    gain = float(out[eye_box[1]:eye_box[1] + eye_box[3], eye_box[0]:eye_box[0] + eye_box[2]].mean()) - \
        float(orig[eye_box[1]:eye_box[1] + eye_box[3], eye_box[0]:eye_box[0] + eye_box[2]].mean())
    assert gain >= 0.8, f"眼区应可观测提亮: gain={gain:.2f}"

    # 闭嘴照（portrait-01 无露齿）：牙 mask 近空 → 全图近恒等，但记录/缓存语义完整
    t = S.whiten_teeth_op(img, amount=0.8)
    assert t["status"] == "ok"
    tout = _arr(t["output_path"])
    prev = _arr(r["output_path"])   # brighten_eyes 产物 = whiten_teeth 的输入链态
    diff = float(np.abs(tout.astype(np.float32) - prev.astype(np.float32)).mean())
    assert diff < 3.0, f"闭嘴照牙齿美白应近恒等: 全图均值差 {diff:.2f}"
    hist = data.history(conn, img)
    assert hist[-1]["capability"] == "whiten_teeth"


@needs_models
def test_whiten_teeth_functional_smile(isolated_home):
    """露齿微笑功能正例（qa fixture，Pexels 1181686）：牙 mask 非空、牙区 V 通道
    可观测上升、效果集中于牙区（额头皮肤对照近不变）。"""
    import cv2
    import numpy as np
    from PIL import Image

    src = FIXTURES / "smile-teeth-01-frontal.jpg"
    if not src.exists():
        pytest.skip("smile fixture 缺失（qa cd5fad1）")
    conn, ing = _ingest(src)
    img = ing["image_id"]
    assert A.assess_image_op(img)["status"] == "ok"
    face = F.load_faces(conn, img)[0]
    lm = face["landmark"]
    lip_pts = [lm[i] for i in S.LIPS_OUTER]
    lxs = [p[0] for p in lip_pts]
    lys = [p[1] for p in lip_pts]
    lx0, ly0 = int(min(lxs)), int(min(lys))
    lw, lh = int(max(lxs) - min(lxs)), int(max(lys) - min(lys))
    teeth_box = (lx0 + int(lw * 0.25), ly0 + int(lh * 0.30), int(lw * 0.5), int(lh * 0.40))
    forehead_box = (lx0 + lw // 4, max(0, ly0 - int(lh * 1.6)), lw // 2, max(8, int(lh * 0.6)))

    def v_mean(arr, box):
        xx, yy, ww, hh = box
        hsv = cv2.cvtColor(arr[yy:yy + hh, xx:xx + ww], cv2.COLOR_RGB2HSV)
        return float(hsv[..., 2].mean())

    orig = _arr(data.get_current(conn, img)["output_path"])
    r = S.whiten_teeth_op(img, amount=1.0)
    assert r["status"] == "ok"

    # 牙 mask 非空（露齿照）
    with Image.open(r["mask_path"]) as mim:
        marr = np.asarray(mim)
    assert marr.max() > 0, "露齿照牙 mask 必须非空"
    assert (marr > 12).sum() > 200, "牙 mask 应覆盖成片牙区（通道序回归护栏，issue #7）"

    out = _arr(r["output_path"])
    v_rise = v_mean(out, teeth_box) - v_mean(orig, teeth_box)
    assert v_rise >= 1.5, f"牙区 V 通道应可观测上升: {v_rise:.2f}"
    d_teeth = _region_mean_absdiff(orig, out, teeth_box)
    d_forehead = _region_mean_absdiff(orig, out, forehead_box)
    assert d_teeth > max(1.5, 2 * d_forehead), \
        f"美白应集中于牙区: teeth={d_teeth:.2f} forehead={d_forehead:.2f}"

    # 第二样本（同人裁切，qa 提供）：op 可用性冒烟（牙可见性不做硬断言）
    src2 = FIXTURES / "smile-teeth-02-pd-crop.jpg"
    if src2.exists():
        _c2, ing2 = _ingest(src2)
        img2 = ing2["image_id"]
        assert A.assess_image_op(img2)["status"] == "ok"
        r2 = S.whiten_teeth_op(img2, amount=0.8)
        assert r2["status"] == "ok" and r2["mask_sha256"] != r["mask_sha256"]
