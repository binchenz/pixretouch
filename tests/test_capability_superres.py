"""工具接缝冒烟测试：super_resolve（工单 10 验收项）。

纪律（Spec · Testing Decisions + 铁律 8 + 工单03 模式）：
- 走 MCP 工具函数入口（server.super_resolve）——主接缝；
- 只断言外部行为：倍率语义（输出尺寸精确=输入×scale）、分块拼接无可见接缝
  （**分块输出 vs 整图单发输出一致性**——重叠区一致性的最强属性式断言，
  不绑定模型字节；另加接缝带偏差不集中断言）、记录链+1、回滚恢复、缓存命中、
  RGBA 链态 alpha lanczos 同步放大（字节级相等）、尺寸守卫结构化报错；
- tmp_path 隔离库/缓存；权重缺失 skipif；绝不污染 var/。
- 40MP 级无 OOM 实证不入 pytest（单轮 ~40min 会拖垮共享套件）——以一次性后台
  实证留档 issue #11（峰值 RSS + 耗时 + 边界测算），qa 可按同法复跑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
X2_PT = REPO_ROOT / "models" / "realesrgan_x2" / "real_esrgan_x2.onnx"
X4_PT = REPO_ROOT / "models" / "realesrgan_x4" / "real_esrgan_x4.onnx"

requires_esrgan = pytest.mark.skipif(
    not (X2_PT.exists() and X4_PT.exists()), reason="Real-ESRGAN 权重未预取"
)
requires_isnet = pytest.mark.skipif(
    not (REPO_ROOT / "models" / "isnet" / "isnet-general-use.onnx").exists(),
    reason="ISNet 权重未预取",
)


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "MODELS_DIR", REPO_ROOT / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


def _make_photo(p: Path, w: int, h: int) -> Path:
    """确定性 fixture：三通道渐变 + 每 40px 网格线 + 种子噪声（接缝敏感纹理）。"""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        (xx * 255 / max(w - 1, 1)).astype(np.float32),
        (yy * 255 / max(h - 1, 1)).astype(np.float32),
        ((xx + yy) * 255 / max(w + h - 2, 1)).astype(np.float32),
    ], axis=2)
    noise = rng.normal(0, 10, size=(h, w, 3)).astype(np.float32)
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for gx in range(0, w, 40):
        d.line([(gx, 0), (gx, h)], fill=(20, 20, 20), width=1)
    img.save(p, "PNG")
    return p


@pytest.fixture()
def photo(tmp_path) -> Path:
    return _make_photo(tmp_path / "photo.png", 480, 320)


@pytest.fixture()
def small_photo(tmp_path) -> Path:
    return _make_photo(tmp_path / "small.png", 128, 96)


def _ingest(p: Path):
    from pixretouch import data, engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, p)


@requires_esrgan
def test_scale_semantics_seam_and_cache(isolated_home, photo):
    """倍率生效 + 分块拼接无可见接缝（对照整图单发）+ 记录链/回滚/缓存。"""
    from pixretouch import data, server
    from pixretouch.capabilities import superres
    from pixretouch.capabilities.cutout import ort_session

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    res = server.super_resolve(img_id, scale=2)
    assert res["status"] == "ok", res
    out = Image.open(res["output_path"])
    # ① 倍率参数生效：输出尺寸精确 = 输入 × scale
    assert out.size == (960, 640) and (res["width"], res["height"]) == (960, 640)

    # ② 无可见接缝：分块输出 vs 整图单发输出一致（重叠区一致性最强断言）
    tiled = np.asarray(out.convert("RGB")).astype(int)
    sess = ort_session("realesrgan_x2")
    whole = superres._run_tile(sess, np.asarray(Image.open(photo).convert("RGB")), 2).astype(int)
    diff = np.abs(tiled - whole)
    assert diff.mean() < 2, f"分块与整图应一致, mean {diff.mean():.2f}"
    assert np.percentile(diff, 99) < 8, "p99 偏差应极小"
    assert diff.max() < 40, f"最大偏差不应有阶跃, max {diff.max()}"
    # 接缝带（x 向 join=480±2, y 向 join=160±2）偏差不集中
    seam_x = diff[:, 478:483].mean()
    seam_y = diff[158:163, :].mean()
    assert seam_x < 4 and seam_y < 4, f"接缝带偏差不应集中: x={seam_x:.2f} y={seam_y:.2f}"

    # ③ 记录链+1、指针、回滚恢复
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "super_resolve"]
    assert data.get_current(conn, img_id)["id"] == res["record_id"]
    data.rollback_to(conn, img_id, ing["record_id"])
    assert data.get_current(conn, img_id)["output_sha256"] == ing["sha256"]

    # ④ 回滚后同参重调 → 命中缓存（链位置敏感语义），产物哈希一致
    again = server.super_resolve(img_id, scale=2)
    assert again["cache_hit"] is True and again["output_sha256"] == res["output_sha256"]
    assert again["record_id"] != res["record_id"], "缓存命中仍追加新记录"


@requires_esrgan
def test_scale4_small(isolated_home, small_photo):
    from pixretouch import server

    conn, ing = _ingest(small_photo)
    res = server.super_resolve(ing["image_id"], scale=4)
    assert res["status"] == "ok", res
    assert Image.open(res["output_path"]).size == (512, 384), "×4 尺寸精确"


@requires_esrgan
@requires_isnet
def test_rgba_alpha_lanczos_upscaled(isolated_home, tmp_path):
    """RGBA 链态（纪律 8 几何增补）：RGB 走模型，alpha lanczos 同步放大且字节级相等。"""
    from pixretouch import server

    photo = _make_photo(tmp_path / "rgba_src.png", 320, 240)
    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    c = server.cutout(img_id)
    assert c["status"] == "ok", c
    alpha_before = np.asarray(Image.open(c["output_path"]).getchannel("A"))

    r = server.super_resolve(img_id, scale=2)
    assert r["status"] == "ok", r
    out_img = Image.open(r["output_path"])
    assert out_img.mode == "RGBA" and out_img.size == (640, 480)
    alpha_after = np.asarray(out_img.getchannel("A"))
    expected = np.asarray(
        Image.fromarray(alpha_before).resize((640, 480), Image.LANCZOS)
    )
    assert np.array_equal(alpha_after, expected), "alpha 应为 lanczos×2 确定性放大（不进生成式模型）"


@requires_esrgan
def test_size_guard_structured_error(isolated_home, photo, monkeypatch):
    """输出尺寸守卫：超限 → 结构化错误、记录链不增长（monkeypatch 阈值模拟大输入）。"""
    from pixretouch import data, server
    from pixretouch.capabilities import superres

    conn, ing = _ingest(photo)
    monkeypatch.setattr(superres, "MAX_OUT_DIM", 600)  # 960 > 600 → 拒绝
    res = server.super_resolve(ing["image_id"], scale=2)
    assert res["status"] == "error" and "600" in res["error"], f"守卫应生效并报阈值: {res}"
    assert [h["capability"] for h in data.history(conn, ing["image_id"])] == ["base"]


def test_validation_errors(isolated_home, photo):
    """scale 非法/图像不存在 → 结构化错误且不增链（无需权重）。"""
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    for bad in (3, 1, 0, -2):
        res = server.super_resolve(ing["image_id"], scale=bad)
        assert res["status"] == "error", f"scale={bad} 应报错: {res}"
    assert server.super_resolve(999999, scale=2)["status"] == "error"
    assert [h["capability"] for h in data.history(conn, ing["image_id"])] == ["base"]
