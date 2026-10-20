"""工具接缝冒烟测试：cutout / bg_replace / bg_blur（工单 08 验收项）。

纪律（Spec · Testing Decisions + 铁律 8 + 工单03 确立的模式）：
- 走 MCP 工具函数入口（server.cutout / server.bg_replace / server.bg_blur）——主接缝；
- 只断言外部行为：返回结构、尺寸一致、目标区域变化（自参照 mask/t 图，不绑定模型
  语义字节）、记录链+1、回滚恢复、缓存命中（先 rollback 再同参——链位置敏感）、
  mask 通道双登记（cache_index 有 "mask:<sha>" 行、mask_path/alpha_path 可独立打开）；
- tmp_path 隔离库/缓存，MODELS_DIR 指仓库真实 models/（权重跨测试共享）；
  权重缺失 skipif；绝不污染 var/。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
ISNET_PT = REPO_ROOT / "models" / "isnet" / "isnet-general-use.onnx"
DA_PT = REPO_ROOT / "models" / "depth_anything_v2_small" / "onnx" / "model.onnx"

requires_isnet = pytest.mark.skipif(not ISNET_PT.exists(), reason="ISNet 权重未预取")
requires_depth = pytest.mark.skipif(not DA_PT.exists(), reason="Depth-Anything 权重未预取")


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """库/缓存/临时目录指向 tmp_path；权重共享仓库 models/；引擎会话前后清场。"""
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "MODELS_DIR", REPO_ROOT / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


@pytest.fixture()
def photo(tmp_path) -> Path:
    """fixture 图：320×240 渐变+确定性噪声背景（虚化可测）+ 中心浅色椭圆「主体」。"""
    p = tmp_path / "photo.png"
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:240, 0:320]
    base = (xx * 0.5 + yy * 0.5).astype(np.float32)
    noise = rng.normal(0, 12, size=(240, 320)).astype(np.float32)
    gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.merge("RGB", (Image.fromarray(gray),) * 3)
    ImageDraw.Draw(img).ellipse([110, 70, 210, 170], fill=(250, 245, 240))
    img.save(p, "PNG")  # 无损：fixture 字节确定
    return p


def _ingest(photo: Path):
    from pixretouch import data, engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, photo)


def _px(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


# ---------------------------------------------------------------- cutout

@requires_isnet
def test_cutout_dual_channel_and_chain(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    res = server.cutout(img_id)
    assert res["status"] == "ok", res
    assert len(res["output_sha256"]) == 64 and res["cache_hit"] is False

    # ① output 通道：记录产物 = RGBA PNG，尺寸一致，链保持「图像状态」语义
    out = Image.open(res["output_path"])
    assert out.mode == "RGBA" and out.size == (320, 240)

    # ② mask 通道：alpha 独立产物入库（可打开、L 模式、同尺寸、哈希一致、cache_index 双登记）
    assert res["alpha_sha256"] == res["mask_sha256"] and len(res["alpha_sha256"]) == 64
    alpha = Image.open(res["alpha_path"])
    assert alpha.mode == "L" and alpha.size == (320, 240)
    row = conn.execute(
        "SELECT output_sha256 FROM cache_index WHERE key=?", (f"mask:{res['alpha_sha256']}",)
    ).fetchone()
    assert row is not None and row["output_sha256"] == res["alpha_sha256"]

    # alpha 非退化且主体可分（属性式：中心显著亮于角落；CPU EP 确定性输出）
    a = np.asarray(alpha).astype(int)
    assert a.std() > 10
    assert a[120, 160] > a[5, 5] + 50

    # ③ 记录链+1、指针、mask_sha256 落库
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "cutout"]
    assert data.get_current(conn, img_id)["id"] == res["record_id"]
    rec = conn.execute("SELECT mask_sha256 FROM records WHERE id=?", (res["record_id"],)).fetchone()
    assert rec["mask_sha256"] == res["mask_sha256"]

    # ④ 回滚恢复 → 原图哈希；再同参 apply → 命中缓存（先回滚，链位置敏感）
    data.rollback_to(conn, img_id, ing["record_id"])
    assert data.get_current(conn, img_id)["output_sha256"] == ing["sha256"]
    again = server.cutout(img_id)
    assert again["cache_hit"] is True and again["output_sha256"] == res["output_sha256"]
    assert again["record_id"] != res["record_id"], "缓存命中仍追加新记录（记录链只追加）"

    # ⑤ 链组合：RGBA 当前态上再 cutout → 复用内嵌 alpha，零重分割（仍 ok 且同产物）
    third = server.cutout(img_id)
    assert third["status"] == "ok" and third["alpha_sha256"] == res["alpha_sha256"]


# ---------------------------------------------------------------- bg_replace

@requires_isnet
def test_bg_replace_color_feather_and_cache(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    target = [30, 120, 200]

    res = server.bg_replace(img_id, color=target, feather=2.0, despill=True)
    assert res["status"] == "ok", res
    out = _px(Path(res["output_path"]))
    src = _px(photo)
    assert out.shape == src.shape, "输出尺寸一致"

    # 自参照属性断言：从 mask 通道产物（羽化后 alpha）取实际分区，不预设主体位置——
    # alpha≈0 区 ≈ 背景色；alpha≈1 区 ≈ 原前景色（合成语义）；角落像素兜底检查
    m = np.asarray(Image.open(res["mask_path"])).astype(np.float32) / 255.0
    fg_zone = m > 0.98
    bg_zone = m < 0.02
    assert fg_zone.any() and bg_zone.any(), "羽化 alpha 应有饱和前景区与背景区"
    assert np.abs(out[5, 5] - np.array(target)).max() <= 8, f"角落应为背景色: {out[5, 5]}"
    assert np.abs(out[bg_zone] - np.array(target)).mean() < 4, "alpha≈0 区应为目标背景色"
    assert np.abs(out[fg_zone] - src[fg_zone]).mean() < 6, "alpha≈1 区应保留原前景色"
    assert np.abs(out[fg_zone] - src[fg_zone]).max() <= 30, "前景区不应有离群偏差"

    # 边缘无硬切：羽化后 alpha（mask 通道产物）存在过渡带
    mi = (m * 255).round().astype(int)
    transition = int(((mi > 5) & (mi < 250)).sum())
    assert transition > 50, f"羽化应产生过渡带, 实际过渡像素 {transition}"

    # 记录链+1；回滚后同参重调命中缓存
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "bg_replace"]
    data.rollback_to(conn, img_id, ing["record_id"])
    again = server.bg_replace(img_id, color=target, feather=2.0, despill=True)
    assert again["cache_hit"] is True and again["output_sha256"] == res["output_sha256"]


@requires_isnet
def test_bg_replace_image_bg_content_sha_in_key(isolated_home, photo, tmp_path):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    bg = tmp_path / "bg.png"

    # 背景图 A：纯红
    Image.new("RGB", (64, 64), (220, 30, 30)).save(bg, "PNG")
    r1 = server.bg_replace(img_id, background_path=str(bg))
    assert r1["status"] == "ok", r1
    assert np.abs(_px(Path(r1["output_path"]))[5, 5] - np.array([220, 30, 30])).max() <= 8

    # 同内容异路径 → 路径不进键（素材规则，lead 裁决）→ 必命中
    data.rollback_to(conn, img_id, ing["record_id"])
    bg2 = tmp_path / "bg-copy.png"
    bg2.write_bytes(bg.read_bytes())
    r_same = server.bg_replace(img_id, background_path=str(bg2))
    assert r_same["cache_hit"] is True and r_same["output_sha256"] == r1["output_sha256"]

    # 同路径换内容（纯绿）→ bg_sha256 变 → 缓存键变 → 必 miss（内容 sha 进键的证据）
    data.rollback_to(conn, img_id, ing["record_id"])
    Image.new("RGB", (64, 64), (30, 200, 60)).save(bg, "PNG")
    r2 = server.bg_replace(img_id, background_path=str(bg))
    assert r2["status"] == "ok" and r2["cache_hit"] is False
    assert r2["output_sha256"] != r1["output_sha256"]
    assert np.abs(_px(Path(r2["output_path"]))[5, 5] - np.array([30, 200, 60])).max() <= 8

    # cover 适配：小背景图被放大盖满 320×240（角落仍是背景色即证据）
    assert np.abs(_px(Path(r2["output_path"]))[235, 315] - np.array([30, 200, 60])).max() <= 8


@requires_isnet
def test_chain_composition_cutout_then_bg_replace(isolated_home, photo):
    """链组合：cutout 的 RGBA 当前态 → bg_replace 直接消费内嵌 alpha。"""
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    c = server.cutout(img_id)
    assert c["status"] == "ok"
    b = server.bg_replace(img_id, color=[10, 10, 10])
    assert b["status"] == "ok"
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "cutout", "bg_replace"]
    out = _px(Path(b["output_path"]))
    assert np.abs(out[5, 5] - np.array([10, 10, 10])).max() <= 8
    # RGBA 态上 bg_replace 产物回到 RGB JPEG（背景不再透明）
    assert Image.open(b["output_path"]).mode == "RGB"


# ---------------------------------------------------------------- bg_blur

@requires_depth
def test_bg_blur_subject_sharp_strength_adjustable(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    src = _px(photo)

    strong = server.bg_blur(img_id, strength=0.9, max_blur=31, focus=0.9)
    assert strong["status"] == "ok", strong
    out_s = _px(Path(strong["output_path"]))
    assert out_s.shape == src.shape, "输出尺寸一致"

    # 自参照属性断言：t 图（mask 通道产物）t==0 处主体清晰度不变，t 高处发生变化
    t = np.asarray(Image.open(strong["mask_path"])).astype(np.float32) / 255.0
    assert t.shape == (240, 320)
    diff = np.abs(out_s - src).mean(axis=2)
    zero = t == 0
    high = t > 0.5
    assert zero.any() and high.any(), "focus 保护与虚化区都应存在"
    assert diff[zero].mean() < 3, f"t=0 区域必须保持清晰, 实际 {diff[zero].mean():.2f}"
    assert diff[high].mean() > diff[zero].mean() + 1, "t 高处应发生可测变化（噪声背景被平滑）"

    # 记录链+1 与 mask 落库
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "bg_blur"]
    rec = conn.execute("SELECT mask_sha256 FROM records WHERE id=?", (strong["record_id"],)).fetchone()
    assert rec["mask_sha256"] == strong["mask_sha256"]

    # 强度可调：回滚后小强度重跑 → t 均值更小、整体变化更小（单调性）
    data.rollback_to(conn, img_id, ing["record_id"])
    weak = server.bg_blur(img_id, strength=0.3, max_blur=31, focus=0.9)
    assert weak["status"] == "ok" and weak["cache_hit"] is False
    t_w = np.asarray(Image.open(weak["mask_path"])).astype(np.float32) / 255.0
    assert t_w.mean() < t.mean(), "strength 越大权重图越强"
    out_w = _px(Path(weak["output_path"]))
    assert np.abs(out_w - src).mean() <= np.abs(out_s - src).mean() + 0.01, "强度单调"

    # 回滚后同参重调 → 命中缓存（链位置敏感语义）
    data.rollback_to(conn, img_id, ing["record_id"])
    hit = server.bg_blur(img_id, strength=0.9, max_blur=31, focus=0.9)
    assert hit["cache_hit"] is True and hit["output_sha256"] == strong["output_sha256"]


# ---------------------------------------------------------------- 校验（无需权重）

def test_validation_errors_structured(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    cases = [
        (server.bg_replace, {"image_id": img_id}),                                   # 双空
        (server.bg_replace, {"image_id": img_id, "color": [0, 0, 0],
                             "background_path": str(photo)}),                          # 双给
        (server.bg_replace, {"image_id": img_id, "color": [300, 0]}),                 # 色值非法
        (server.bg_replace, {"image_id": img_id, "background_path": "/no/such.png"}), # 背景不存在
        (server.bg_blur, {"image_id": img_id, "strength": 0.0}),                      # 强度越界
        (server.bg_blur, {"image_id": img_id, "strength": 1.5}),
        (server.bg_blur, {"image_id": img_id, "max_blur": 1}),                        # 核越界
        (server.bg_blur, {"image_id": img_id, "focus": 0.0}),                         # focus 越界
        (server.cutout, {"image_id": 999999}),                                        # 图像不存在
    ]
    for fn, kwargs in cases:
        res = fn(**kwargs)
        assert res["status"] == "error", f"{fn.__name__} {kwargs} 应结构化报错: {res}"
        assert "error" in res

    # 任何失败调用都不得追加记录
    assert [h["capability"] for h in data.history(conn, img_id)] == ["base"]
