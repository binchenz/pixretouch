"""工具接缝冒烟测试：color_grade（工单 09 验收项）。

纪律（Spec · Testing Decisions + 铁律 8 + 工单03 确立的模式）：
- 走 MCP 工具函数入口（server.color_grade）——主接缝；
- 只断言外部行为：返回结构、尺寸一致、参数组方向性效果（白平衡 R↑B↓/曝光变亮/
  对比 std 增大）、identity LUT 往返保真（解析+三线性正确性的强断言）、bw-classic
  去色+渐变单调+可复现、素材规则（内容 sha 进键/路径不进键）、记录链+1、回滚恢复、
  缓存命中、RGBA 链态 alpha 直通；属性式断言，不绑定字节；
- 本单能力无模型权重（纯数学管线）——测试无需 skipif；RGBA 链态用例依赖 ISNet，
  单独 skipif。tmp_path 隔离库/缓存；绝不污染 var/。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ISNET_PT = REPO_ROOT / "models" / "isnet" / "isnet-general-use.onnx"

requires_isnet = pytest.mark.skipif(not ISNET_PT.exists(), reason="ISNet 权重未预取")


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
    """fixture 图 256×256（确定性）：上半色相渐变带（HSV h 0..179），
    左下低对比灰块（100..150），右下全量程灰渐变（0..255）。"""
    import cv2

    p = tmp_path / "color_photo.png"
    img = np.zeros((256, 256, 3), np.uint8)
    hsv = np.zeros((128, 256, 3), np.uint8)
    hsv[..., 0] = np.linspace(0, 179, 256, dtype=np.uint8)[None, :]
    hsv[..., 1] = 200
    hsv[..., 2] = 230
    img[:128] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img[128:, :128] = np.linspace(100, 150, 128, dtype=np.uint8)[None, :, None]
    img[128:, 128:] = np.linspace(0, 255, 128, dtype=np.uint8)[None, :, None]
    Image.fromarray(img).save(p, "PNG")
    return p


def _ingest(photo: Path):
    from pixretouch import data, engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, photo)


def _px(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def _write_cube(p: Path, n: int, gamma: float = 1.0) -> Path:
    """生成确定性 .cube（gamma=1 → identity；行序 red 最快，与解析约定一致）。"""
    axis = np.linspace(0.0, 1.0, n) ** gamma
    with open(p, "w", encoding="ascii") as f:
        f.write(f'TITLE "test"\nLUT_3D_SIZE {n}\n')
        for bb in range(n):
            for gg in range(n):
                for rr in range(n):
                    f.write(f"{axis[rr]:.6f} {axis[gg]:.6f} {axis[bb]:.6f}\n")
    return p


# ---------------------------------------------------------------- LUT 解析与应用

def test_identity_lut_roundtrip(isolated_home, photo, tmp_path):
    """identity cube 往返 ≈ 原图：.cube 解析 + 三线性插值正确性的强断言（含色相渐变带）。"""
    from pixretouch import server

    conn, ing = _ingest(photo)
    cube = _write_cube(tmp_path / "identity.cube", 17)
    res = server.color_grade(ing["image_id"], lut_path=str(cube))
    assert res["status"] == "ok", res
    src, out = _px(photo), _px(res["output_path"])
    assert src.shape == out.shape, "输出尺寸一致"
    diff = np.abs(src - out)
    assert diff.mean() < 3, f"identity LUT 应近似恒等, mean diff {diff.mean():.2f}"
    assert diff.max() < 12, f"identity LUT 不应有离群偏差, max diff {diff.max()}"


def test_bw_classic_achromatic_monotonic_reproducible(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    res = server.color_grade(img_id, lut="bw-classic")
    assert res["status"] == "ok", res
    src, out = _px(photo), _px(res["output_path"])
    assert src.shape == out.shape

    # ① 去色：逐像素通道极差 ≈ 0（灰度输出）
    spread = (out.max(axis=2) - out.min(axis=2))
    assert spread.mean() < 6, f"bw-classic 应输出灰度, 平均通道极差 {spread.mean():.2f}"

    # ② 渐变单调性：右下全量程灰渐变经单调 LUT 曲线后仍单调（容 JPEG 噪声）
    ramp = out[200, 128:, 0]
    assert (np.diff(ramp) >= -2).all(), "灰渐变经 bw-classic 后亮度应单调不减"

    # ③ 可复现基准：回滚后同参重调 → 命中缓存 + 产物字节一致
    data.rollback_to(conn, img_id, ing["record_id"])
    again = server.color_grade(img_id, lut="bw-classic")
    assert again["cache_hit"] is True and again["output_sha256"] == res["output_sha256"]
    assert again["record_id"] != res["record_id"], "缓存命中仍追加新记录（记录链只追加）"


def test_lut_material_rule_content_sha_in_key(isolated_home, photo, tmp_path):
    """素材规则：内容 sha 进键、路径不进键；内容变 ⇒ 键变必 miss。"""
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    cube_a = _write_cube(tmp_path / "a.cube", 9)
    r1 = server.color_grade(img_id, lut_path=str(cube_a))
    assert r1["status"] == "ok" and r1["cache_hit"] is False

    # 同内容复制到路径 B → 键不变 → 必命中
    data.rollback_to(conn, img_id, ing["record_id"])
    cube_b = tmp_path / "b.cube"
    cube_b.write_bytes(cube_a.read_bytes())
    r2 = server.color_grade(img_id, lut_path=str(cube_b))
    assert r2["cache_hit"] is True and r2["output_sha256"] == r1["output_sha256"]

    # 路径 A 换内容（gamma 2.0 压暗曲线）→ 键变 → 必 miss 且效果不同
    data.rollback_to(conn, img_id, ing["record_id"])
    _write_cube(cube_a, 9, gamma=2.0)
    r3 = server.color_grade(img_id, lut_path=str(cube_a))
    assert r3["cache_hit"] is False and r3["output_sha256"] != r1["output_sha256"]
    assert _px(r3["output_path"]).mean() < _px(r1["output_path"]).mean(), "gamma2 应压暗"


# ---------------------------------------------------------------- 白平衡 / 影调参数组

def test_wb_tone_groups_independent_and_combined(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    src = _px(photo)

    def grade_rollback(**kw):
        data.rollback_to(conn, img_id, ing["record_id"])   # 每组参数都从同一链态出发（组独立性）
        r = server.color_grade(img_id, **kw)
        assert r["status"] == "ok", (kw, r)
        return r, _px(r["output_path"])

    # 白平衡组：+暖 → R 均值升、B 均值降
    _, out = grade_rollback(temperature=60)
    m0, m1 = src.mean(axis=(0, 1)), out.mean(axis=(0, 1))
    assert m1[0] > m0[0] + 2 and m1[2] < m0[2] - 2, f"色温+60 应暖移: {m0} -> {m1}"

    # 影调组：曝光 +1EV → 整体显著变亮
    _, out = grade_rollback(exposure=1.0)
    assert out.mean() > src.mean() + 15, "曝光+1EV 应变亮"

    # 影调组：对比 +60 → 低对比灰块 std 显著增大（软 S 曲线拉开中间调）
    _, out = grade_rollback(contrast=60)
    blk_s = src[128:, :128].std()
    blk_o = out[128:, :128].std()
    assert blk_o > blk_s * 1.3, f"对比+60 应拉开: std {blk_s:.1f} -> {blk_o:.1f}"

    # 三组组合为一次调用（验收项）
    r, out = grade_rollback(temperature=30, exposure=0.5, contrast=30, lut="bw-classic")
    assert r["status"] == "ok" and out.shape == src.shape

    # 每次调用记录链只追加：base + 4 次 grade（回滚不删记录）
    total = conn.execute("SELECT COUNT(*) c FROM records WHERE image_id=?", (img_id,)).fetchone()["c"]
    assert total == 5
    assert conn.execute(
        "SELECT COUNT(DISTINCT capability) c FROM records WHERE image_id=? AND capability='color_grade'",
        (img_id,)).fetchone()["c"] == 1


def test_no_adjustment_and_validation_errors(isolated_home, photo, tmp_path):
    """无调整项/越界/坏 cube → 结构化错误且记录链不增长（不依赖权重）。"""
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]
    bad = tmp_path / "bad.cube"
    bad.write_text("not a cube at all\n")

    cases = [
        {},                                                        # 无任何调整
        {"temperature": 200},                                      # 越界
        {"exposure": 5.0},                                         # 越界
        {"contrast": -500},                                        # 越界
        {"lut": "no-such-lut"},                                    # 未知内置名
        {"lut": "bw-classic", "lut_path": str(bad)},               # 二选一冲突
        {"lut_path": str(tmp_path / "missing.cube")},              # 不存在
        {"lut_path": str(bad)},                                    # 格式非法（fail-fast）
    ]
    for kw in cases:
        res = server.color_grade(img_id, **kw)
        assert res["status"] == "error", f"{kw} 应结构化报错: {res}"
    assert server.color_grade(999999, temperature=10)["status"] == "error"  # 图像不存在

    assert [h["capability"] for h in data.history(conn, img_id)] == ["base"]


def test_bw_classic_asset_matches_generator(tmp_path):
    """再生一致性（lead 批复 3）：生成器输出与已提交资产逐字节比对。

    防止将来改了配方常数/生成逻辑却忘记再生 luts/bw-classic.cube 资产。
    """
    from pixretouch.capabilities.color import (
        BW_CLASSIC_NAME,
        LUT_DIR,
        generate_bw_classic_cube,
    )

    committed = (LUT_DIR / f"{BW_CLASSIC_NAME}.cube").read_bytes()
    regen = tmp_path / "regen.cube"
    generate_bw_classic_cube(regen)
    assert regen.read_bytes() == committed, \
        "bw-classic.cube 资产与生成器不一致——请运行 generate_bw_classic_cube 再生并提交"


def test_original_readonly_and_rollback(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    before = photo.read_bytes()
    res = server.color_grade(ing["image_id"], temperature=-40)
    assert res["status"] == "ok"
    assert photo.read_bytes() == before, "原图只读：调色不得触碰原文件"
    data.rollback_to(conn, ing["image_id"], ing["record_id"])
    assert data.get_current(conn, ing["image_id"])["output_sha256"] == ing["sha256"], \
        "回滚=移动当前态指针，生效状态复原"


@requires_isnet
def test_rgba_chain_alpha_passthrough(isolated_home, photo):
    """RGBA 链态策略（纪律 8）：cutout → color_grade 只调 RGB，alpha 字节级直通。"""
    from pixretouch import server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    c = server.cutout(img_id)
    assert c["status"] == "ok", c
    rgba_before = Image.open(c["output_path"])
    assert rgba_before.mode == "RGBA"
    alpha_before = np.asarray(rgba_before.getchannel("A"))
    rgb_before = np.asarray(rgba_before.convert("RGB")).astype(int)

    g = server.color_grade(img_id, temperature=60)
    assert g["status"] == "ok", g
    out_img = Image.open(g["output_path"])
    assert out_img.mode == "RGBA", "RGBA 链态产物必须保持 RGBA（不得隐式平整）"
    assert np.array_equal(np.asarray(out_img.getchannel("A")), alpha_before), "alpha 必须字节级直通"
    rgb_after = np.asarray(out_img.convert("RGB")).astype(int)
    assert np.abs(rgb_after - rgb_before).mean() > 2, "RGB 通道应发生调色变化"
