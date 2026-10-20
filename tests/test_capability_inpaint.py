"""工具接缝冒烟测试：blemish_remove（工单 03 验收项）。

纪律（Spec · Testing Decisions + 铁律 8）：
- 走 MCP 工具函数入口（server.blemish_remove）—— 主接缝；
- 只断言外部行为：返回结构、输出尺寸一致、目标区域变化、蒙版外近似不变、
  记录链+1、回滚恢复、缓存命中；属性式断言，不绑定模型输出字节；
- 库/缓存/临时目录用 tmp_path 隔离（PIXRETOUCH_HOME 由 conftest 全局兜底），
  MODELS_DIR 指向仓库 models/（真实权重跨测试共享）；权重缺失 skipif；绝不污染 var/。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
LAMA_DIR = REPO_ROOT / "models" / "lama"

requires_lama = pytest.mark.skipif(
    not (LAMA_DIR / "big-lama.pt").exists(),
    reason="LaMa 权重未预取（models/lama/big-lama.pt）",
)


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """库/缓存/临时目录指向 tmp_path；权重共享仓库 models/；引擎会话前后清场。"""
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    # 权重共享仓库真实 models/（weight_path() 会再拼 WEIGHT_NAME 一层）
    monkeypatch.setattr(config, "MODELS_DIR", REPO_ROOT / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


@pytest.fixture()
def photo(tmp_path) -> Path:
    """fixture 图：256×256 灰渐变 + (102..121,102..121) 20×20 亮白方块「污点」。"""
    p = tmp_path / "photo.jpg"
    yy, xx = np.mgrid[0:256, 0:256]
    gray = ((xx + yy) / 2).astype(np.uint8)
    img = Image.merge("RGB", (Image.fromarray(gray),) * 3)
    ImageDraw.Draw(img).rectangle([102, 102, 121, 121], fill=(255, 255, 255))
    img.save(p, "JPEG", quality=95, subsampling=0)
    return p


def _ingest(photo: Path):
    from pixretouch import data, engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, photo)


@requires_lama
def test_blemish_remove_points_tool_seam(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    res = server.blemish_remove(img_id, points=[[112, 112]], brush=12)

    # ① 工具契约：状态 + 记录ID + 产物哈希 + 产物存在
    assert res["status"] == "ok", res
    assert isinstance(res["record_id"], int) and res["record_id"] > 0
    assert len(res["output_sha256"]) == 64
    assert res["cache_hit"] is False
    out_path = Path(res["output_path"])
    assert out_path.exists()

    src = np.asarray(Image.open(photo).convert("RGB")).astype(int)
    out = np.asarray(Image.open(out_path).convert("RGB")).astype(int)

    # ② 输出尺寸一致
    assert src.shape == out.shape

    diff = np.abs(src - out)
    # ③ 目标区域发生变化（亮白方块被去除 → 亮度显著下降）；属性式断言
    core = diff[104:120, 104:120]
    assert core.mean() > 20, f"污点区域应显著变化, 实际 {core.mean():.1f}"
    assert out[104:120, 104:120].mean() < src[104:120, 104:120].mean() - 20
    # 蒙版外区域近似不变（合成只改 mask 内；差异仅 JPEG 重编码噪声级）
    far = diff[0:80, 160:256]
    assert far.mean() < 6, f"蒙版外区域应近似不变, 实际 {far.mean():.1f}"

    # ④ 记录链+1、指针指向新记录、mask 哈希落库
    hist = data.history(conn, img_id)
    assert [h["capability"] for h in hist] == ["base", "blemish_remove"]
    assert data.get_current(conn, img_id)["id"] == res["record_id"]
    row = conn.execute(
        "SELECT mask_sha256 FROM records WHERE id=?", (res["record_id"],)
    ).fetchone()
    assert row["mask_sha256"] == res["mask_sha256"] and len(row["mask_sha256"]) == 64

    # ⑤ 回滚恢复：指针回根、生效状态=原图哈希；记录链只追加不缩水
    data.rollback_to(conn, img_id, hist[0]["id"])
    assert data.get_current(conn, img_id)["output_sha256"] == ing["sha256"]
    total = conn.execute(
        "SELECT COUNT(*) c FROM records WHERE image_id=?", (img_id,)
    ).fetchone()["c"]
    assert total == 2


@requires_lama
def test_boxes_hint_and_cache_hit(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    first = server.blemish_remove(img_id, boxes=[[100, 100, 124, 124]])
    assert first["status"] == "ok" and first["cache_hit"] is False
    src = np.asarray(Image.open(photo).convert("RGB")).astype(int)
    out = np.asarray(Image.open(first["output_path"]).convert("RGB")).astype(int)
    assert np.abs(src - out)[104:120, 104:120].mean() > 20, "框提示同样应去除污点"

    # 回滚到根后同参重调 → 同键命中缓存、零渲染、产物哈希一致
    data.rollback_to(conn, img_id, ing["record_id"])
    again = server.blemish_remove(img_id, boxes=[[100, 100, 124, 124]])
    assert again["cache_hit"] is True
    assert again["output_sha256"] == first["output_sha256"]
    assert again["record_id"] != first["record_id"], "缓存命中仍应追加新记录（记录链只追加）"


@requires_lama
def test_brush_changes_mask_and_cache_key(isolated_home, photo):
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    a = server.blemish_remove(img_id, points=[[112, 112]], brush=6)
    data.rollback_to(conn, img_id, ing["record_id"])
    b = server.blemish_remove(img_id, points=[[112, 112]], brush=20)

    assert a["status"] == "ok" and b["status"] == "ok"
    assert a["mask_sha256"] != b["mask_sha256"], "brush 变化 → mask 变化"
    assert a["cache_hit"] is False and b["cache_hit"] is False, \
        "mask 哈希参与缓存键 → 不同键各自渲染"
    assert a["output_sha256"] != b["output_sha256"]


def test_validation_errors_structured(isolated_home, photo):
    """校验失败 → 结构化错误且记录链不增长（本例不依赖权重，不 skip）。"""
    from pixretouch import data, server

    conn, ing = _ingest(photo)
    img_id = ing["image_id"]

    for kwargs in (
        {},                                    # points 与 boxes 全空
        {"points": [[9999, 0]]},               # 点越界
        {"boxes": [[10, 10, 5, 5]]},           # 框非法（x0>=x1）
        {"points": [[112, 112]], "brush": 0},  # brush 超范围
        {"points": [[112]], "brush": 5},       # 点形状错误
    ):
        res = server.blemish_remove(img_id, **kwargs)
        assert res["status"] == "error", kwargs
        assert "error" in res

    # 不存在的图像 → 结构化错误而非抛穿 MCP
    assert server.blemish_remove(999999, points=[[1, 1]])["status"] == "error"

    # 任何失败调用都不得追加记录
    assert [h["capability"] for h in data.history(conn, img_id)] == ["base"]
