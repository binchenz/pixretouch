"""入库工具冒烟测试：ingest / ingest_album（工单 13 首步，提前落）。

纪律：走 MCP 工具函数入口（主接缝）；只断言外部行为——返回结构、库效果
（images/records 行、根记录+指针）、幂等去重、失败隔离清单；tmp_path 隔离，
无模型权重依赖（始终运行）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "ALBUM_INBOX", tmp_path / "inbox")
    monkeypatch.setattr(config, "MODELS_DIR", Path(__file__).resolve().parents[1] / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


def _make_jpg(p: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p, "JPEG")
    return p


def _make_tiff(p: Path) -> Path:
    Image.new("RGB", (32, 32), (10, 200, 90)).save(p, "TIFF")
    return p


def test_ingest_single_idempotent_and_validation(isolated_home, tmp_path):
    from pixretouch import data, engine, server

    conn, _ = engine.get_session()
    jpg = _make_jpg(tmp_path / "a.jpg", 1)

    r1 = server.ingest(str(jpg))
    assert r1["status"] == "ok" and r1["duplicate"] is False
    assert isinstance(r1["image_id"], int) and len(r1["sha256"]) == 64

    # 库效果：images 行 + 根记录 base + 当前态指针指向根
    cur = data.get_current(conn, r1["image_id"])
    assert cur["capability"] == "base" and cur["output_sha256"] == r1["sha256"]

    # 幂等：重复入库 → duplicate=True 同 image_id，不新增行
    r2 = server.ingest(str(jpg))
    assert r2["status"] == "ok" and r2["duplicate"] is True and r2["image_id"] == r1["image_id"]
    assert conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"] == 1

    # 原图只读
    assert jpg.exists()

    # 校验：不存在 / 非 JPEG-TIFF / 腐坏文件 → 结构化错误且不入库
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff garbage")
    for bad in (str(tmp_path / "missing.jpg"), str(tmp_path / "notes.txt"), str(tmp_path / "broken.jpg")):
        res = server.ingest(bad)
        assert res["status"] == "error", bad
    assert conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"] == 1


def test_ingest_album_isolation_and_dedup(isolated_home, tmp_path):
    from pixretouch import engine, server

    conn, _ = engine.get_session()
    album = tmp_path / "inbox"
    (album / "sub").mkdir(parents=True)
    _make_jpg(album / "img1.jpg", 11)
    _make_tiff(album / "img2.tif")
    _make_jpg(album / "sub" / "img3.jpeg", 33)          # 递归收纳
    (album / "broken.jpg").write_bytes(b"\xff\xd8 garbage bytes")  # 腐坏 → failed
    (album / "readme.txt").write_text("skip me")        # 非图像 → skipped
    (album / ".hidden.jpg").write_bytes(b"x")           # 隐藏文件 → 忽略

    res = server.ingest_album(str(album))
    assert res["status"] == "ok"
    assert len(res["ingested"]) == 3, res["ingested"]
    assert res["total"] == 4                              # ingested + failed
    assert len(res["failed"]) == 1 and "broken" in res["failed"][0]["path"]
    assert len(res["skipped"]) == 1 and res["skipped"][0]["path"].endswith("readme.txt")
    assert all(it["duplicate"] is False for it in res["ingested"])
    # 确定性顺序：按路径排序
    paths = [it["path"] for it in res["ingested"]]
    assert paths == sorted(paths)

    # 库效果：3 张图各有根记录与指针
    assert conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"] == 3
    assert conn.execute("SELECT COUNT(*) c FROM records WHERE capability='base'").fetchone()["c"] == 3

    # 幂等：整本重扫 → 全部 duplicate=True，行数不变
    res2 = server.ingest_album(str(album))
    assert len(res2["ingested"]) == 3 and all(it["duplicate"] is True for it in res2["ingested"])
    assert conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"] == 3


def test_ingest_album_default_inbox_and_missing_dir(isolated_home, tmp_path):
    from pixretouch import config, server

    # 显式不存在目录 → 结构化错误
    res = server.ingest_album(str(tmp_path / "no-such-dir"))
    assert res["status"] == "error" and "不存在" in res["error"]

    # 默认目录 = config.ALBUM_INBOX：ensure_dirs 语义保证其存在 → 空相册 ok/total=0
    res2 = server.ingest_album()
    assert res2["status"] == "ok" and res2["directory"] == str(config.ALBUM_INBOX) and res2["total"] == 0
