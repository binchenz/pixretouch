"""记录链查询/回退工具测试（task-10 补充面，ref issue #13）。

主接缝 = chainops op 函数（server 薄包装的实现体）；只断言外部行为：
返回结构、指针位置、记录链只追加不删除；无模型依赖。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pixretouch import chainops, data


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "MODELS_DIR", Path(__file__).resolve().parents[1] / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


def _chain(tmp_path):
    """ingest + 两条 stamp 记录（base → A → B），返回 (conn, image_id, rec_a, rec_b)。"""
    from pixretouch import engine

    conn, cache = engine.get_session()
    src = tmp_path / "photo.bin"
    src.write_bytes(b"CHAIN-TEST" * 100)
    ing = data.ingest(conn, src)
    img = ing["image_id"]

    def render(in_path: Path, params: dict) -> Path:
        out = tmp_path / f"r_{params['tag']}.bin"
        out.write_bytes(in_path.read_bytes() + params["tag"].encode())
        return out

    a = data.apply(conn, cache, img, "stamp", {"tag": "A"}, render)
    b = data.apply(conn, cache, img, "stamp", {"tag": "B"}, render)
    return conn, img, a["record_id"], b["record_id"]


def test_history_and_rollback_roundtrip(isolated_home):
    conn, img, rec_a, rec_b = _chain(isolated_home)

    h = chainops.history_op(img)
    assert h["status"] == "ok"
    assert [r["capability"] for r in h["records"]] == ["base", "stamp", "stamp"]
    assert h["current_record_id"] == rec_b

    r = chainops.rollback_to_op(img, rec_a)
    assert r["status"] == "ok" and r["current"]["id"] == rec_a
    h2 = chainops.history_op(img)
    assert h2["current_record_id"] == rec_a
    assert len(h2["records"]) == 2, "回退后当前态历史截短"
    total = conn.execute("SELECT COUNT(*) c FROM records WHERE image_id=?", (img,)).fetchone()["c"]
    assert total == 3, "记录链只追加：回退不删除记录"


def test_chainops_structured_errors(isolated_home):
    conn, img, rec_a, _ = _chain(isolated_home)

    r = chainops.history_op(999999)
    assert r["status"] == "error" and "KeyError" in r["error"]
    r = chainops.rollback_to_op(999999, rec_a)
    assert r["status"] == "error" and "KeyError" in r["error"]

    # 跨图回退必须拒绝（记录不属于该图像）
    src2 = isolated_home / "other.bin"
    src2.write_bytes(b"OTHER" * 50)
    other = data.ingest(conn, src2)["image_id"]
    r = chainops.rollback_to_op(other, rec_a)
    assert r["status"] == "error" and "KeyError" in r["error"]
    # 失败不得移动任何指针
    assert data.get_current(conn, other)["capability"] == "base"
