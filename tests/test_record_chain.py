"""记录链核心测试（工单 02 验收项）。

测试纪律：走 data 模块公开函数（与 MCP 工具同一进程内边界），只断言外部行为——
库行、指针位置、产物存在性、缓存命中计数；不断言实现细节。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pixretouch import data


@pytest.fixture()
def env(tmp_path):
    db = tmp_path / "test.db"
    conn = data.connect(db)
    data.init_schema(conn)
    cache = data.ContentCache(conn, tmp_path / "cache")
    # 造一张 fixture「照片」：字节内容确定，属性断言不依赖图像库
    src = tmp_path / "photo.bin"
    src.write_bytes(b"ORIGINAL-IMAGE-BYTES" * 100)
    return conn, cache, src, tmp_path


def _stamp_render(tmp: Path, marker: bytes, calls: list):
    def render(input_path: Path, params: dict) -> Path:
        calls.append(1)
        out = tmp / f"render_{len(calls)}.bin"
        out.write_bytes(input_path.read_bytes() + marker + params.get("extra", "").encode())
        return out
    return render


def test_ingest_original_readonly_and_root_record(env):
    conn, cache, src, tmp = env
    before = src.read_bytes()
    r = data.ingest(conn, src)
    assert r["duplicate"] is False
    assert src.read_bytes() == before, "ingest 不得改动原图"
    cur = data.get_current(conn, r["image_id"])
    assert cur["capability"] == "base"
    assert cur["output_sha256"] == r["sha256"]
    # 重复 ingest 幂等
    r2 = data.ingest(conn, src)
    assert r2["duplicate"] is True and r2["image_id"] == r["image_id"]


def test_apply_append_pointer_and_rollback(env):
    conn, cache, src, tmp = env
    r = data.ingest(conn, src)
    img = r["image_id"]
    calls: list = []
    render = _stamp_render(tmp, b"|EDIT1", calls)

    a1 = data.apply(conn, cache, img, "stamp", {"extra": "A"}, render)
    a2 = data.apply(conn, cache, img, "stamp", {"extra": "B"}, render)
    assert a1["record_id"] != a2["record_id"]
    assert data.get_current(conn, img)["id"] == a2["record_id"]
    hist = data.history(conn, img)
    assert [h["capability"] for h in hist] == ["base", "stamp", "stamp"]

    # 回滚 = 移动指针，链不动
    data.rollback_to(conn, img, a1["record_id"])
    assert data.get_current(conn, img)["id"] == a1["record_id"]
    assert len(data.history(conn, img)) == 2          # 当前态历史截短
    all_recs = conn.execute("SELECT COUNT(*) c FROM records WHERE image_id=?", (img,)).fetchone()["c"]
    assert all_recs == 3, "记录链只追加，回滚不删除"

    # 回滚到根 = 原图状态
    root = hist[0]["id"]
    data.rollback_to(conn, img, root)
    assert data.get_current(conn, img)["output_sha256"] == r["sha256"]

    # 跨图回滚必须拒绝
    src2 = tmp / "other.bin"
    src2.write_bytes(b"OTHER" * 50)
    r_other = data.ingest(conn, src2)
    with pytest.raises(KeyError):
        data.rollback_to(conn, r_other["image_id"], a1["record_id"])


def test_cache_hit_zero_render(env):
    conn, cache, src, tmp = env
    r = data.ingest(conn, src)
    img = r["image_id"]
    calls: list = []
    render = _stamp_render(tmp, b"|X", calls)

    first = data.apply(conn, cache, img, "stamp", {"extra": "V"}, render)
    assert first["cache_hit"] is False and len(calls) == 1

    data.rollback_to(conn, img, r["record_id"])   # 回到同一输入状态
    second = data.apply(conn, cache, img, "stamp", {"extra": "V"}, render)
    assert second["cache_hit"] is True and len(calls) == 1, "同键必须命中缓存，零渲染"
    assert second["output_sha256"] == first["output_sha256"]

    # 参数不同 → 键不同 → 重新渲染
    data.rollback_to(conn, img, r["record_id"])
    third = data.apply(conn, cache, img, "stamp", {"extra": "W"}, render)
    assert third["cache_hit"] is False and len(calls) == 2


def test_cluster_addressing_recorded(env):
    conn, cache, src, tmp = env
    r = data.ingest(conn, src)
    conn.execute("INSERT INTO clusters(name, person_type, created_at) VALUES('新娘','female','2026-01-01')")
    conn.commit()
    calls: list = []
    out = data.apply(conn, cache, r["image_id"], "stamp", {"extra": "C"},
                     _stamp_render(tmp, b"|C", calls), cluster_id=1, mask_sha="m" * 64)
    row = conn.execute("SELECT cluster_id, mask_sha256 FROM records WHERE id=?",
                       (out["record_id"],)).fetchone()
    assert row["cluster_id"] == 1 and row["mask_sha256"] == "m" * 64


def test_crash_recovery_pointer_durable(env, tmp_path):
    conn, cache, src, tmp = env
    r = data.ingest(conn, src)
    calls: list = []
    a1 = data.apply(conn, cache, r["image_id"], "stamp", {"extra": "P"}, _stamp_render(tmp, b"|P", calls))
    db_path = tmp / "test.db"
    conn.close()
    conn2 = data.connect(db_path)          # 模拟重启后重开
    assert data.get_current(conn2, r["image_id"])["id"] == a1["record_id"]
    conn2.close()
