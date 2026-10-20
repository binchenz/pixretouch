"""数据模型与记录链核心（工单 02 / issue #3）。

设计依据（ADR-0002）：
- 每张图一条追加式记录链：records.parent_id 指向前驱，永不修改、永不删除；
- images.current_record_id 即「当前态指针」，回滚 = 移动指针；
- 急切渲染 + 内容寻址缓存：apply() 先查缓存，未命中才调用能力渲染函数；
  缓存键 = (输入哈希, 能力, 规范化参数, mask 哈希?)。
- 原图只读：ingest 只登记哈希与路径，绝不写原文件。

本模块不依赖任何具体能力：渲染函数由调用方（能力注册表）注入，
使其成为「能力 → 记录链」模式的唯一挂接点。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  path TEXT NOT NULL,
  width INTEGER, height INTEGER, format TEXT,
  ingested_at TEXT NOT NULL,
  current_record_id INTEGER,
  meta TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS records (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id),
  parent_id INTEGER REFERENCES records(id),
  capability TEXT NOT NULL,
  params TEXT NOT NULL,
  output_sha256 TEXT,
  output_path TEXT,
  cluster_id INTEGER,
  mask_sha256 TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY,
  name TEXT,
  person_type TEXT NOT NULL DEFAULT 'unknown',
  default_suit_id INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id),
  cluster_id INTEGER REFERENCES clusters(id),
  bbox TEXT NOT NULL,
  landmark TEXT,
  quality REAL,
  embedding BLOB,
  detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preset_suits (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  recipe TEXT NOT NULL,
  is_builtin INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS cache_index (
  key TEXT PRIMARY KEY,
  output_sha256 TEXT NOT NULL,
  output_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    # PRAGMA 不支持绑定参数，且 journal_mode 须在脚本事务外执行
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_DDL)
    conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
    conn.commit()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- 缓存

class ContentCache:
    """内容寻址缓存：键→(输出哈希, 存储路径)；字节按哈希扇区存放。"""

    def __init__(self, conn: sqlite3.Connection, root: Path):
        self.conn = conn
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(input_sha: str, capability: str, params: dict, mask_sha: str | None = None) -> str:
        canon = json.dumps(
            {"in": input_sha, "cap": capability, "p": params, "mask": mask_sha},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(canon.encode()).hexdigest()

    def path_for(self, sha: str) -> Path:
        return self.root / sha[:2] / sha[2:4] / sha

    def get(self, key: str) -> Path | None:
        row = self.conn.execute(
            "SELECT output_path FROM cache_index WHERE key=?", (key,)
        ).fetchone()
        if row and Path(row["output_path"]).exists():
            return Path(row["output_path"])
        return None

    def put(self, key: str, src: Path) -> tuple[str, Path]:
        sha = file_sha256(src)
        dest = self.path_for(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        self.conn.execute(
            "INSERT OR IGNORE INTO cache_index(key, output_sha256, output_path, created_at) VALUES(?,?,?,?)",
            (key, sha, str(dest), _now()),
        )
        self.conn.commit()
        return sha, dest


# ---------------------------------------------------------------- 记录链

def ingest(conn: sqlite3.Connection, image_path: Path) -> dict[str, Any]:
    """原图只读入库：登记哈希与元数据，建根记录（capability='base'），指针指向根。"""
    image_path = Path(image_path)
    sha = file_sha256(image_path)
    existing = conn.execute("SELECT id FROM images WHERE sha256=?", (sha,)).fetchone()
    if existing:
        return {"image_id": existing["id"], "sha256": sha, "duplicate": True}

    w = h = None
    fmt = image_path.suffix.lstrip(".").lower()
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
            fmt = (im.format or fmt).lower()
    except Exception:  # noqa: BLE001
        pass

    cur = conn.execute(
        "INSERT INTO images(sha256, path, width, height, format, ingested_at) VALUES(?,?,?,?,?,?)",
        (sha, str(image_path), w, h, fmt, _now()),
    )
    image_id = cur.lastrowid
    rec = conn.execute(
        "INSERT INTO records(image_id, parent_id, capability, params, output_sha256, output_path, created_at)"
        " VALUES(?,NULL,'base','{}',?,?,?)",
        (image_id, sha, str(image_path), _now()),
    )
    conn.execute("UPDATE images SET current_record_id=? WHERE id=?", (rec.lastrowid, image_id))
    conn.commit()
    return {"image_id": image_id, "sha256": sha, "record_id": rec.lastrowid, "duplicate": False}


def get_current(conn: sqlite3.Connection, image_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT r.id, r.capability, r.params, r.output_sha256, r.output_path, r.cluster_id"
        " FROM images i JOIN records r ON r.id = i.current_record_id WHERE i.id=?",
        (image_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"image {image_id} has no current record")
    return dict(row)


def history(conn: sqlite3.Connection, image_id: int) -> list[dict[str, Any]]:
    """从当前态沿 parent 链回溯到根（返回顺序：根 → 当前）。"""
    rows = conn.execute(
        "SELECT id, parent_id, capability, params, output_sha256, cluster_id, created_at"
        " FROM records WHERE image_id=? ORDER BY id", (image_id,)
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    cur = conn.execute("SELECT current_record_id FROM images WHERE id=?", (image_id,)).fetchone()
    if cur is None or cur["current_record_id"] is None:
        return []
    chain, node = [], cur["current_record_id"]
    while node is not None:
        chain.append(by_id[node])
        node = by_id[node]["parent_id"]
    return list(reversed(chain))


def rollback_to(conn: sqlite3.Connection, image_id: int, record_id: int) -> dict[str, Any]:
    """回滚 = 移动当前态指针（记录链不动）。目标必须属于同一图像。"""
    row = conn.execute(
        "SELECT id FROM records WHERE id=? AND image_id=?", (record_id, image_id)
    ).fetchone()
    if row is None:
        raise KeyError(f"record {record_id} does not belong to image {image_id}")
    conn.execute("UPDATE images SET current_record_id=? WHERE id=?", (record_id, image_id))
    conn.commit()
    return get_current(conn, image_id)


def apply(
    conn: sqlite3.Connection,
    cache: ContentCache,
    image_id: int,
    capability: str,
    params: dict,
    render: Callable[[Path, dict], Path],
    cluster_id: int | None = None,
    mask_sha: str | None = None,
) -> dict[str, Any]:
    """「能力 → 记录链」标准挂接点（工单 03 起全部能力沿用）。

    先查缓存（键含当前输出哈希 → 链上位置敏感），未命中调用 render 急切渲染，
    产物入内容寻址缓存，随后追加记录并移动指针。
    render(input_path, params) -> 临时产物路径；本函数负责将其固化进缓存。
    """
    cur = get_current(conn, image_id)
    key = cache.key(cur["output_sha256"], capability, params, mask_sha)
    hit = cache.get(key)
    if hit is not None:
        out_sha, out_path, cache_hit = file_sha256(hit), hit, True
    else:
        rendered = Path(render(Path(cur["output_path"]), params))
        out_sha, out_path = cache.put(key, rendered)
        cache_hit = False

    rec = conn.execute(
        "INSERT INTO records(image_id, parent_id, capability, params, output_sha256, output_path,"
        " cluster_id, mask_sha256, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (image_id, cur["id"], capability,
         json.dumps(params, sort_keys=True, ensure_ascii=False),
         out_sha, str(out_path), cluster_id, mask_sha, _now()),
    )
    conn.execute("UPDATE images SET current_record_id=? WHERE id=?", (rec.lastrowid, image_id))
    conn.commit()
    return {
        "record_id": rec.lastrowid,
        "output_sha256": out_sha,
        "output_path": str(out_path),
        "cache_hit": cache_hit,
    }
