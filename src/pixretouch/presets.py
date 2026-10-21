"""预设方案库（工单 11 / issue #12）：种子方案 + 版本化配方 + CRUD。

预设方案（GLOSSARY）：命名的、可复用、可版本化的能力参数组合，可整体应用到
一张图或一个人。存储于 preset_suits 表（name, version, recipe JSON,
is_builtin），UNIQUE(name, version)——同名多版本并存，apply 默认取最新版；
配方一经入库不改（版本化追加，可审计）。

配方格式（recipe_version=1）：
    {
      "recipe_version": 1,
      "steps": [
        {"capability": "<能力目录名>", "params": {<渲染级参数>},
         "scope": "person" | "image"}
      ]
    }
- scope=person：人像域能力（smooth_skin / whiten_teeth / brighten_eyes /
  face_enhance），apply 时按人展开（聚类路由见 capabilities/preset_engine）；
- scope=image：全图能力（color_grade 等），apply 时执行一次；
- params 是**渲染级规范形**（直接喂 @capability 渲染函数；缺省项由渲染层
  params.get 兜底）——save_preset 做结构校验，数值语义边界由各能力自守。
- straight 基准 = 零步骤配方：输出内容恒等于输入（工单 11 验收锚点，
  也是「方案引擎自身不引入画质损失」的对照证明）。

种子 5 套（Spec 实现决定：natural / editorial / bw-classic / rescue / straight）：
参数是经验起点，skill 决策树（工单 12）会给出选择规则；用户可 save_preset
自建（如「我的韩式哑光」，Spec 用户故事 21）。

MCP stdio 纪律：不 print；日志 logging→stderr。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence

from .engine import get_session

log = logging.getLogger("pixretouch.presets")

RECIPE_VERSION = 1
# person 域步骤可用的能力（引擎会为其构建人脸 mask 并按聚类展开）
PERSON_CAPABILITIES = ("smooth_skin", "whiten_teeth", "brighten_eyes", "face_enhance")

# ------------------------------------------------------------------ 种子方案
SEED_SUITS: dict[str, list[dict[str, Any]]] = {
    "natural": [
        {"capability": "smooth_skin", "params": {"strength": 0.4}, "scope": "person"},
        {"capability": "brighten_eyes", "params": {"amount": 0.25}, "scope": "person"},
        {"capability": "color_grade", "params": {"temperature": 4, "exposure": 0.1}, "scope": "image"},
    ],
    "editorial": [
        {"capability": "face_enhance", "params": {"strength": 0.25}, "scope": "person"},
        {"capability": "smooth_skin", "params": {"strength": 0.55}, "scope": "person"},
        {"capability": "whiten_teeth", "params": {"amount": 0.4}, "scope": "person"},
        {"capability": "color_grade", "params": {"contrast": 18, "temperature": -4}, "scope": "image"},
    ],
    "bw-classic": [
        {"capability": "smooth_skin", "params": {"strength": 0.4}, "scope": "person"},
        {"capability": "color_grade", "params": {"lut": "bw-classic"}, "scope": "image"},
    ],
    "rescue": [
        {"capability": "face_enhance", "params": {"strength": 0.5}, "scope": "person"},
        {"capability": "smooth_skin", "params": {"strength": 0.5}, "scope": "person"},
        {"capability": "brighten_eyes", "params": {"amount": 0.3}, "scope": "person"},
        {"capability": "color_grade", "params": {"wb_auto": True, "exposure": 0.5}, "scope": "image"},
    ],
    "straight": [],   # 对照基准：零步骤，输出恒等输入
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def recipe_json(steps: Sequence[dict[str, Any]]) -> str:
    return json.dumps({"recipe_version": RECIPE_VERSION, "steps": list(steps)},
                      ensure_ascii=False, sort_keys=True)


def validate_steps(steps: Any) -> list[dict[str, Any]]:
    """结构校验（save/seed 共用）：能力在册、scope 合法、person 域白名单、params 为 dict。"""
    from . import capabilities as _caps

    _caps.load_all()   # 全量注册表（幂等：import 有缓存）——避免部分导入误报「不在能力目录」
    CAPABILITIES = _caps.CAPABILITIES

    if not isinstance(steps, list):
        raise ValueError(f"steps 必须是列表，收到 {type(steps).__name__}")
    out = []
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            raise ValueError(f"steps[{i}] 必须是对象，收到 {type(st).__name__}")
        cap = st.get("capability")
        if cap not in CAPABILITIES:
            raise ValueError(f"steps[{i}]: 能力 {cap!r} 不在能力目录（{sorted(CAPABILITIES)}）")
        scope = st.get("scope", "image")
        if scope not in ("person", "image"):
            raise ValueError(f"steps[{i}]: scope 须为 person|image，收到 {scope!r}")
        if scope == "person" and cap not in PERSON_CAPABILITIES:
            raise ValueError(
                f"steps[{i}]: person 域仅支持 {PERSON_CAPABILITIES}，收到 {cap!r}"
                f"（全图能力请用 scope=image）")
        params = st.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"steps[{i}]: params 必须是对象")
        if cap == "color_grade":
            # LUT 参数 fail-fast（issue #16：不支持的形态在入库时响亮拒绝，不静默丢弃）
            lut, lut_path = params.get("lut"), params.get("lut_path")
            if lut is not None and lut_path is not None:
                raise ValueError(f"steps[{i}]: lut（内置名）与 lut_path（外部文件）二选一")
            if lut is not None and not isinstance(lut, str):
                raise ValueError(f"steps[{i}]: lut 须为内置 LUT 名字符串，收到 {lut!r}")
            if lut_path is not None and not isinstance(lut_path, str):
                raise ValueError(f"steps[{i}]: lut_path 须为路径字符串，收到 {lut_path!r}")
        out.append({"capability": cap, "params": params, "scope": scope})
    return out


def ensure_seeds(conn: sqlite3.Connection) -> int:
    """幂等入库 5 套种子方案（INSERT OR IGNORE by (name, version)）；返回新增数。"""
    n = 0
    for name, steps in SEED_SUITS.items():
        cur = conn.execute(
            "INSERT OR IGNORE INTO preset_suits(name, version, recipe, is_builtin, created_at)"
            " VALUES(?,?,?,?,?)",
            (name, 1, recipe_json(steps), 1, _now()),
        )
        n += cur.rowcount or 0
    if n:
        conn.commit()
        log.info("种子预设方案入库 %s 套", n)
    return n


def save_suit(conn: sqlite3.Connection, name: str, steps: Sequence[dict[str, Any]],
              version: int | None = None) -> dict[str, Any]:
    """自建/升版方案：version 缺省 = 同名最新版 +1（版本化追加，不改旧版）。"""
    name = str(name).strip()
    if not name:
        raise ValueError("方案名不能为空")
    steps = validate_steps(steps)
    if version is None:
        row = conn.execute("SELECT MAX(version) v FROM preset_suits WHERE name=?", (name,)).fetchone()
        version = int(row["v"] or 0) + 1
    version = int(version)
    if version < 1:
        raise ValueError(f"version 须 ≥1，收到 {version}")
    conn.execute(
        "INSERT INTO preset_suits(name, version, recipe, is_builtin, created_at) VALUES(?,?,?,?,?)",
        (name, version, recipe_json(steps), 0, _now()),
    )
    conn.commit()
    return {"id": conn.execute("SELECT last_insert_rowid()").fetchone()[0],
            "name": name, "version": version, "steps": len(steps)}


def list_suits(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, name, version, recipe, is_builtin, created_at FROM preset_suits"
        " ORDER BY name, version"
    ).fetchall()
    out = []
    for r in rows:
        recipe = json.loads(r["recipe"])
        out.append({
            "id": r["id"], "name": r["name"], "version": r["version"],
            "is_builtin": bool(r["is_builtin"]), "created_at": r["created_at"],
            "recipe_version": recipe.get("recipe_version"),
            "steps": recipe.get("steps", []),
        })
    return out


def resolve_suit(conn: sqlite3.Connection, ref: int | str) -> dict[str, Any]:
    """按 id 或名称（取最新版）解析方案行；找不到 → KeyError（结构化错误面）。"""
    if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().lstrip("-").isdigit()):
        row = conn.execute(
            "SELECT id, name, version, recipe, is_builtin FROM preset_suits WHERE id=?",
            (int(ref),)).fetchone()
        if row is None:
            raise KeyError(f"preset suit id={ref} not found")
    else:
        row = conn.execute(
            "SELECT id, name, version, recipe, is_builtin FROM preset_suits WHERE name=?"
            " ORDER BY version DESC LIMIT 1", (str(ref),)).fetchone()
        if row is None:
            raise KeyError(f"preset suit {ref!r} not found（list_presets 查看可用方案）")
    d = dict(row)
    d["steps"] = json.loads(d["recipe"]).get("steps", [])
    return d


def bind_cluster_suit(conn: sqlite3.Connection, cluster_id: int, suit_id: int | None) -> dict[str, Any]:
    """把人物聚类绑定到默认预设方案（None=解绑）；批量应用时按人路由的依据。"""
    if conn.execute("SELECT id FROM clusters WHERE id=?", (cluster_id,)).fetchone() is None:
        raise KeyError(f"cluster {cluster_id} not found")
    if suit_id is not None:
        if conn.execute("SELECT id FROM preset_suits WHERE id=?", (suit_id,)).fetchone() is None:
            raise KeyError(f"preset suit id={suit_id} not found")
    conn.execute("UPDATE clusters SET default_suit_id=? WHERE id=?",
                 (None if suit_id is None else int(suit_id), cluster_id))
    conn.commit()
    return {"cluster_id": cluster_id, "default_suit_id": suit_id}


# ------------------------------------------------------------------ 工具实现（接缝）
def list_presets_op() -> dict[str, Any]:
    """list_presets 工具实现（server.py 只做薄包装）。"""
    conn, _cache = get_session()
    try:
        ensure_seeds(conn)
        return {"status": "ok", "presets": list_suits(conn)}
    except Exception as e:  # noqa: BLE001
        log.exception("list_presets 失败")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def save_preset_op(name: str, steps: list, version: int | None = None) -> dict[str, Any]:
    """save_preset 工具实现：自建方案入库，立即可被 apply_preset 消费。"""
    conn, _cache = get_session()
    try:
        ensure_seeds(conn)
        return {"status": "ok", **save_suit(conn, name, steps, version)}
    except Exception as e:  # noqa: BLE001
        log.exception("save_preset(%s) 失败", name)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def bind_cluster_suit_op(cluster_id: int, suit_id: int | None = None) -> dict[str, Any]:
    """bind_cluster_suit 工具实现：聚类默认方案绑定/解绑（按人路由前置件）。"""
    conn, _cache = get_session()
    try:
        ensure_seeds(conn)
        return {"status": "ok", **bind_cluster_suit(conn, int(cluster_id), suit_id)}
    except Exception as e:  # noqa: BLE001
        log.exception("bind_cluster_suit(%s, %s) 失败", cluster_id, suit_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
