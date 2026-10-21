"""顶接缝 e2e：迷你相册 → 七阶段工具序列（复刻 album-oneshot.js 的调用序列）。

驱动方式（lead 批准的设计勘察结论）：pytest 无法执行 JS workflow 引擎 →
本测试以**直接 MCP 工具调用**按 oneshot 脚本的同一序列驱动（工具序列=脚本行为
规范，二者由 workflows/ 脚本 docstring 双向锁定）；JS 文本另做 node --check；
真实 DSH 运行归 #14/#15（qa 一发+检查点两模式）。

段结构（lead O1~O6 裁决）：
  S0 入库+选片：ingest_album → selection_report（verdict 唯一事实源，dev-b 契约：
     images[].{image_id,verdict∈keep/review/reject,score,reasons,assess}）→
     set_verdicts(source="report") 持久化 → 人工推翻（用户故事3）=set_verdicts(source="human")；
  S1 聚类：cluster_faces（人物聚类）；
  S2 批量套用：**骨架期跳过**（apply_preset 依赖 task-9 预设引擎，lead O6；落地后接线，
     含批式 fan-out 失败隔离 + mediapipe timeout 重试断言）；
  S2' 联络表：contact_sheet（review 打标、reject 单列、幂等）；
  S3 导出：export_album（规整命名 + manifest）。
终态断言：三交付物 + 库（images/records/meta.verdict/clusters）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mini-album"
NEGATIVES = {"eyes-closed-01-negative.jpg", "waste-blur-01.jpg"}  # qa 制备的反例（已验 reject）


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(config, "MODELS_DIR", REPO_ROOT / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="qa mini-album fixture 缺失")
def test_e2e_album_pipeline_skeleton(isolated_home):
    from pixretouch import data, engine, server

    conn, _ = engine.get_session()

    # ---- S0a 入库（失败隔离 + 幂等 + 非图像 skipped）----
    ing = server.ingest_album(str(FIXTURES))
    assert ing["status"] == "ok", ing
    n_images = len(list(FIXTURES.glob("*.jpg")))
    assert len(ing["ingested"]) == n_images and ing["failed"] == []
    assert any(s["path"].endswith("SOURCES.md") for s in ing["skipped"]), "非图像候选应入 skipped"
    ids = [it["image_id"] for it in ing["ingested"]]
    by_name = {Path(it["path"]).name: it["image_id"] for it in ing["ingested"]}

    # ---- S0b 选片报告（verdict 唯一事实源；未质检图自动补测）----
    rep = server.selection_report(None)
    assert rep["status"] == "ok", rep
    assert rep["album"]["total"] == n_images
    verdict_of = {e["image_id"]: e["verdict"] for e in rep["images"]}
    assert set(verdict_of.values()) <= {"keep", "review", "reject"}
    # qa 已验证的反例校准：闭眼/模糊必 reject（dev-b assess 校准为事实源，不另设常数）
    for neg in NEGATIVES:
        assert verdict_of[by_name[neg]] == "reject", f"{neg} 应判 reject"

    # ---- S0c verdict 持久化（source=report）+ 人工推翻（用户故事3）----
    items = [{"image_id": e["image_id"], "verdict": e["verdict"],
              "reason": "; ".join(e.get("reasons") or [])} for e in rep["images"]]
    sv = server.set_verdicts(items, source="report")
    assert sv["status"] == "ok" and sv["count"] == n_images

    waste_id = by_name["waste-blur-01.jpg"]
    over = server.set_verdicts([{"image_id": waste_id, "verdict": "keep",
                                 "reason": "那张逆光的留下"}], source="human")
    assert over["status"] == "ok"
    meta = json.loads(conn.execute("SELECT meta FROM images WHERE id=?", (waste_id,)).fetchone()["meta"])
    assert meta["verdict"] == {**meta["verdict"], "value": "keep", "source": "human"}

    # ---- S1 聚类（人物聚类；分组细节由 dev-b task-3 测试断言，此处只验段可跑通）----
    cl = server.cluster_faces(None)
    assert cl["status"] == "ok", cl

    # ---- S2 批量套用：骨架期跳过（task-9 预设引擎落地后接线，lead O6）----
    # 接线后此处断言：批式 fan-out（默认批10）逐图结果清单、单图失败隔离、
    # mediapipe timeout 类错误自动重试一次、每图记录链按配方增长。

    # ---- S2' 联络表（交付物②：HTML+PNG；review 打标、reject 单列、幂等）----
    cs = server.contact_sheet(None)
    assert cs["status"] == "ok", cs
    sheet, index = Path(cs["sheet_png"]), Path(cs["index_html"])
    assert sheet.exists() and index.exists()
    with Image.open(sheet) as im:
        assert im.size[0] > 0 and im.size[1] > 0
    n_reject_now = sum(1 for v in verdict_of.values() if v == "reject") - 1  # waste 被人工翻成 keep
    assert cs["stats"]["shown"] == n_images - n_reject_now
    assert cs["stats"]["review_marked"] == sum(1 for v in verdict_of.values() if v == "review")
    assert cs["stats"]["rejected_listed"] == n_reject_now
    assert "联络表" in index.read_text(encoding="utf-8")
    cs2 = server.contact_sheet(None)
    assert cs2["cached"] is True and cs2["directory"] == cs["directory"], "同输入幂等（报告类内容寻址）"

    # ---- S3 导出（交付物③：规整命名 + manifest；reject 跳过）----
    ex = server.export_album(None, album_name="e2e-album")
    assert ex["status"] == "ok", ex
    assert ex["count"] == n_images - n_reject_now and ex["skipped_rejected"] == n_reject_now
    out_dir = Path(ex["directory"])
    files = sorted(p.name for p in out_dir.iterdir() if p.suffix != ".json")
    assert len(files) == ex["count"]
    assert all(re.match(r"^\d{3}_.+\.(jpg|png)$", f) for f in files), files
    manifest = json.loads(Path(ex["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["count"] == ex["count"]
    sample = manifest["items"][0]
    cur = data.get_current(conn, sample["image_id"])
    assert sample["sha256"] == cur["output_sha256"], "manifest sha=当前态产物哈希"
    assert (out_dir / sample["file"]).exists()

    # ---- 库终态 ----
    assert conn.execute("SELECT COUNT(*) c FROM images").fetchone()["c"] == n_images
    assert conn.execute(
        "SELECT COUNT(*) c FROM records WHERE capability='base'"
    ).fetchone()["c"] == n_images
    n_verdict = conn.execute(
        "SELECT COUNT(*) c FROM images WHERE meta LIKE '%\"verdict\"%'"
    ).fetchone()["c"]
    assert n_verdict == n_images, "verdict 全量持久化在 images.meta"
    assert conn.execute("SELECT COUNT(*) c FROM clusters").fetchone()["c"] >= 1


@pytest.mark.skip(reason="S2 apply fan-out 段待 task-9 预设引擎落地后接线（lead O6 裁决）")
def test_e2e_apply_stage_wiring_placeholder():
    """接线清单（task-9 落地后实现）：apply_preset 批式 fan-out（默认批10，args 可调 8~12）；
    批内逐图结构化失败隔离 + mediapipe timeout 类错误自动重试一次（lead 裁决2）；
    每图记录链按配方增长、整体生效或整体无痕（事务性）；随后重跑联络表/导出断言修图后状态。"""
