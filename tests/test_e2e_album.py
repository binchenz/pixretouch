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
  S2 批量套用：已接线 dev-b c79c73a 契约（suit 按名 fan-out、事务化单图、
     apply→assess 成对复检硬规则）；失败隔离+timeout 重试策略镜像（lead 裁决2）
     以故障注入断言；
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

    # ---- S2 批量套用（接线 dev-b c79c73a 契约：suit 按名、事务化单图、
    #      apply→assess 成对复检硬规则；lead 裁决2 策略镜像：timeout 类自动重试
    #      一次再隔离，其余错误直接隔离——与 workflows/album-apply.js 同一策略）----
    from pixretouch.capabilities import preset_engine as preset_cap

    def final_verdict(iid: int):
        m = json.loads(conn.execute("SELECT meta FROM images WHERE id=?", (iid,)).fetchone()["meta"])
        return (m.get("verdict") or {}).get("value")

    targets = [i for i in ids if final_verdict(i) != "reject"]   # O3：reject→跳过
    assert by_name["eyes-closed-01-negative.jpg"] not in targets
    assert waste_id in targets, "人工推翻（source=human）后 fan-out 应消费 meta 终值"

    def apply_with_retry(image_id: int, suit: str = "natural"):
        r = server.apply_preset(image_id, suit)
        attempts = 1
        if r["status"] == "error" and "mediapipe inference timeout" in r.get("error", "").lower():
            r = server.apply_preset(image_id, suit)   # 事务性保证同参重试安全（dev-b 契约）
            attempts = 2
        if r["status"] == "ok":
            rc = server.assess_image(image_id)         # 质检回路硬规则：修图后必复检
            assert rc["status"] == "ok", rc
        return r, attempts

    flaky_id = targets[0]
    broken_id = targets[1] if len(targets) > 1 else None
    real_op = preset_cap.apply_preset_op
    calls: dict[int, int] = {}

    def flaky_op(image_id, suit, cluster_id=None):
        if image_id == flaky_id:
            calls[image_id] = calls.get(image_id, 0) + 1
            if calls[image_id] == 1:
                return {"status": "error", "error": "RuntimeError: mediapipe inference timeout (容器自愈触发)"}
        if broken_id is not None and image_id == broken_id:
            return {"status": "error", "error": "ValueError: 注入的持久性故障"}
        return real_op(image_id, suit, cluster_id=cluster_id)

    applied, isolated = [], []
    preset_cap.apply_preset_op = flaky_op
    try:
        for iid in targets:
            r, attempts = apply_with_retry(iid)
            if r["status"] == "ok":
                applied.append(iid)
                assert r["record_ids"] and len(r["record_ids"]) >= 1
                assert r["routing"], "routing 供联络表「谁用了哪套」打标（dev-b 契约）"
                hist = data.history(conn, iid)
                assert len(hist) == 1 + len(r["record_ids"]), "记录链按配方精确增长"
            else:
                isolated.append({"image_id": iid, "error": r["error"], "attempts": attempts})
    finally:
        preset_cap.apply_preset_op = real_op

    # 失败隔离语义：flaky 重试一次成功；broken 直接隔离且零记录残留；流水线走完全量
    assert flaky_id in applied and calls[flaky_id] == 2, "timeout 类应自动重试一次并成功"
    if broken_id is not None:
        ent = next(x for x in isolated if x["image_id"] == broken_id)
        assert ent["attempts"] == 1, "非 timeout 类错误直接隔离不重试"
        assert [h["capability"] for h in data.history(conn, broken_id)] == ["base"], \
            "事务性：失败 apply 零记录残留（整体无痕）"
    assert len(applied) + len(isolated) == len(targets), "单图失败不阻塞，流水线走完"

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
    assert all(re.match(r"^\d{3}_.+\.(jpg|png|tif|webp)$", f) for f in files), files
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


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="qa mini-album fixture 缺失")
def test_timeout_resolve_by_db(isolated_home, tmp_path):
    """P3 处置分支（lead 硬约束2：超时≠失败≠回滚）——慢渲染 monkeypatch 用例。

    镜像 workflows/album-*.js 的 P3 硬规则：client 超时放弃等待后，server 侧孤儿
    渲染仍会完成并单事务落库；处置必须「查库定论」（history 链已增长=已套用，
    计入成功且**禁止重跑**——防双倍渲染+双份记录），而非当失败直接重试。
    """
    import threading
    import time

    import numpy as np
    from PIL import Image

    from pixretouch import data, engine, server
    from pixretouch.capabilities import preset_engine as preset_cap

    p = tmp_path / "slow.jpg"
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(p, "JPEG")
    r_ing = server.ingest(str(p))
    assert r_ing["status"] == "ok", r_ing
    img_id = r_ing["image_id"]
    conn, _cache = engine.get_session()

    # 纯 image 域自建方案（噪声图无人脸行，person 域步骤会结构化报错——真实流程
    # 选片段已自动补测人脸，此处聚焦 P3 处置分支本身）
    sp = server.save_preset("e2e-timeout-probe", [
        {"capability": "color_grade", "params": {"temperature": 5}, "scope": "image"},
    ])
    assert sp["status"] == "ok", sp

    calls = {"n": 0}
    real_op = preset_cap.apply_preset_op

    def slow_op(image_id, suit, cluster_id=None):
        calls["n"] += 1
        time.sleep(1.5)  # 模拟慢渲染：client 已超时放弃，server 仍在渲染
        return real_op(image_id, suit, cluster_id=cluster_id)

    def resolve_by_db(image_id: int, len_before: int, deadline_s: float = 8.0) -> dict:
        """P3 镜像 helper：查库定论（链增长=已提交成功；宽限内未增长才可安全重跑）。"""
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            if len(data.history(conn, image_id)) > len_before:
                return {"status": "ok", "resolved": "db-committed"}
            time.sleep(0.3)
        return {"status": "timeout_uncommitted", "resolved": "safe-to-retry"}

    preset_cap.apply_preset_op = slow_op
    th = None
    try:
        len_before = len(data.history(conn, img_id))
        # 模拟 client 超时：后台线程发起真实调用（server 侧继续），主流程提前放弃等待
        th = threading.Thread(target=lambda: server.apply_preset(img_id, "e2e-timeout-probe"), daemon=True)
        th.start()
        time.sleep(0.4)  # client 超时点（渲染远未完成）

        res = resolve_by_db(img_id, len_before)  # 查库定论（孤儿渲染 ~1.5s 后单事务落库）
        assert res == {"status": "ok", "resolved": "db-committed"}, res
        assert calls["n"] == 1, "查库定论成功后禁止重跑（防双倍渲染+双份记录）"
        hist = data.history(conn, img_id)
        assert len(hist) > len_before and hist[-1]["capability"] != "base"
    finally:
        preset_cap.apply_preset_op = real_op
        if th is not None:
            th.join(15)
