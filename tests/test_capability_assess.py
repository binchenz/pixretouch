"""选片质检测试（工单 04 / issue #5 验收项）。

验收映射：
- assess 单图工具落库            → test_assess_writes_meta_no_record_appended
                                   + test_assess_op_on_real_portrait（op 主接缝）
- 选片报告相册级 JSON+取舍+理由   → test_selection_report_structure_and_autofill
                                   + test_mini_album_verdicts
- EAR 闭眼判定 fixture 正反例     → test_ear_geometry_open_vs_closed（几何级，确定性）
                                   + test_assess_op_on_real_portrait（真人睁眼正例）
                                   + test_eyes_closed_negative_op（真人闭眼负例）
- 人脸框与 landmark 持久化        → test_faces_persist_and_roundtrip

纪律（Spec·Testing Decisions + 铁律 8 + 工单 03 范本）：
- 主接缝 = op 函数（server @mcp.tool() 薄包装的实现体）；
- 只断言外部行为：meta 维度、faces 表行、报告 JSON 结构、verdict 性质；属性式断言；
- 库/缓存/报告 tmp_path 隔离（conftest 全局兜底 PIXRETOUCH_HOME），MODELS_DIR 指向
  仓库 models/（真实权重跨测试共享，零网络）；权重缺失 skipif；绝不污染 var/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixretouch import data
from pixretouch.capabilities import assess as A
from pixretouch.capabilities import faces as F

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "mini-album"

MODELS_READY = (
    (REPO / "models" / "face_landmarker" / "face_landmarker.task").exists()
    and (REPO / "models" / "face_detector" / "blaze_face_short_range.tflite").exists()
)
needs_models = pytest.mark.skipif(not MODELS_READY, reason="人脸模型权重缺失（先跑 prefetch）")
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES / "portrait-01-frontal-clear.jpg").exists(), reason="mini-album fixture 缺失"
)


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """库/缓存/报告指向 tmp_path；权重共享仓库 models/；引擎会话前后清场（工单 03 范本）。"""
    from pixretouch import config, engine

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "MODELS_DIR", REPO / "models")
    engine.reset_session()
    yield tmp_path
    engine.reset_session()


@pytest.fixture(scope="module", autouse=True)
def _release_models():
    """模块收场时优雅关闭 mediapipe 对象（避免解释器退出期 __del__ 噪声）。"""
    yield
    F.reset_models()


@pytest.fixture()
def conn(tmp_path):
    c = data.connect(tmp_path / "test.db")
    data.init_schema(c)
    yield c
    c.close()


def _ingest(path: Path):
    from pixretouch import engine

    conn, _cache = engine.get_session()
    return conn, data.ingest(conn, path)


# ------------------------------------------------------------------ 合成图
def _noise_jpeg(path: Path, w: int = 640, h: int = 480, seed: int = 7, level: str = "mid") -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    if level == "dark":
        arr = rng.integers(10, 30, (h, w, 3), dtype=np.uint8)
    elif level == "bright":
        arr = rng.integers(235, 256, (h, w, 3), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = (200, 30, 30)
    Image.fromarray(arr).save(path, quality=92)
    return path


def _blur(src: Path, dst: Path, sigma: float = 9.0) -> Path:
    import cv2

    img = cv2.imread(str(src))
    cv2.imwrite(str(dst), cv2.GaussianBlur(img, (0, 0), sigma))
    return dst


def _dims(**over) -> dict:
    """构造一份合法的 assess 维度 dict（_verdict 输入）。"""
    base = {
        "width": 1024, "height": 768, "sharpness": 400.0, "exposure_mean": 128.0,
        "clip_hi": 0.01, "clip_lo": 0.01, "face_detection": True, "face_count": 1,
        "max_face_area": 0.06, "max_face_side": 250, "face_exposure_mean": 128.0,
        "eyes_closed": False, "wink": False, "ear_threshold": F.EAR_CLOSED_THRESHOLD,
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ EAR 几何正反例（验收项 3）
def test_ear_geometry_open_vs_closed():
    """确定性几何 fixture：睁眼 6 点 EAR 高于阈值，闭眼低于阈值（正反例）。"""
    # 睁眼：外眼角(0,0) 内眼角(30,0)，上睑两点 y=-5，下睑两点 y=+5 → EAR=(10+10)/(2*30)≈0.33
    open_eye = [(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)]
    # 闭眼：上下睑贴合到 y=±0.25 → EAR≈0.017
    closed_eye = [(0, 0), (10, -0.25), (20, -0.25), (30, 0), (20, 0.25), (10, 0.25)]
    ear_open = F.eye_aspect_ratio(open_eye)
    ear_closed = F.eye_aspect_ratio(closed_eye)
    assert ear_open > F.EAR_CLOSED_THRESHOLD > ear_closed
    assert 0.25 < ear_open < 0.45 and ear_closed < 0.05

    face_open = F.Face(bbox=(0, 0, 10, 10), quality=0.9, landmarks=[], ear_left=ear_open, ear_right=ear_open)
    face_closed = F.Face(bbox=(0, 0, 10, 10), quality=0.9, landmarks=[], ear_left=ear_closed, ear_right=ear_closed)
    face_wink = F.Face(bbox=(0, 0, 10, 10), quality=0.9, landmarks=[], ear_left=ear_open, ear_right=ear_closed)
    assert not face_open.eyes_closed() and not face_open.wink()
    assert face_closed.eyes_closed() and not face_closed.wink()      # 反例：双眼闭
    assert face_wink.wink() and not face_wink.eyes_closed()          # 单眼 → 复核而非闭眼


# ------------------------------------------------------------------ 清晰度/曝光性质（无模型）
def test_sharpness_orders_blur_below_sharp(tmp_path):
    sharp = _noise_jpeg(tmp_path / "sharp.jpg")
    blurred = _blur(sharp, tmp_path / "blurred.jpg")
    ds = A.pixel_dimensions(sharp)
    db = A.pixel_dimensions(blurred)
    assert ds["sharpness"] > db["sharpness"]
    assert db["sharpness"] < A.SHARP_FLOOR < ds["sharpness"]


def test_exposure_flags_dark_and_bright(tmp_path):
    dark = A.pixel_dimensions(_noise_jpeg(tmp_path / "dark.jpg", level="dark"))
    bright = A.pixel_dimensions(_noise_jpeg(tmp_path / "bright.jpg", level="bright"))
    assert dark["exposure_mean"] < A.EXPOSURE_LOW
    assert bright["exposure_mean"] > A.EXPOSURE_HIGH


def test_verdict_rules():
    v, reasons, _ = A._verdict(_dims(face_count=0))
    assert v == "reject" and any("未检出人脸" in r for r in reasons)

    v, reasons, _ = A._verdict(_dims(eyes_closed=True))
    assert v == "reject" and any("闭眼" in r for r in reasons)

    v, reasons, _ = A._verdict(_dims(width=160, height=120, sharpness=900.0))
    assert v == "reject" and any("分辨率不足" in r for r in reasons)

    # 模糊门控：EAR 结论不可靠时不出眼部理由
    v, reasons, _ = A._verdict(_dims(sharpness=5.0, eyes_closed=True))
    assert v == "reject" and any("模糊" in r for r in reasons)
    assert not any("闭眼" in r for r in reasons)

    # 白背景高溢出但脸部曝光正常 → 不淘汰（portrait-02 实测场景）
    v, reasons, _ = A._verdict(_dims(clip_hi=0.41, face_exposure_mean=187.7))
    assert v == "keep" and reasons == []

    # 绝对像素不足的脸（可检出但无修图价值）
    v, reasons, _ = A._verdict(_dims(max_face_side=28))
    assert v == "reject" and any("分辨率不足" in r for r in reasons)

    v, reasons, score = A._verdict(_dims())
    assert v == "keep" and reasons == [] and 0.0 < score <= 1.0


# ------------------------------------------------------------------ assess 落库（验收项 1）
def test_assess_writes_meta_no_record_appended(conn, tmp_path):
    src = _noise_jpeg(tmp_path / "photo.jpg")
    r = data.ingest(conn, src)
    img = r["image_id"]
    hist_before = len(data.history(conn, img))

    out = A.assess_image(conn, img, detect_faces=False)
    a = out["assess"]
    assert a["face_detection"] is False
    assert a["sharpness"] > A.SHARP_FLOOR

    # 维度落库到图像行
    row = conn.execute("SELECT meta FROM images WHERE id=?", (img,)).fetchone()
    meta = json.loads(row["meta"])
    assert "assess" in meta and meta["assess"]["sharpness"] == a["sharpness"]

    # 分析不是编辑：记录链不动、指针不动
    assert len(data.history(conn, img)) == hist_before
    assert data.get_current(conn, img)["capability"] == "base"


# ------------------------------------------------------------------ 选片报告（验收项 2）
def test_selection_report_structure_and_autofill(conn, tmp_path):
    sharp = data.ingest(conn, _noise_jpeg(tmp_path / "good.jpg", seed=1))["image_id"]
    dark = data.ingest(conn, _noise_jpeg(tmp_path / "dark.jpg", seed=2, level="dark"))["image_id"]
    lazy = data.ingest(conn, _noise_jpeg(tmp_path / "lazy.jpg", seed=3))["image_id"]

    A.assess_image(conn, sharp, detect_faces=False)
    A.assess_image(conn, dark, detect_faces=False)
    # lazy 不质检：报告必须自动补测（detect_faces=False → 纯像素维度，无模型依赖）

    reports_dir = tmp_path / "reports"
    rep = A.selection_report(conn, reports_dir=reports_dir, detect_faces=False)
    assert rep["album"] == {"total": 3, "keep": 2, "review": 0, "reject": 1}
    by_id = {e["image_id"]: e for e in rep["images"]}
    assert by_id[dark]["verdict"] == "reject" and by_id[dark]["reasons"]
    assert by_id[sharp]["verdict"] == "keep" and by_id[lazy]["verdict"] == "keep"
    # 排序：keep 段在 reject 段之前
    assert [e["verdict"] for e in rep["images"]] == ["keep", "keep", "reject"]
    # JSON 落盘（注入目录，不碰 var/）
    saved = Path(rep["report_path"])
    assert saved.exists() and saved.parent == reports_dir
    assert json.loads(saved.read_text())["album"]["total"] == 3
    # 自动补测已写回 meta
    lazy_meta = json.loads(conn.execute("SELECT meta FROM images WHERE id=?", (lazy,)).fetchone()["meta"])
    assert lazy_meta["assess"]["face_detection"] is False


# ------------------------------------------------------------------ op 接缝（结构化错误，无模型依赖）
def test_op_structured_errors(isolated_home):
    """坏 image_id → 结构化错误不抛穿；空库报告 → ok 且零图。"""
    r = A.assess_image_op(999999)
    assert r["status"] == "error" and "KeyError" in r["error"]
    r = A.detect_faces_op(999999)
    assert r["status"] == "error" and "KeyError" in r["error"]
    r = A.selection_report_op([999999])
    assert r["status"] == "error" and "KeyError" in r["error"]

    r = A.selection_report_op([])
    assert r["status"] == "ok" and r["album"] == {"total": 0, "keep": 0, "review": 0, "reject": 0}


# ------------------------------------------------------------------ 集成：真人 fixture（需模型）
@needs_models
@needs_fixtures
def test_assess_op_on_real_portrait(isolated_home):
    """真人睁眼正例（op 主接缝）：维度落库、EAR 高于阈值、记录链不增长。"""
    conn, ing = _ingest(FIXTURES / "portrait-01-frontal-clear.jpg")
    img = ing["image_id"]

    res = A.assess_image_op(img)
    assert res["status"] == "ok", res
    a = res["assess"]
    assert a["face_count"] == 1
    assert a["eyes_closed"] is False and a["wink"] is False
    face = a["faces"][0]
    assert face["ear_left"] > F.EAR_CLOSED_THRESHOLD and face["ear_right"] > F.EAR_CLOSED_THRESHOLD
    assert face["quality"] and face["quality"] > 0.5
    assert a["max_face_side"] >= A.FACE_SIDE_REVIEW

    # 维度落库 + 分析不是编辑
    meta = json.loads(conn.execute("SELECT meta FROM images WHERE id=?", (img,)).fetchone()["meta"])
    assert meta["assess"]["face_count"] == 1
    assert [h["capability"] for h in data.history(conn, img)] == ["base"]


@needs_models
@needs_fixtures
def test_eyes_closed_negative_op(isolated_home):
    """真人闭眼负例：双眼 EAR 低于阈值 → eyes_closed，选片报告 reject「闭眼」。"""
    conn, ing = _ingest(FIXTURES / "eyes-closed-01-negative.jpg")
    res = A.assess_image_op(ing["image_id"])
    assert res["status"] == "ok"
    a = res["assess"]
    assert a["face_count"] == 1 and a["eyes_closed"] is True
    assert a["faces"][0]["ear_left"] < F.EAR_CLOSED_THRESHOLD
    assert a["faces"][0]["ear_right"] < F.EAR_CLOSED_THRESHOLD

    rep = A.selection_report_op([ing["image_id"]])
    assert rep["status"] == "ok"
    assert rep["images"][0]["verdict"] == "reject"
    assert any("闭眼" in r for r in rep["images"][0]["reasons"])


@needs_models
@needs_fixtures
def test_faces_persist_and_roundtrip(isolated_home):
    """人脸框与 landmark 持久化到 faces 表；读回结构完整；重跑覆盖不翻倍（幂等）。"""
    conn, ing = _ingest(FIXTURES / "group-01-three-frontal.jpg")
    img = ing["image_id"]

    res = A.detect_faces_op(img)
    assert res["status"] == "ok"
    assert res["face_count"] == 3, "3 人合影必须检出 3 脸（含低分脸，校准阈值 0.2/0.35）"

    rows = F.load_faces(conn, img)
    assert len(rows) == 3
    for r in rows:
        assert len(r["bbox"]) == 4 and all(isinstance(v, int) for v in r["bbox"])
        assert len(r["landmark"]) == 478
        assert r["quality"] is None or 0.0 <= r["quality"] <= 1.0
        x, y, w, h = r["bbox"]
        assert w > 0 and h > 0 and x >= 0 and y >= 0

    # 重跑 → 覆盖式写入，行数不翻倍
    A.detect_faces_op(img)
    assert len(F.load_faces(conn, img)) == 3


@needs_models
@needs_fixtures
def test_mini_album_verdicts(isolated_home):
    """mini-album 六张端到端选片：正例 keep、负例 reject 且理由可读。"""
    ids = {}
    for f in sorted(FIXTURES.glob("*.jpg")):
        conn, ing = _ingest(f)
        ids[f.name] = ing["image_id"]
        assert A.assess_image_op(ing["image_id"])["status"] == "ok"

    rep = A.selection_report_op(list(ids.values()))
    assert rep["status"] == "ok"
    verdict_by_name = {n: next(e["verdict"] for e in rep["images"] if e["image_id"] == i) for n, i in ids.items()}
    assert verdict_by_name["portrait-01-frontal-clear.jpg"] == "keep"
    assert verdict_by_name["portrait-02-frontal-clear.jpg"] == "keep"
    assert verdict_by_name["group-01-three-frontal.jpg"] == "keep"
    assert verdict_by_name["eyes-closed-01-negative.jpg"] == "reject"
    assert verdict_by_name["waste-blur-01.jpg"] == "reject"
    assert verdict_by_name["lowres-01.jpg"] == "reject"
    for e in rep["images"]:
        if e["verdict"] == "reject":
            assert e["reasons"], "reject 必须给出可读理由"
    # 报告落盘在隔离 VAR_DIR 下（不碰真实 var/）
    assert Path(rep["report_path"]).exists()
    assert str(isolated_home) in rep["report_path"]


# ------------------------------------------------------------------ 推理容器（issue #5 A 类挂起回归）
def test_inference_container_timeout_recycles_generation(monkeypatch):
    """容器自愈（无需模型，顺序无关）：作业超时 → 结构化 RuntimeError + 当代毒化；
    下次调用重建新代（计数器 +1）并正常返回。"""
    import time

    inf = F._INFERENCE
    monkeypatch.setattr(F, "LOAD_TIMEOUT_S", 0.5)
    monkeypatch.setattr(F, "INFER_TIMEOUT_S", 0.5)
    with pytest.raises(RuntimeError, match="timeout"):
        inf.call(lambda handles: time.sleep(5))
    assert inf._gen is None, "超时后当代必须被毒化弃置"

    counter_at_poison = inf.gen_counter
    monkeypatch.setattr(F, "LOAD_TIMEOUT_S", 15.0)
    r = inf.call(lambda handles: ("ok", isinstance(handles, dict)))
    assert r == ("ok", True), "自愈：新代必须正常执行作业"
    assert inf.gen_counter == counter_at_poison + 1, "毒化后下次调用必须重建新代"
    F.reset_models()


@needs_models
@needs_fixtures
def test_assess_then_detect_cross_thread_no_deadlock(isolated_home):
    """回归（issue #5 A 类挂起）：assess×6 → detect_faces，每步各在新线程执行
    （模拟 MCP 工作线程派发）；超时守卫：任何一步挂起即红，且 daemon 线程不阻塞
    pytest 退出。容器化后最坏情形 = 结构化超时错误（status=error）而非永久挂起。
    """
    import threading

    names = [
        "portrait-01-frontal-clear.jpg",
        "portrait-02-frontal-clear.jpg",
        "group-01-three-frontal.jpg",
        "eyes-closed-01-negative.jpg",
        "waste-blur-01.jpg",
        "lowres-01.jpg",
    ]
    ids = {}
    for n in names:
        conn, ing = _ingest(FIXTURES / n)
        ids[n] = ing["image_id"]

    results: dict = {}

    def run(key, fn, *a):
        results[key] = fn(*a)

    for n in names:
        t = threading.Thread(target=run, args=(n, A.assess_image_op, ids[n]), daemon=True)
        t.start()
        t.join(timeout=180)
        assert not t.is_alive(), f"assess({n}) 挂起（issue #5 A 类回归）"
        assert results[n]["status"] == "ok", results[n]

    t = threading.Thread(target=run, args=("detect", A.detect_faces_op,
                                           ids["group-01-three-frontal.jpg"]), daemon=True)
    t.start()
    t.join(timeout=180)
    assert not t.is_alive(), "detect_faces 在 assess×6 后挂起（issue #5 A 类回归）"
    assert results["detect"]["status"] == "ok", results["detect"]
    assert results["detect"]["face_count"] == 3
