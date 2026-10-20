"""工具接缝冒烟测试（骨架期）：doctor 返回结构化报告。

测试纪律（Spec · Testing Decisions）：只断言外部行为——返回结构、状态枚举、路径存在；
不断言实现细节，不绑定具体版本号字节。
"""
from pixretouch import config, doctor


def test_doctor_returns_structured_report():
    report = doctor.run_checks()
    assert report["status"] in ("ok", "degraded")
    for key in ("python", "torch", "libs", "disk", "mirrors", "models", "paths"):
        assert key in report, f"missing section: {key}"
    assert report["python"]["ok"] is True


def test_config_dirs_created():
    config.ensure_dirs()
    for d in (config.VAR_DIR, config.CACHE_DIR, config.MODELS_DIR, config.ALBUM_INBOX):
        assert d.is_dir()


def test_doctor_models_registry_shape():
    report = doctor.run_checks()
    models = report["models"]
    assert models["registry_size"] >= 0
    assert set(models["present"]) | set(models["missing"]) <= set(
        __import__("pixretouch.capabilities", fromlist=["WEIGHT_SPECS"]).WEIGHT_SPECS
    )
