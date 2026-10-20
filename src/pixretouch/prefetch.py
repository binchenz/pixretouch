"""权重后台预取：hf-mirror 优先，ModelScope 兜底位（由各能力工单按需接入）。

用法：python -m pixretouch.prefetch [--json]
注册表为空时是零操作（骨架期常态，能力工单逐个补录 WEIGHT_SPECS）。
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from . import config

log = logging.getLogger("pixretouch.prefetch")


def fetch_all() -> dict[str, Any]:
    from .capabilities import WEIGHT_SPECS, load_all

    load_all()  # 触发各能力模块 @capability/weight 注册副作用，否则注册表恒空

    if not WEIGHT_SPECS:
        return {"registry_empty": True, "results": {}}

    from huggingface_hub import hf_hub_download, snapshot_download

    results: dict[str, Any] = {}
    for name, spec in WEIGHT_SPECS.items():
        dest = config.MODELS_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if spec.get("kind", "file") == "repo":
                path: Any = snapshot_download(spec["repo"], local_dir=dest, endpoint=config.HF_MIRROR)
            else:
                path = hf_hub_download(spec["repo"], spec["file"], local_dir=dest, endpoint=config.HF_MIRROR)
            results[name] = {"ok": True, "path": str(path)}
            log.info("prefetched %s -> %s", name, path)
        except Exception as e:  # noqa: BLE001
            results[name] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "modelscope_fallback_id": spec.get("modelscope"),
            }
            log.warning("prefetch failed for %s: %s", name, e)
    return {"registry_empty": False, "results": results}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    print(json.dumps(fetch_all(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
