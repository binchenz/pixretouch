"""主体抠图（ISNet）— 工单 08。

能力：从库内图像产出 alpha 蒙版（主体抠图）。**双通道登记语义**（lead 裁决
2026-10-21，#12 预设方案引擎按此消费）：
- output 通道：记录产物 = RGBA PNG（原 RGB + alpha）→ 记录链保持「图像状态」语义，
  bg_replace 等后续工具可直接消费其 alpha（链组合，零重复推理）；
- mask 通道：alpha PNG 以 cache.put("mask:"+sha) 独立登记进内容寻址缓存（独立产物），
  sha 落 records.mask_sha256 并随工具结果返回（alpha_path/alpha_sha256）。
  alpha 既是产物又可被后续步骤当 mask 用——两通道各取所需。

mask 通道的一般语义（后续能力沿用）：承载「渲染实际消费的权重/蒙版图」——
blemish_remove=提示 mask、cutout=alpha、bg_blur=虚化权重图 t。

权重源选型（2026-10-21 curl 勘察，含 lead「零依赖优先」评估结论）：
- x-Liola-x/isnet-general-use-onnx「isnet-general-use.onnx」：apache-2.0，
  176,213,804 B，rembg 生态标准 ISNet 导出 → 选定；
- onnx-community/ISNet-ONNX：agpl-3.0 → 弃（许可污染，兜底字段仅作灾难恢复
  线索，真启用须先自评许可并在此登记结论）；
- AlekseyKorshuk / NimaBoscarino 的 .pth：state_dict 加载需 vendor ISNet 架构
  代码（~700 行）→ 弃；
- torchscript 版 ISNet：hf-mirror 搜索无结果 → 纯 torch 零依赖路线不可行（已报 lead）。
- 结论：裸 onnxruntime（依赖树 numpy>=1.21.6/protobuf/flatbuffers，无钉死项，
  与 rembg pip 包的依赖地狱不同级）+ **锁定 CPUExecutionProvider**——CoreML EP
  数值不保证确定性，会污染内容寻址缓存（确定性 > 加速）。

前/后处理约定（参考 rembg 开源实现白盒重写，不引入该包）：
resize 方形 → (x/255 - ImageNet mean)/std → 输出 min-max 归一 → resize 回原尺寸。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .. import config, data
from ..engine import EngineContext, get_context, get_session, png_sha256
from . import WEIGHT_SPECS, capability, weight

log = logging.getLogger("pixretouch.capabilities.cutout")

WEIGHT_NAME = "isnet"
CAPABILITY_NAME = "cutout"
DEFAULT_INPUT = 1024  # ISNet general-use 标准输入边长；实测模型声明形状后修正

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

weight(WEIGHT_NAME, {
    "repo": "x-Liola-x/isnet-general-use-onnx",
    "file": "isnet-general-use.onnx",
    "kind": "file",
    "modelscope": "onnx-community/ISNet-ONNX",  # caveat: agpl-3.0，见模块 docstring
})

# ---------------------------------------------------------------- ORT 基建（能力模块共用）

_ORT_SESSIONS: dict[str, Any] = {}
_ORT_LOCK = threading.Lock()


def weight_file(weight_name: str) -> Path:
    """models/<权重名>/<spec.file>（file 可含子路径，如 onnx/model.onnx）。"""
    spec = WEIGHT_SPECS[weight_name]
    return config.MODELS_DIR / weight_name / spec["file"]


def ort_session(weight_name: str) -> Any:
    """权重名 → 常驻 ORT InferenceSession（懒加载 + CPU EP 锁定保字节确定）。"""
    sess = _ORT_SESSIONS.get(weight_name)
    if sess is not None:
        return sess
    with _ORT_LOCK:
        sess = _ORT_SESSIONS.get(weight_name)
        if sess is None:
            import onnxruntime as ort

            pt = weight_file(weight_name)
            if not pt.exists():
                raise FileNotFoundError(
                    f"权重缺失：{pt}。先执行：HF_ENDPOINT={config.HF_MIRROR}"
                    f" python -m pixretouch.prefetch"
                )
            so = ort.SessionOptions()
            so.log_severity_level = 3  # ORT 日志不入 stdout（stdio 纪律），只留 error
            sess = ort.InferenceSession(str(pt), sess_options=so, providers=["CPUExecutionProvider"])
            _ORT_SESSIONS[weight_name] = sess
            log.info("ORT 会话常驻：%s（%s）", weight_name, pt)
        return sess


def _declared_size(sess: Any, fallback: int) -> int:
    shape = sess.get_inputs()[0].shape
    dim = shape[-1]
    return dim if isinstance(dim, int) and dim > 0 else fallback


def _preprocess(image_rgb: Any, size: int) -> Any:
    """PIL RGB → (1,3,size,size) float32 ImageNet 归一（numpy 数组）。"""
    import numpy as np
    from PIL import Image

    resized = image_rgb.resize((size, size), Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])


def _squash_2d(arr: Any) -> Any:
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    return a


# ---------------------------------------------------------------- 分割

def segment_alpha(image_rgb: Any) -> Any:
    """PIL RGB → alpha（PIL "L"，同尺寸）。ISNet 确定性推理（CPU EP，同输入同字节）。"""
    import numpy as np
    from PIL import Image

    sess = ort_session(WEIGHT_NAME)
    w, h = image_rgb.size
    x = _preprocess(image_rgb, _declared_size(sess, DEFAULT_INPUT))
    pred = _squash_2d(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    mi, ma = float(pred.min()), float(pred.max())
    pred = np.zeros_like(pred) if ma - mi < 1e-6 else (pred - mi) / (ma - mi)
    alpha = Image.fromarray((pred * 255.0).round().astype(np.uint8))
    return alpha.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- 能力注册

@capability(
    CAPABILITY_NAME,
    params_schema={"variant": "分割模型变体；一期仅 'general'（ISNet general-use）"},
    defaults={"variant": "general"},
    weight=WEIGHT_NAME,
)
def render(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：当前态图像 → RGBA PNG（RGB 原样 + alpha 附加；PNG 无损不再退化链上画质）。"""
    from PIL import Image

    with Image.open(image) as im:
        rgb = im.convert("RGB")
    alpha = ctx.payload.get("alpha")
    if alpha is None:  # 纪律：仅凭 params 可重建（payload 只是省算）
        alpha = segment_alpha(rgb)
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    out = ctx.scratch_file("cutout", ".png")
    rgba.save(out, "PNG")
    return out


# ---------------------------------------------------------------- 工具实现

def cutout_op(image_id: int) -> dict[str, Any]:
    """cutout 工具实现（server.py 只做 @mcp.tool() 薄包装）。"""
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    alpha_tmp: Path | None = None
    try:
        cur = data.get_current(conn, image_id)
        with Image.open(cur["output_path"]) as im:
            embedded = im.getchannel("A") if im.mode == "RGBA" else None
            rgb = im.convert("RGB")
        params = {"variant": "general"}
        alpha = embedded if embedded is not None else segment_alpha(rgb)  # 链组合：已抠过则零推理
        a_sha = png_sha256(alpha)

        ctx = get_context()
        ctx.payload["alpha"] = alpha

        def _render(in_path: Path, prm: dict) -> Path:
            out = render(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, CAPABILITY_NAME, params, _render, mask_sha=a_sha)

        # mask 通道双登记：alpha 固化为独立内容寻址产物（key 前缀 mask: 区分渲染缓存）
        alpha_tmp = ctx.scratch_file("alpha", ".png")
        alpha.save(alpha_tmp, "PNG")
        _, alpha_path = cache.put(f"mask:{a_sha}", alpha_tmp)
        return {
            "status": "ok", "mask_sha256": a_sha,
            "alpha_sha256": a_sha, "alpha_path": str(alpha_path), **result,
        }
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("cutout(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
        if alpha_tmp is not None:
            alpha_tmp.unlink(missing_ok=True)
