"""超分增强（Real-ESRGAN）— 工单 10。

super_resolve(image_id, scale∈{2,4})：原生 x2/x4 双 ONNX 权重放大低分辨率图像。

大图分块（验收核心）：输入按 256px 瓦片 + 16px 单边重叠（stride 224）网格切分，
逐瓦片推理，**只回贴有效中心区**——模型边缘伪影落在重叠带内被丢弃，瓦片接缝处
输出为相邻瓦片各自「远离边界」的区域拼接，无可见接缝；瓦片内部 mod-4 对齐 pad
（奇边输入安全，qa #11 回归）。内存峰值 = 单瓦片 float 中间态 + 全幅输出 uint8
画布（40MP×2 实测峰值 RSS 1925MB，M2 Max/32GB 余量充足，边界测算见 issue #11）。
分块常数是输出字节的一部分 → tiler 描述串进 params 参与缓存键（改常数必须升
TILER_VERSION，旧缓存自然失效）。

输入域留档（qa 探针实证 2026-10-21）：PIL DecompressionBomb 安全阈值（~178MP 硬错、
~89MP 警告）先于本模块 65500px/8GB 输出守卫触发——超大输入在 Image.open 即被
结构化错误容器化（status=error）。**不解除 PIL 安全阈值**（zip-bomb 防线）；
一期输入域实际软上限以 PIL 阈值为准，输出域上限以双守卫为准。

RGBA 链态（纪律 8 增补，几何变换类能力）：RGB 走模型超分；alpha 通道以 lanczos
确定性重采样同步放大——alpha 是蒙版语义，不进生成式模型（防幻觉边缘）；产物保持
RGBA（PNG 无损）。

确定性（纪律 6）：ORT CPUExecutionProvider 锁定；固定分块网格 + uint8 舍入 →
同输入同参数字节一致。

权重源选型（2026-10-21 curl 勘察，lead 前瞻 (a) 结论）：
- SceneWorks/real-esrgan-onnx：bsd-3-clause，原生 x2+x4 双模型
  （67,073,434 / 67,051,616 B）→ 选定：零新增依赖（复用 onnxruntime 1.31.0）；
- liangmu261004/RealESRGAN_x4plus.onnx（68,811,310 B，仅 x4；官方 outscale=2
  语义=4x 后降采样）→ 备胎；
- amd/realesrgan-x4plus（.pth）、Comfy-Org（.safetensors）：加载需 vendor
  RRDBNet/basicsr 架构代码，且 realesrgan/basicsr pip 系重钉死（与 numpy 2.x/
  torch 2.14 冲突面大）→ 弃；
- Samo629 general-x4v3：无 license 标签 → 弃。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import config, data
from ..engine import EngineContext, get_context, get_session
from . import capability, weight
from .cutout import ort_session

log = logging.getLogger("pixretouch.capabilities.superres")

CAPABILITY_NAME = "super_resolve"
TILE = 256          # 输入瓦片边长（px）
OVERLAP = 16        # 单边重叠（px，输入尺度）；stride = TILE - 2*OVERLAP
TILER_VERSION = 1   # 分块常数变更必须递增（tiler 串参与缓存键）
MAX_OUT_DIM = 65500      # JPEG 边长上限（libjpeg 65535，留余量）
MAX_CANVAS_BYTES = 1 << 33  # 输出画布内存上限 8GB（M2 Max/32GB 的安全水位）

# 通道序：SceneWorks README 明确 input = RGB [0,1]（1:1 导出自官方
# RealESRGAN_x2plus/x4plus.pth，onnxruntime-CPU 与 torch 参考 PSNR≈101dB，
# 下载文件 sha256 与 README 公布值逐字节核对一致）；保留开关以备换源。
_INPUT_BGR = False

weight("realesrgan_x2", {
    "repo": "SceneWorks/real-esrgan-onnx",
    "file": "real_esrgan_x2.onnx",
    "kind": "file",
    # 兜底为官方 .pth 命名系（muse/RealESRGAN_x2plus），真走兜底需 vendor RRDBNet
    # 或转 ONNX，届时在此登记结论
    "modelscope": "muse/RealESRGAN_x2plus",
})
weight("realesrgan_x4", {
    "repo": "SceneWorks/real-esrgan-onnx",
    "file": "real_esrgan_x4.onnx",
    "kind": "file",
    "modelscope": "muse/RealESRGAN_x4plus",  # caveat 同上
})


def _windows(dim: int) -> list[int]:
    """一维瓦片窗口起点（含末端对齐窗，保证全覆盖且相邻重叠 ≥ 2*OVERLAP）。"""
    if dim <= TILE:
        return [0]
    stride = TILE - 2 * OVERLAP
    starts = list(range(0, dim - TILE + 1, stride))
    if starts[-1] != dim - TILE:
        starts.append(dim - TILE)
    return starts


def _run_tile(sess: Any, tile_rgb: Any, scale: int) -> Any:
    """单瓦片推理：uint8 HWC RGB → uint8 HWC RGB（×scale）。

    奇边修复（qa #11 实证）：RRDBNet 内部 pixel-shuffle/上采样要求输入边长为
    模型倍数对齐（奇边 → ONNX Reshape 崩）。瓦片先 symmetric pad 到 **mod-4**
    （x2 的 pixel-shuffle 需偶边；mod-4 对 ×2/×4 双模型统一留余量），推理后裁回
    原边×scale；边缘/小图奇边瓦片同路径处理。pad/裁回确定性；偶边输入输出字节与
    修复前完全一致——TILER_VERSION 无需递增（修复前奇边路径只产生结构化错误，
    不存在缓存条目）。
    """
    import numpy as np

    h, w = tile_rgb.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    t = tile_rgb
    if ph or pw:
        t = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    t = t[:, :, ::-1] if _INPUT_BGR else t  # RGB↔BGR（按实测约定）
    x = np.ascontiguousarray(
        np.transpose(t.astype(np.float32) / 255.0, (2, 0, 1))[None]
    )
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    out = np.asarray(out[0])
    if out.shape[0] == 3:  # NCHW → HWC
        out = np.transpose(out, (1, 2, 0))
    if _INPUT_BGR:
        out = out[:, :, ::-1]
    out = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)
    return out[: h * scale, : w * scale]


def upscale(rgb_u8: Any, scale: int) -> Any:
    """整图分块超分：uint8 (h,w,3) RGB → uint8 (h*scale, w*scale) RGB。"""
    import numpy as np

    sess = ort_session(f"realesrgan_x{scale}")
    h, w = rgb_u8.shape[:2]
    out_h, out_w = h * scale, w * scale
    if max(out_h, out_w) > MAX_OUT_DIM:
        raise ValueError(f"输出尺寸 {out_w}x{out_h} 超 JPEG 边长上限 {MAX_OUT_DIM}")
    if out_h * out_w * 3 > MAX_CANVAS_BYTES:
        raise ValueError(f"输出画布 {out_w}x{out_h} 超内存安全水位（{MAX_CANVAS_BYTES >> 30}GB）")

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    ov = OVERLAP * scale
    ys, xs = _windows(h), _windows(w)
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        y1 = min(y0 + TILE, h)
        vy0 = 0 if y0 == 0 else ov
        vy1 = (y1 - y0) * scale if y1 == h else (y1 - y0) * scale - ov
        for x0 in xs:
            x1 = min(x0 + TILE, w)
            vx0 = 0 if x0 == 0 else ov
            vx1 = (x1 - x0) * scale if x1 == w else (x1 - x0) * scale - ov
            tile_out = _run_tile(sess, rgb_u8[y0:y1, x0:x1], scale)
            canvas[y0 * scale + vy0: y0 * scale + vy1,
                   x0 * scale + vx0: x0 * scale + vx1] = tile_out[vy0:vy1, vx0:vx1]
            done += 1
            if done % 50 == 0 or done == total:
                log.info("super_resolve x%d 进度 %d/%d 瓦片", scale, done, total)
    return canvas


# ---------------------------------------------------------------- 能力注册

@capability(
    CAPABILITY_NAME,
    params_schema={
        "scale": "放大倍率，2 或 4（原生双模型）",
        "tiler": "分块器描述串（常数+版本，参与缓存键；实现自动生成）",
    },
    defaults={"scale": 4},
    weight="realesrgan_x4",
)
def render(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：当前态 → 分块超分 → 确定性产物（RGB→JPEG q95/4:4:4；RGBA→PNG，
    alpha 以 lanczos 同步放大——纪律 8 几何变换增补）。"""
    import numpy as np
    from PIL import Image

    scale = int(params["scale"])
    with Image.open(image) as im:
        alpha = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = im.convert("RGB")
    out = upscale(np.asarray(rgb), scale)
    img_out = Image.fromarray(out)
    if alpha is not None:
        w, h = alpha.size
        alpha = alpha.resize((w * scale, h * scale), Image.LANCZOS)
        img_out.putalpha(alpha)
        dst = ctx.scratch_file("super_resolve", ".png")
        img_out.save(dst, "PNG")
    else:
        dst = ctx.scratch_file("super_resolve", ".jpg")
        img_out.save(dst, "JPEG", quality=95, subsampling=0)
    return dst


# ---------------------------------------------------------------- 工具实现

def super_resolve_op(image_id: int, scale: int = 4) -> dict[str, Any]:
    """super_resolve 工具实现（server.py 只做 @mcp.tool() 薄包装）。"""
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    try:
        cur = data.get_current(conn, image_id)
        scale = int(scale)
        if scale not in (2, 4):
            raise ValueError(f"scale 仅支持 2 或 4（原生双模型），收到 {scale!r}")
        with Image.open(cur["output_path"]) as im:
            w, h = im.size
        if max(w * scale, h * scale) > MAX_OUT_DIM:
            raise ValueError(
                f"输出 {w * scale}x{h * scale} 超 JPEG 边长上限 {MAX_OUT_DIM}，"
                f"请先裁剪或改用 scale=2"
            )
        if (w * scale) * (h * scale) * 3 > MAX_CANVAS_BYTES:
            raise ValueError(f"输出画布超内存安全水位（{MAX_CANVAS_BYTES >> 30}GB）")

        params = {"scale": scale, "tiler": f"grid{TILE}o{OVERLAP}v{TILER_VERSION}"}
        ctx = get_context()

        def _render(in_path: Path, prm: dict) -> Path:
            out = render(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, CAPABILITY_NAME, params, _render)
        return {"status": "ok", "width": w * scale, "height": h * scale, **result}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("super_resolve(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
