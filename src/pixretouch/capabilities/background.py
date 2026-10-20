"""背景替换/虚化 — 工单 08。

bg_replace：主体抠图（alpha 复用 cutout.segment_alpha；当前态已是 RGBA 则直接消费
其 alpha——链组合零推理）→ 与纯色或指定背景图合成。边缘羽化（alpha 高斯过渡）+
溢色处理（premultiplied 腐蚀采样：边缘带前景色改由内缩腐蚀估计，去除原背景留边）。
bg_blur：Depth-Anything-V2（small，ONNX）深度驱动的自然虚化——深度图 → 虚化权重图 t
（focus 保护：depth≥focus 恒清晰，主体清晰度不变），4 级模糊阶梯按 t 连续插值合成，
强度可调。t 图走 mask 通道双登记（mask_sha256/mask_path），测试可自参照做属性断言。

mask 通道双登记规则（与 cutout.py 一致）：params 可确定性派生的 mask 仅记 sha；
模型派生的 mask（alpha/深度权重图）另以 cache.put("mask:"+sha) 固化内容寻址文件。

外部素材规则（lead 裁决 2026-10-21，模式 docstring 纪律 7）：背景图等外部素材以其
**内容 sha256** 记入 params（bg_sha256）参与缓存键，**路径不进键**；素材内容以
cache.put("material:"+sha) 固化进内容寻址存储，渲染经 ctx.cache.path_for(sha) 按 sha
取回（payload 仅省算）——内容变 ⇒ 键变不假命中；同内容换路径 ⇒ 键不变必命中。
RGBA 链态（纪律 8）：bg_blur 只处理 RGB、alpha 直通、产物保持 RGBA（PNG 无损）；
bg_replace 是合成类操作——消费 alpha 输出 RGB 属其语义本身，不算平整破坏。

权重源选型（2026-10-21 curl 勘察）：
- onnx-community/depth-anything-v2-small「onnx/model.onnx」：apache-2.0，
  99,060,839 B → 选定（V2 合成数据训练许可干净；small 版 CPU ~1s 够用）；
- depth-anything/Depth-Anything-V2-Small-hf：transformers 格式需新增重依赖 → 弃；
- torchscript 版 Depth-Anything：hf-mirror 搜索无结果 → 纯 torch 零依赖路线不可行；
- onnxruntime 锁 CPUExecutionProvider（同 cutout.py：确定性 > 加速）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import config, data
from ..engine import EngineContext, get_context, get_session, png_sha256
from . import capability, weight
from .cutout import _declared_size, _preprocess, _squash_2d, ort_session, segment_alpha

log = logging.getLogger("pixretouch.capabilities.background")

DEPTH_WEIGHT = "depth_anything_v2_small"
DEFAULT_DEPTH_INPUT = 518  # DINOv2 patch14：518 = 14×37

weight(DEPTH_WEIGHT, {
    "repo": "onnx-community/depth-anything-v2-small",
    "file": "onnx/model.onnx",
    "kind": "file",
    # caveat：ModelScope 上 depth-anything 组织目前只勘察到 Large 档（safetensors/非 ONNX），
    # 真走兜底需换档或转 ONNX，届时在此登记结论
    "modelscope": "depth-anything/Depth-Anything-V2-Large",
})


# ---------------------------------------------------------------- 深度

def depth_map(image_rgb: Any) -> Any:
    """PIL RGB → 归一化视差 np.float32 (h,w) ∈[0,1]，1=最近（Depth-Anything 语义）。"""
    import cv2
    import numpy as np

    sess = ort_session(DEPTH_WEIGHT)
    w, h = image_rgb.size
    x = _preprocess(image_rgb, _declared_size(sess, DEFAULT_DEPTH_INPUT))
    d = _squash_2d(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    mi, ma = float(d.min()), float(d.max())
    d = np.zeros_like(d) if ma - mi < 1e-6 else (d - mi) / (ma - mi)
    return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- bg_replace

def _cover_fit(bg_img: Any, size: tuple[int, int]) -> Any:
    """背景图等比放大到盖满目标尺寸后中心裁剪（cover）。"""
    from PIL import Image

    w, h = size
    bw, bh = bg_img.size
    scale = max(w / bw, h / bh)
    nb = bg_img.resize((max(w, round(bw * scale)), max(h, round(bh * scale))), Image.LANCZOS)
    left, top = (nb.width - w) // 2, (nb.height - h) // 2
    return nb.crop((left, top, left + w, top + h))


def _bg_array(params: dict, size: tuple[int, int], ctx: EngineContext | None) -> Any:
    """params → 背景 float32 (h,w,3) 0..255。payload 有预构建则直接用（仅省算）。"""
    import numpy as np
    from PIL import Image

    if ctx is not None and ctx.payload.get("bg") is not None:
        return ctx.payload["bg"]
    w, h = size
    if params["mode"] == "color":
        return np.full((h, w, 3), params["color"], dtype=np.float32)
    # 正规取用路径：内容寻址存储（op 阶段已固化 material:<sha>）；params 只含 sha 不含路径
    if ctx is None or ctx.cache is None:
        raise FileNotFoundError("背景素材取用需 ctx.cache（素材规则，模式 docstring 纪律 7）")
    mat = ctx.cache.path_for(params["bg_sha256"])
    if not mat.exists():
        raise FileNotFoundError(f"背景素材未在内容寻址存储中：{params['bg_sha256']}")
    with Image.open(mat) as im:
        bg = _cover_fit(im.convert("RGB"), (w, h))
    return np.asarray(bg, dtype=np.float32)


@capability(
    "bg_replace",
    params_schema={
        "mode": "'color' 纯色 | 'image' 指定背景图",
        "color": "[r,g,b] 0..255（mode=color）",
        "bg_sha256": "背景图内容 sha256（mode=image，自动填充，参与缓存键；路径不进键）",
        "fit": "背景图适配方式，一期仅 'cover'",
        "feather": "边缘羽化半径（像素）0..10，默认 2",
        "despill": "溢色处理开关，默认 true",
    },
    defaults={"feather": 2.0, "despill": True, "fit": "cover"},
    weight=None,
)
def render_bg_replace(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：当前态 + alpha → 与新背景合成的 RGB JPEG（确定性编码 q95/4:4:4）。"""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image) as im:
        alpha = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = im.convert("RGB")
    w, h = rgb.size
    if alpha is None:
        alpha = ctx.payload.get("alpha")
        if alpha is None:  # 纪律：仅凭 params 可重建
            alpha = segment_alpha(rgb)

    a = np.asarray(alpha, dtype=np.float32) / 255.0
    feather = float(params.get("feather", 2.0))
    if feather > 0:
        k = int(2 * round(feather) + 1)
        a = cv2.GaussianBlur(a, (k, k), 0)
    a3 = a[..., None]

    fg = np.asarray(rgb, dtype=np.float32)
    if params.get("despill", True):
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        a_e = cv2.erode(a, kern)
        fa_e = cv2.erode(fg * a3, kern)
        fg_est = fa_e / np.maximum(a_e[..., None], 1e-6)
        band = (a_e > 0.05) & (a > 0.02) & (a < 0.98)
        fg = np.where(band[..., None], fg_est, fg)

    bg = _bg_array(params, (w, h), ctx)
    out = np.clip(bg * (1.0 - a3) + fg * a3, 0, 255).round().astype(np.uint8)
    dst = ctx.scratch_file("bg_replace", ".jpg")
    Image.fromarray(out).save(dst, "JPEG", quality=95, subsampling=0)
    return dst


# ---------------------------------------------------------------- bg_blur

@capability(
    "bg_blur",
    params_schema={
        "strength": "虚化强度 0..1（缩放权重图 t 的上限），默认 0.6",
        "max_blur": "最大模糊核直径 px（自动取奇数），3..101，默认 31",
        "focus": "主体保护深度阈值 0..1：depth≥focus 恒清晰，默认 0.9",
    },
    defaults={"strength": 0.6, "max_blur": 31, "focus": 0.9},
    weight=DEPTH_WEIGHT,
)
def render_bg_blur(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：深度→权重图 t→4 级模糊阶梯插值合成（t=0 处像素原样，主体清晰度不变）。"""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image) as im:
        alpha_ch = im.getchannel("A") if im.mode == "RGBA" else None  # 纪律 8：alpha 直通
        rgb = im.convert("RGB")
    t = ctx.payload.get("t")
    if t is None:  # 纪律：仅凭 params 可重建
        t = blur_weight_map(rgb, params)

    img_f = np.asarray(rgb, dtype=np.float32)
    strength = float(params["strength"])
    max_blur = int(params["max_blur"])
    levels = 4
    base = img_f
    tn = t * levels
    for i in range(1, levels + 1):
        k = max(3, int(max_blur * i / levels) | 1)
        lvl = cv2.GaussianBlur(img_f, (k, k), 0)
        w_i = np.clip(tn - (i - 1), 0.0, 1.0)[..., None]
        base = base * (1.0 - w_i) + lvl * w_i
    out = np.clip(base, 0, 255).round().astype(np.uint8)
    img_out = Image.fromarray(out)
    if alpha_ch is not None:
        img_out.putalpha(alpha_ch)
        dst = ctx.scratch_file("bg_blur", ".png")
        img_out.save(dst, "PNG")  # RGBA 态保持无损
    else:
        dst = ctx.scratch_file("bg_blur", ".jpg")
        img_out.save(dst, "JPEG", quality=95, subsampling=0)
    return dst


def blur_weight_map(image_rgb: Any, params: dict) -> Any:
    """深度 → 虚化权重图 t ∈[0,strength]：depth≥focus 恒 0（清晰），远端趋近 strength。"""
    import cv2
    import numpy as np

    depth = depth_map(image_rgb)
    focus = float(params["focus"])
    t = np.clip((focus - depth) / max(focus, 1e-6), 0.0, 1.0) * float(params["strength"])
    return cv2.GaussianBlur(t.astype(np.float32), (0, 0), 3.0)  # 过渡防 banding


# ---------------------------------------------------------------- 工具实现

def _validate_replace(color: Any, background_path: Any, feather: Any, despill: Any) -> None:
    if (color is None) == (background_path is None):
        raise ValueError("color 与 background_path 必须二选一")
    if color is not None:
        if len(color) != 3 or any(not (0 <= int(c) <= 255) for c in color):
            raise ValueError(f"color 须为 3 个 0..255 整数，收到 {color!r}")
    if not (0.0 <= float(feather) <= 10.0):
        raise ValueError(f"feather 须在 0..10，收到 {feather!r}")
    bool(despill)


def _normalize_replace(color: Any, background_path: Any, feather: Any,
                       despill: Any) -> tuple[dict[str, Any], "Path | None"]:
    """规范化 → (params, 背景图解析路径或 None)。params 只含内容 sha 不含路径（素材规则）。"""
    params: dict[str, Any] = {"feather": round(float(feather), 3), "despill": bool(despill)}
    if color is not None:
        params.update({"mode": "color", "color": [int(c) for c in color]})
        return params, None
    p = Path(background_path).expanduser()
    if not p.is_absolute():
        p = config.PROJECT_HOME / p
    if not p.exists():
        raise ValueError(f"背景图不存在：{p}")
    params.update({
        "mode": "image", "fit": "cover",
        "bg_sha256": data.file_sha256(p),  # 内容 sha 进缓存键；路径不进键（lead 裁决）
    })
    return params, p


def bg_replace_op(image_id: int, color: list | None = None, background_path: str | None = None,
                  feather: float = 2.0, despill: bool = True) -> dict[str, Any]:
    """bg_replace 工具实现（server.py 只做薄包装）。"""
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    mask_tmp: Path | None = None
    try:
        cur = data.get_current(conn, image_id)
        _validate_replace(color, background_path, feather, despill)
        params, bg_src = _normalize_replace(color, background_path, feather, despill)
        if bg_src is not None:
            cache.put(f"material:{params['bg_sha256']}", bg_src)  # 素材固化进内容寻址存储

        with Image.open(cur["output_path"]) as im:
            embedded = im.getchannel("A") if im.mode == "RGBA" else None
            rgb = im.convert("RGB")
            size = im.size
        alpha = embedded if embedded is not None else segment_alpha(rgb)

        # mask 通道：羽化后 alpha（渲染实际消费的权重图）——模型派生，双登记
        import cv2
        import numpy as np
        a = np.asarray(alpha, dtype=np.float32) / 255.0
        if params["feather"] > 0:
            k = int(2 * round(params["feather"]) + 1)
            a = cv2.GaussianBlur(a, (k, k), 0)
        a_img = Image.fromarray((a * 255.0).round().astype(np.uint8))
        m_sha = png_sha256(a_img)

        ctx = get_context()
        ctx.cache = cache  # 素材正规取用入口：path_for(bg_sha256)
        ctx.payload["alpha"] = alpha
        if params["mode"] == "image":
            with Image.open(bg_src) as im:
                ctx.payload["bg"] = np.asarray(_cover_fit(im.convert("RGB"), size), dtype=np.float32)
        else:
            ctx.payload["bg"] = np.full((size[1], size[0], 3), params["color"], dtype=np.float32)

        def _render(in_path: Path, prm: dict) -> Path:
            out = render_bg_replace(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, "bg_replace", params, _render, mask_sha=m_sha)
        mask_tmp = ctx.scratch_file("mask", ".png")
        a_img.save(mask_tmp, "PNG")
        _, mask_path = cache.put(f"mask:{m_sha}", mask_tmp)
        return {"status": "ok", "mask_sha256": m_sha, "mask_path": str(mask_path), **result}
    except Exception as e:  # noqa: BLE001
        log.exception("bg_replace(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
        if mask_tmp is not None:
            mask_tmp.unlink(missing_ok=True)


def bg_blur_op(image_id: int, strength: float = 0.6, max_blur: int = 31,
               focus: float = 0.9) -> dict[str, Any]:
    """bg_blur 工具实现（server.py 只做薄包装）。"""
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    mask_tmp: Path | None = None
    try:
        cur = data.get_current(conn, image_id)
        strength = float(strength)
        focus = float(focus)
        max_blur = int(max_blur)
        if not (0.0 < strength <= 1.0):
            raise ValueError(f"strength 须在 (0,1]，收到 {strength!r}")
        if not (0.0 < focus <= 1.0):
            raise ValueError(f"focus 须在 (0,1]，收到 {focus!r}")
        if not (3 <= max_blur <= 101):
            raise ValueError(f"max_blur 须在 3..101，收到 {max_blur!r}")
        max_blur = max_blur | 1  # 归一到奇数核
        params = {"strength": round(strength, 4), "max_blur": max_blur, "focus": round(focus, 4)}

        with Image.open(cur["output_path"]) as im:
            rgb = im.convert("RGB")
        t = blur_weight_map(rgb, params)
        t_img = Image.fromarray((t * 255.0).round().astype("uint8"))
        m_sha = png_sha256(t_img)

        ctx = get_context()
        ctx.payload["t"] = t

        def _render(in_path: Path, prm: dict) -> Path:
            out = render_bg_blur(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, "bg_blur", params, _render, mask_sha=m_sha)
        mask_tmp = ctx.scratch_file("mask", ".png")
        t_img.save(mask_tmp, "PNG")
        _, mask_path = cache.put(f"mask:{m_sha}", mask_tmp)
        return {"status": "ok", "mask_sha256": m_sha, "mask_path": str(mask_path), **result}
    except Exception as e:  # noqa: BLE001
        log.exception("bg_blur(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
        if mask_tmp is not None:
            mask_tmp.unlink(missing_ok=True)
