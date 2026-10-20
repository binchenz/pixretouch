"""调色（白平衡 / 影调 / cube 3D LUT）— 工单 09。

三组参数可独立或组合为一次 color_grade 调用，管线顺序固定（业界惯例：先校正后创作）：
    白平衡（temperature/tint 通道增益，或 wb_auto 灰世界）
  → 影调（exposure 曝光档数 + contrast 软 S 曲线）
  → 3D LUT（.cube，三线性插值）。
计算在线性光域（sRGB 近似 γ=2.2）进行，编码域出 LUT，减少色带。本单无模型权重
（纯数学管线），零新增依赖（numpy/cv2 既有）。

bw-classic：本单**自研**经典黑白配方（ADR-0001：不搬运第三方 LUT 资产），
以版本化 .cube 文件固化（src/pixretouch/luts/bw-classic.cube，33³ 文本资产，
由本文件 generate_bw_classic_cube() 确定性再生——配方常数即文档）。#12 预设方案
引擎将其打包为种子方案。配方：Rec601 luma(0.299/0.587/0.114) → 软 S 对比
（中心斜率 1.18，端点不动）→ 0.98 gamma 趾部。

LUT 素材规则（lead 裁决 2026-10-21，模式 docstring 纪律 7）：无论内置名还是外部
路径，一律以 **文件内容 sha256** 记入 params 并以 cache.put("material:"+sha) 固化进
内容寻址存储，渲染经 ctx.cache.path_for(sha) 按 sha 取回，**路径不进键**——
配方内容变 ⇒ 键变不假命中；同内容换路径 ⇒ 键不变必命中。
RGBA 链态策略（纪律 8）：只调 RGB 通道、alpha 直通，产物保持 RGBA（PNG 无损）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import config, data
from ..engine import EngineContext, get_context, get_session
from . import capability

log = logging.getLogger("pixretouch.capabilities.color")

CAPABILITY_NAME = "color_grade"
GAMMA = 2.2
LUT_DIR = config.PKG_ROOT / "luts"

# ---------------------------------------------------------------- bw-classic 自研配方

BW_CLASSIC_NAME = "bw-classic"
BW_CLASSIC_SIZE = 33
BW_LUMA = (0.299, 0.587, 0.114)   # Rec601 经典 luma 权重
BW_CONTRAST_K = 0.18              # 软 S 中心斜率 = 1+K = 1.18，端点保持不动
BW_GAMMA_TOE = 0.98               # 轻微趾部提亮（经典黑调纸感）


def _bw_curve(v: Any) -> Any:
    """经典黑白影调曲线：软 S 对比 + gamma 趾部（numpy 广播，确定性）。"""
    import numpy as np

    d = v - 0.5
    v = 0.5 + d * (1.0 + BW_CONTRAST_K * (1.0 - (2.0 * d) ** 2))
    return np.clip(v, 0.0, 1.0) ** BW_GAMMA_TOE


def generate_bw_classic_cube(dest: Path) -> Path:
    """确定性再生 bw-classic.cube（.cube 标准：red 变化最快 → reshape [b][g][r]）。"""
    import numpy as np

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = BW_CLASSIC_SIZE
    axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
    # 网格顺序：b 慢、g 中、r 快（与 .cube 行序一致）
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")  # [b,g,r]
    lum = BW_LUMA[0] * r + BW_LUMA[1] * g + BW_LUMA[2] * b
    out = _bw_curve(lum)
    lines = [
        "# bw-classic - PixRetouch original classic B&W recipe (ticket 09)",
        f"# Rec601 luma{BW_LUMA} + soft-S contrast (center slope {1 + BW_CONTRAST_K:.2f}) + gamma toe {BW_GAMMA_TOE}",
        "# Regenerate: pixretouch.capabilities.color.generate_bw_classic_cube(<dest>)",
        f'TITLE "{BW_CLASSIC_NAME}"',
        f"LUT_3D_SIZE {n}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    flat = out.reshape(-1)
    with open(dest, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        # 灰值 triplet：三通道同值，%.6f 定点保证字节确定
        f.writelines(f"{v:.6f} {v:.6f} {v:.6f}\n" for v in flat)
    log.info("bw-classic.cube 已生成：%s", dest)
    return dest


def builtin_lut_path(name: str) -> Path:
    """内置 LUT 名 → 仓库资产路径；bw-classic 缺失时确定性再生（字节一致）。"""
    if name != BW_CLASSIC_NAME:
        raise ValueError(f"未知内置 LUT：{name!r}（可用：{BW_CLASSIC_NAME}）")
    p = LUT_DIR / f"{BW_CLASSIC_NAME}.cube"
    if not p.exists():
        generate_bw_classic_cube(p)
    return p


# ---------------------------------------------------------------- .cube 解析

def parse_cube(path: Path) -> tuple[Any, Any, Any]:
    """解析 .cube 3D LUT → (lut (N,N,N,3) 索引 [b][g][r], domain_min, domain_max)。

    .cube 标准：LUT_3D_SIZE N；可选 DOMAIN_MIN/DOMAIN_MAX；随后 N³ 行 "r g b"，
    red 变化最快。格式错误一律 ValueError（结构化错误返回给 agent）。
    """
    import numpy as np

    n: int | None = None
    dmin = np.zeros(3, dtype=np.float32)
    dmax = np.ones(3, dtype=np.float32)
    values: list[list[float]] = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        head = parts[0].upper()
        if head in ("TITLE",):
            continue
        if head == "LUT_3D_SIZE":
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(f"{path}:{ln} LUT_3D_SIZE 非法：{raw!r}")
            n = int(parts[1])
            continue
        if head in ("DOMAIN_MIN", "DOMAIN_MAX"):
            if len(parts) != 4:
                raise ValueError(f"{path}:{ln} {head} 需 3 个分量：{raw!r}")
            vals = np.array([float(x) for x in parts[1:4]], dtype=np.float32)
            if head == "DOMAIN_MIN":
                dmin = vals
            else:
                dmax = vals
            continue
        if len(parts) >= 3:
            try:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError as e:
                raise ValueError(f"{path}:{ln} 数据行非法：{raw!r}") from e
            continue
        raise ValueError(f"{path}:{ln} 无法识别的行：{raw!r}")

    if n is None:
        raise ValueError(f"{path}: 缺少 LUT_3D_SIZE")
    if len(values) != n**3:
        raise ValueError(f"{path}: 数据行数 {len(values)} ≠ N³={n**3}")
    lut = np.asarray(values, dtype=np.float32).reshape(n, n, n, 3)  # [b][g][r]
    if float(dmax.min() - dmin.max()) <= 0 and not bool((dmax > dmin).all()):
        raise ValueError(f"{path}: DOMAIN 非法 min={dmin} max={dmax}")
    return lut, dmin, dmax


def apply_lut3d(img: Any, lut: Any, dmin: Any, dmax: Any, chunk: int = 2048) -> Any:
    """三线性插值应用 3D LUT。img: float32 (h,w,3) ∈[0,1]（编码域）。分块防大图画内存峰值。"""
    import numpy as np

    n = lut.shape[0]
    x = np.clip(img, 0.0, 1.0)
    if dmin is not None:
        x = (x - dmin) / np.maximum(dmax - dmin, 1e-6)
        x = np.clip(x, 0.0, 1.0)
    out = np.empty_like(x)
    h = x.shape[0]
    for y0 in range(0, h, chunk):
        blk = x[y0:y0 + chunk] * (n - 1)
        r, g, b = blk[..., 0], blk[..., 1], blk[..., 2]
        r0 = np.floor(r).astype(np.int32); g0 = np.floor(g).astype(np.int32); b0 = np.floor(b).astype(np.int32)
        r1 = np.minimum(r0 + 1, n - 1); g1 = np.minimum(g0 + 1, n - 1); b1 = np.minimum(b0 + 1, n - 1)
        fr = (r - r0)[..., None]; fg = (g - g0)[..., None]; fb = (b - b0)[..., None]
        # 8 角采样（lut 索引 [b][g][r]）
        c000 = lut[b0, g0, r0]; c100 = lut[b0, g0, r1]
        c010 = lut[b0, g1, r0]; c110 = lut[b0, g1, r1]
        c001 = lut[b1, g0, r0]; c101 = lut[b1, g0, r1]
        c011 = lut[b1, g1, r0]; c111 = lut[b1, g1, r1]
        c00 = c000 * (1 - fr) + c100 * fr
        c01 = c001 * (1 - fr) + c101 * fr
        c10 = c010 * (1 - fr) + c110 * fr
        c11 = c011 * (1 - fr) + c111 * fr
        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg
        out[y0:y0 + chunk] = c0 * (1 - fb) + c1 * fb
    return out


# ---------------------------------------------------------------- 调色管线

def apply_grade(rgb_u8: Any, params: dict, ctx: EngineContext | None = None) -> Any:
    """uint8 RGB → 调色 → uint8 RGB。线性域做白平衡/曝光，编码域做对比/LUT。"""
    import numpy as np

    img = rgb_u8.astype(np.float32) / 255.0
    lin = np.power(np.clip(img, 0.0, 1.0), GAMMA)

    # ① 白平衡
    temperature = int(params.get("temperature", 0))
    tint = int(params.get("tint", 0))
    if params.get("wb_auto", False):
        means = lin.reshape(-1, 3).mean(axis=0)
        ref = float(means.mean())
        gains = np.where(means > 1e-6, ref / np.maximum(means, 1e-6), 1.0).astype(np.float32)
        lin = lin * gains
    if temperature or tint:
        g_r = np.float32(1.0 + 0.0025 * temperature)
        g_b = np.float32(1.0 - 0.0025 * temperature)
        g_g = np.float32(1.0 - 0.0020 * tint)
        lin = lin * np.array([g_r, g_g, g_b], dtype=np.float32)

    # ② 影调：曝光（线性域，档数）
    ev = float(params.get("exposure", 0.0))
    if ev:
        lin = lin * np.float32(2.0 ** ev)

    # 回编码域
    enc = np.power(np.clip(lin, 0.0, 1.0), 1.0 / GAMMA)

    # ③ 影调：对比（软 S 曲线，端点不动不裁剪）
    contrast = int(params.get("contrast", 0))
    if contrast:
        k = 1.5 * (contrast / 100.0)
        d = enc - 0.5
        enc = np.clip(0.5 + d * (1.0 + k * (1.0 - (2.0 * d) ** 2)), 0.0, 1.0)

    # ④ 3D LUT（编码域）——素材规则：payload 省算快路径，正规取用 = ctx.cache 按内容 sha
    lut_ref = params.get("lut")
    if lut_ref:
        packed = ctx.payload.get("lut") if ctx is not None else None
        if packed is None:
            if ctx is None or ctx.cache is None:
                raise FileNotFoundError("LUT 素材取用需 ctx.cache（素材规则，模式 docstring 纪律 7）")
            packed = parse_cube(ctx.cache.path_for(lut_ref["sha256"]))
        lut, dmin, dmax = packed
        enc = np.clip(apply_lut3d(enc, lut, dmin, dmax), 0.0, 1.0)

    return (enc * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- 能力注册

@capability(
    CAPABILITY_NAME,
    params_schema={
        "temperature": "白平衡色温 -100..100（+暖 −冷），默认 0",
        "tint": "白平衡色调 -100..100（+品红 −绿），默认 0",
        "wb_auto": "灰世界自动白平衡，默认 false",
        "exposure": "曝光档数 -2.0..2.0 EV，默认 0",
        "contrast": "对比 -100..100（软 S 曲线），默认 0",
        "lut": "内置 LUT 名（一期：'bw-classic'）",
        "lut_path": "外部 .cube 路径（与 lut 二选一）",
    },
    defaults={
        "temperature": 0, "tint": 0, "wb_auto": False,
        "exposure": 0.0, "contrast": 0, "lut": None, "lut_path": None,
    },
    weight=None,  # 纯数学管线，无模型权重
)
def render(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：当前态 → 调色 → 确定性产物（RGB→JPEG q95/4:4:4/无 EXIF；RGBA→PNG 无损）。

    RGBA 链态（cutout 产物）：只调 RGB 通道、alpha 直通（纪律 8，保链可组合）。
    """
    from PIL import Image

    with Image.open(image) as im:
        alpha = im.getchannel("A") if im.mode == "RGBA" else None
        rgb = im.convert("RGB")
    import numpy as np

    out = apply_grade(np.asarray(rgb), params, ctx)
    img_out = Image.fromarray(out)
    if alpha is not None:
        img_out.putalpha(alpha)
        dst = ctx.scratch_file("color_grade", ".png")
        img_out.save(dst, "PNG")
    else:
        dst = ctx.scratch_file("color_grade", ".jpg")
        img_out.save(dst, "JPEG", quality=95, subsampling=0)
    return dst


# ---------------------------------------------------------------- 工具实现

def color_grade_op(
    image_id: int,
    temperature: int = 0,
    tint: int = 0,
    wb_auto: bool = False,
    exposure: float = 0.0,
    contrast: int = 0,
    lut: str | None = None,
    lut_path: str | None = None,
) -> dict[str, Any]:
    """color_grade 工具实现（server.py 只做 @mcp.tool() 薄包装）。"""
    conn, cache = get_session()
    tmp_out: list[Path] = []
    try:
        data.get_current(conn, image_id)  # 图像存在性（KeyError → 结构化错误）

        # ---- 校验与规范化
        temperature, tint, contrast = int(temperature), int(tint), int(contrast)
        exposure = round(float(exposure), 4)
        if not -100 <= temperature <= 100:
            raise ValueError(f"temperature 须在 -100..100，收到 {temperature}")
        if not -100 <= tint <= 100:
            raise ValueError(f"tint 须在 -100..100，收到 {tint}")
        if not -2.0 <= exposure <= 2.0:
            raise ValueError(f"exposure 须在 -2.0..2.0 EV，收到 {exposure}")
        if not -100 <= contrast <= 100:
            raise ValueError(f"contrast 须在 -100..100，收到 {contrast}")
        if (lut is not None) and (lut_path is not None):
            raise ValueError("lut（内置名）与 lut_path（外部文件）二选一")

        lut_ref: dict[str, Any] | None = None
        lut_src: Path | None = None
        if lut is not None:
            lut_src = builtin_lut_path(str(lut))
            lut_ref = {"name": str(lut), "sha256": data.file_sha256(lut_src)}
        elif lut_path is not None:
            p = Path(str(lut_path)).expanduser()
            if not p.is_absolute():
                p = config.PROJECT_HOME / p
            if not p.exists():
                raise ValueError(f"LUT 文件不存在：{p}")
            lut_src = p
            lut_ref = {"name": None, "sha256": data.file_sha256(p)}  # 路径不进键（素材规则）
        packed_lut = parse_cube(lut_src) if lut_src is not None else None  # fail-fast：格式错误立即结构化返回

        active = (temperature != 0 or tint != 0 or bool(wb_auto)
                  or exposure != 0.0 or contrast != 0 or lut_ref is not None)
        if not active:
            raise ValueError("至少需要一项调整（temperature/tint/wb_auto/exposure/contrast/lut）")

        params: dict[str, Any] = {
            "temperature": temperature, "tint": tint, "wb_auto": bool(wb_auto),
            "exposure": exposure, "contrast": contrast, "lut": lut_ref,
        }
        if lut_src is not None:
            cache.put(f"material:{lut_ref['sha256']}", lut_src)  # 素材固化进内容寻址存储
        ctx = get_context()
        ctx.cache = cache  # 素材正规取用入口：path_for(lut sha)
        if packed_lut is not None:
            ctx.payload["lut"] = packed_lut  # 省算快路径（重建走 cache 按 sha）

        def _render(in_path: Path, prm: dict) -> Path:
            out = render(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, CAPABILITY_NAME, params, _render)
        return {"status": "ok", **result}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约）
        log.exception("color_grade(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
