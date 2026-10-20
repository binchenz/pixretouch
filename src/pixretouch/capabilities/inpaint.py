"""污点修补（LaMa）— 工单 03，「能力 → 记录链」的 tracer bullet。

能力：给出库内图像与点/框提示，去除痘痘、路人、杂物（LaMa 大掩膜修补）；
结果作为记录追加进记录链，回滚 = 移动当前态指针。

== 标准挂接模式（后续能力工单复制此清单）==
1. 能力模块 capabilities/<name>.py（本文件即范本）：
   - weight("<权重名>", {"repo", "file", "kind": "file"|"repo", "modelscope": <兜底 id 或 None>})
     登记权重规格 → prefetch 经 hf-mirror 下载到 models/<权重名>/（铁律 6）；
   - @capability("<能力名>", params_schema=..., defaults=...) 装饰渲染函数
     fn(image: Path, params: dict, ctx: EngineContext) -> Path（临时产物路径）。
     渲染函数纪律：只读输入；不碰库；重依赖（torch/numpy/cv2/PIL）函数内惰性导入；
     模型懒加载后模块级常驻（ADR-0003）；严禁 print（stdout = MCP 协议信道），日志走
     logging→stderr；输出编码必须确定性（同输入+同参数 ⇒ 字节一致），否则缓存断言失效。
2. server.py 最小增量（mcp 2.x：from mcp.server import MCPServer；v1 的
   mcp.server.fastmcp 已移除。@mcp.tool() 仍返回原函数 → 测试可直接调用工具函数）：
   from .capabilities import <name>（触发注册副作用）+ @mcp.tool() 薄包装，
   全部逻辑委托 <name>.<tool>_op() —— 热文件只留挂载面，冲突面最小。
   DSH 注册事实（lead 2026-10-21）：DSH 以 StdioClientTransport spawn
   `.venv/bin/python -m pixretouch.server`（cwd=仓库根）；env = 父环境剔除
   DSH_* 与 /KEY|PASSWORD|SECRET|TOKEN/i 变量 + 显式注入 PIXRETOUCH_HOME=仓库根、
   HF_ENDPOINT=hf-mirror → server 启动路径不得依赖任何被清洗变量；DSH 侧工具名为
   mcp__pixretouch__<tool>（skill 工单按此名引用）；toolCallTimeoutMs=600s →
   模型必须保持懒加载（首调冷加载 5~15s 在预算内），严禁 server 启动时预载权重。
3. 工具实现 <tool>_op()：校验并规范化 params →（可选）构建 mask 算 mask_sha256 →
   engine.get_session() → data.apply(conn, cache, image_id, "<能力名>", params,
   render 闭包, mask_sha=...) → {"status":"ok", record_id, output_sha256, output_path,
   cache_hit}；失败捕获后返回 {"status":"error","error":...}（结构化错误，不抛穿 MCP）；
   finally 清理临时产物（缓存中已有固化副本）。
4. mask 与缓存键：mask 必须由 params 确定性派生；其 PNG 字节 sha256 作为 mask_sha
   进缓存键并落 records.mask_sha256。预构建对象经 ctx.payload 传入渲染函数——payload
   不得影响确定性（仅凭 params 重建结果一致）。mask 源于模型的能力（如皮肤磨皮）
   在渲染 mask 后同样算 sha 传入。
   mask 通道语义（工单 08 补充，lead 裁决）：mask 通道承载「渲染实际消费的权重/
   蒙版图」（blemish=提示 mask、cutout=alpha、bg_blur=虚化权重图 t）。params 可
   确定性派生的 mask 仅记 sha；模型派生的 mask（alpha/深度类）另做**双登记**——
   cache.put("mask:"+sha) 固化内容寻址文件 + 工具结果返回 mask_path/alpha_path，
   且记录 output 保持图像状态（如 cutout=RGBA PNG）保证链可组合；外部素材引用
   （如 background_path）以其内容 sha256 记入 params 参与缓存键。#12 预设方案
   引擎按此语义消费。
   **缓存键是链位置敏感的**：key = (当前记录的输出哈希, 能力, 规范化参数, mask_sha)，
   ADR-0002 设计使然。冒烟测试必守三条推论：
   a) 测「同参重调 cache_hit=true」的正确姿势 = 先 rollback_to 回同一链态，再同参
      apply → 键复现 → 命中；
   b) 不回滚的连续同参 apply 必然 miss（当前输出哈希已变）——不是 bug，测试别写反；
   c) 同参换 mask 必 miss 且 cache_index 行数 +1 —— 可作 mask 语义的正向断言。
5. tests/test_capability_<name>.py：走工具函数入口（主接缝）；tmp_path 隔离库与缓存、
   MODELS_DIR 指向仓库 models/（权重跨测试共享）；权重缺失 skipif；属性式断言
   （尺寸一致/目标区域变化/非目标区域近似不变/记录链+1/回滚恢复/缓存命中），
   不绑定模型输出字节。
"""
from __future__ import annotations

import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import Any

from .. import config, data
from ..engine import EngineContext, get_context, get_session
from . import WEIGHT_SPECS, capability, weight

log = logging.getLogger("pixretouch.capabilities.inpaint")

WEIGHT_NAME = "lama"
CAPABILITY_NAME = "blemish_remove"

# 权重源选型（2026-10-21 curl 勘察，全部经 hf-mirror 验证可达）：
# - okaris/simple-lama「big-lama.pt」：torchscript 单文件（zip 魔数已验），apache-2.0，
#   205,803,670 B，torch.jit.load 即推理，零新增依赖 → 选定；
# - smartywu/big-lama：弃。zip 内为 lightning best.ckpt + config.yaml（官方 release 原样），
#   加载需 vendor 原版 LaMa 架构代码，成本高；
# - simple-lama-inpainting pip 包：弃。依赖钉死 numpy<2.0/opencv<5.0/pillow<10，与共享
#   venv（numpy 2.4/opencv 5.0/pillow 13）硬冲突；且其权重从 GitHub release 下载，
#   违反供给链决策（权重必须 hf-mirror 优先）。本文件的前/后处理约定参考其开源实现
#   （MIT，源头为 advimman/lama）白盒重写，不引入该依赖；
# - camenduru/big-lama：弃，同为 lightning ckpt。
# modelscope 兜底 iic/cv_fft_inpainting_lama：注意其权重为 ModelScope pipeline 格式
# 而非 torchscript，真走兜底需转换或改走 modelscope pipeline（届时在此登记结论）。
weight(WEIGHT_NAME, {
    "repo": "okaris/simple-lama",
    "file": "big-lama.pt",
    "kind": "file",
    "modelscope": "iic/cv_fft_inpainting_lama",
})


# ---------------------------------------------------------------- mask

def _normalize(size: tuple[int, int], points: Any, boxes: Any, brush: Any) -> dict[str, Any]:
    """校验并规范化提示 → 规范形 params（缓存键要求稳定规范形）。"""
    w, h = size
    brush = int(brush)
    if not 1 <= brush <= 64:
        raise ValueError(f"brush 须在 1..64，收到 {brush}")
    pts: list[list[int]] = []
    for p in points or []:
        if len(p) != 2:
            raise ValueError(f"提示点须为 [x,y]，收到 {p!r}")
        x, y = int(p[0]), int(p[1])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"提示点 ({x},{y}) 越界，图像为 {w}x{h}")
        pts.append([x, y])
    bxs: list[list[int]] = []
    for b in boxes or []:
        if len(b) != 4:
            raise ValueError(f"提示框须为 [x0,y0,x1,y1]，收到 {b!r}")
        x0, y0, x1, y1 = (int(v) for v in b)
        if not (0 <= x0 < x1 < w and 0 <= y0 < y1 < h):
            raise ValueError(f"提示框 ({x0},{y0},{x1},{y1}) 非法或越界，图像为 {w}x{h}")
        bxs.append([x0, y0, x1, y1])
    if not pts and not bxs:
        raise ValueError("points 与 boxes 不能同时为空")
    return {"points": pts, "boxes": bxs, "brush": brush}


def build_mask(size: tuple[int, int], params: dict) -> Any:
    """params → mask（PIL "L"）确定性派生：点→半径 brush 圆、框→矩形，再外扩膨胀。

    膨胀给 LaMa 留出污点周边的过渡上下文（原仓库惯例），核尺寸随 brush 联动。
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    w, h = size
    brush = int(params.get("brush", 10))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x, y in params.get("points") or []:
        draw.ellipse([x - brush, y - brush, x + brush, y + brush], fill=255)
    for x0, y0, x1, y1 in params.get("boxes") or []:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    arr = np.asarray(mask)
    k = max(3, (brush // 2) | 1)
    arr = cv2.dilate(arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return Image.fromarray(arr)


def mask_sha256(mask: Any) -> str:
    """mask 的 PNG 字节 sha256（PIL PNG 编码无时间戳，同 mask 字节确定）。"""
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------- LaMa 推理

_MODEL: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def weight_path() -> Path:
    """models/lama/big-lama.pt（文件名取自 WEIGHT_SPECS，与 prefetch 落位一致）。"""
    spec = WEIGHT_SPECS[WEIGHT_NAME]
    return config.MODELS_DIR / WEIGHT_NAME / Path(spec["file"]).name


def _get_model() -> tuple[Any, str]:
    """懒加载 + 常驻（ADR-0003）。MPS 优先；冒烟前向失败则降级 CPU 并记录。

    LaMa 的 FFC 块含 FFT 算子，MPS 支持随 torch 版本演进，故加载后立即用
    64x64 全零输入做一次冒烟前向探测，而不是等真实调用半途失败。
    """
    if "model" in _MODEL:
        return _MODEL["model"], _MODEL["device"]
    with _MODEL_LOCK:
        if "model" in _MODEL:
            return _MODEL["model"], _MODEL["device"]
        import torch

        pt = weight_path()
        if not pt.exists():
            raise FileNotFoundError(
                f"LaMa 权重缺失：{pt}。先执行："
                f"HF_ENDPOINT={config.HF_MIRROR} python -m pixretouch.prefetch"
            )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = torch.jit.load(str(pt), map_location=device).eval()
        if device == "mps":
            try:
                with torch.inference_mode():
                    model(
                        torch.zeros(1, 3, 64, 64, device=device),
                        torch.zeros(1, 1, 64, 64, device=device),
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("LaMa MPS 冒烟前向失败（%s: %s），降级 CPU", type(e).__name__, e)
                device = "cpu"
                model = torch.jit.load(str(pt), map_location="cpu").eval()
        _MODEL["model"], _MODEL["device"] = model, device
        log.info("LaMa 常驻：device=%s path=%s", device, pt)
        return model, device


def lama_inpaint(image_rgb: Any, mask_l: Any) -> Any:
    """image_rgb: PIL RGB；mask_l: PIL "L"（>0 视为待修补）。返回同尺寸 PIL RGB。

    前/后处理遵循 LaMa 原版推理约定（参考 MIT 开源 simple-lama-inpainting 白盒重写）：
    [0,1] float32、mask 二值化、symmetric pad 到 8 的倍数（编码器 8x 下采样）、
    推理后裁回原尺寸。输出仅 mask 内采用网络结果、mask 外保留原像素——
    全图质量损失最小化，且「蒙版外近似不变」成为可断言属性。
    """
    import numpy as np
    import torch
    from PIL import Image

    model, device = _get_model()
    w, h = image_rgb.size

    img = np.transpose(np.asarray(image_rgb, dtype=np.float32), (2, 0, 1)) / 255.0
    msk = np.asarray(mask_l, dtype=np.float32)
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    msk = np.pad(msk, ((0, ph), (0, pw)), mode="symmetric")

    img_t = torch.from_numpy(img).unsqueeze(0).to(device)
    msk_t = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(device)
    msk_t = (msk_t > 0) * 1

    with torch.inference_mode():
        out = model(img_t, msk_t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    res = out[0].permute(1, 2, 0).detach().cpu().numpy()
    res = np.clip(res * 255, 0, 255).astype(np.uint8)[:h, :w]
    return Image.composite(Image.fromarray(res), image_rgb, mask_l)


# ---------------------------------------------------------------- 能力注册

@capability(
    CAPABILITY_NAME,
    params_schema={
        "points": "提示点 [[x,y],...]，像素坐标（可选）",
        "boxes": "提示框 [[x0,y0,x1,y1],...]（可选）；points/boxes 至少其一",
        "brush": "提示点半径（像素），1..64，默认 10",
    },
    defaults={"brush": 10},
    weight=WEIGHT_NAME,
)
def render(image: Path, params: dict, ctx: EngineContext) -> Path:
    """渲染函数：只读输入 → LaMa 修补 → 确定性 JPEG 临时产物（quality=95, 4:4:4, 无 EXIF）。"""
    from PIL import Image

    with Image.open(image) as im:
        rgb = im.convert("RGB")
    mask = ctx.payload.get("mask") or build_mask(rgb.size, params)
    out_img = lama_inpaint(rgb, mask)
    out = ctx.scratch_file("blemish", ".jpg")
    out_img.save(out, "JPEG", quality=95, subsampling=0)
    return out


# ---------------------------------------------------------------- 工具实现

def blemish_remove_op(
    image_id: int,
    points: list | None = None,
    boxes: list | None = None,
    brush: int = 10,
) -> dict[str, Any]:
    """blemish_remove 工具实现（server.py 只做 @mcp.tool() 薄包装）。"""
    from PIL import Image

    conn, cache = get_session()
    tmp_out: list[Path] = []
    try:
        cur = data.get_current(conn, image_id)  # 图像不存在/无当前态 → KeyError
        with Image.open(cur["output_path"]) as im:
            size = im.size
        params = _normalize(size, points, boxes, brush)
        mask = build_mask(size, params)
        m_sha = mask_sha256(mask)
        ctx = get_context()
        ctx.payload["mask"] = mask  # 预构建传入渲染函数；仅凭 params 重建结果一致

        def _render(in_path: Path, prm: dict) -> Path:
            out = render(in_path, prm, ctx)
            tmp_out.append(out)
            return out

        result = data.apply(conn, cache, image_id, CAPABILITY_NAME, params, _render, mask_sha=m_sha)
        return {"status": "ok", "mask_sha256": m_sha, **result}
    except Exception as e:  # noqa: BLE001  结构化错误返回（Spec·工具契约），完整栈走 stderr
        log.exception("blemish_remove(image_id=%s) 失败", image_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in tmp_out:
            p.unlink(missing_ok=True)
