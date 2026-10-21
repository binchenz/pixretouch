"""常驻 MCP 服务入口（ADR-0003）。

stdio 传输；stdout 是协议信道，任何日志只走 stderr。
基于 mcp 2.x：from mcp.server import MCPServer（v1 的 mcp.server.fastmcp 已移除）。
工具随能力工单逐个挂载：doctor（工单01）、blemish_remove（工单03）、
assess_image / detect_faces / selection_report（工单04）、
cluster_faces / list_clusters / rename_cluster / merge_clusters / assign_face（工单05）、
cutout / bg_replace / bg_blur（工单08）、color_grade（工单09）、
super_resolve（工单10）。
挂载模式（后续能力照抄）：from .capabilities import <name> 触发注册副作用，
@mcp.tool() 只做薄包装，逻辑全部在能力模块的 <tool>_op() 里。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server import MCPServer  # mcp 2.x API（v1 的 mcp.server.fastmcp 已移除）

from . import doctor as doctor_mod
from .capabilities import assess as assess_cap  # 逻辑主体（工单 04：选片质检）
from .capabilities import background as background_cap  # 逻辑主体（工单 08：背景替换/虚化）
from .capabilities import clusters as clusters_cap  # 逻辑主体（工单 05：人物聚类与修正）
from .capabilities import color as color_cap  # 逻辑主体（工单 09：调色）
from .capabilities import cutout as cutout_cap  # 逻辑主体（工单 08：主体抠图）
from .capabilities import inpaint as inpaint_cap  # noqa: F401  能力注册副作用（工单 03）
from .capabilities import superres as superres_cap  # 逻辑主体（工单 10：超分增强）

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("pixretouch.server")

mcp = MCPServer("pixretouch")


@mcp.tool()
def doctor() -> dict[str, Any]:
    """环境健康自诊断：python/torch-MPS/核心库/磁盘余量/模型缓存/镜像源可达性，结构化报告。"""
    return doctor_mod.run_checks()


@mcp.tool()
def blemish_remove(
    image_id: int,
    points: list[list[int]] | None = None,
    boxes: list[list[int]] | None = None,
    brush: int = 10,
) -> dict[str, Any]:
    """污点修补（LaMa）：去除痘痘、路人、杂物；结果作为记录追加进记录链，可回滚。

    Args:
        image_id: 库内图像 ID（ingest 返回）。作用于当前态指针指向的产物。
        points: 提示点 [[x,y],...]，像素坐标。
        boxes: 提示框 [[x0,y0,x1,y1],...]。points/boxes 至少给一个。
        brush: 提示点半径（像素），1..64，默认 10。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit, mask_sha256}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return inpaint_cap.blemish_remove_op(image_id, points=points, boxes=boxes, brush=brush)


@mcp.tool()
def assess_image(image_id: int) -> dict[str, Any]:
    """选片质检（单图）：清晰度/曝光/闭眼(EAR)/人脸可检出四维度落库，人脸落 faces 表。

    分析型操作：不追加记录链、不改当前态指针；维度写入 images.meta['assess']。

    Args:
        image_id: 库内图像 ID（ingest 返回）。作用于当前态指针指向的产物。

    Returns:
        成功 {status:"ok", image_id, assess:{sharpness, exposure_mean, face_exposure_mean,
        face_count, max_face_side, eyes_closed, wink, faces:[{bbox, quality, ear_left,
        ear_right, eyes_closed, wink}], ...}}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return assess_cap.assess_image_op(image_id)


@mcp.tool()
def detect_faces(image_id: int) -> dict[str, Any]:
    """重检人脸+478 点 landmark 并覆盖写 faces 表（人物聚类前的刷新入口）。

    Args:
        image_id: 库内图像 ID。

    Returns:
        成功 {status:"ok", image_id, face_count, faces:[{bbox, quality, ear_left,
        ear_right, eyes_closed, wink}]}；失败 {status:"error", error}。
    """
    return assess_cap.detect_faces_op(image_id)


@mcp.tool()
def selection_report(image_ids: list[int] | None = None) -> dict[str, Any]:
    """相册级选片报告：每张 keep/review/reject + 可读理由，JSON 落 var/reports/。

    未质检的图自动补测（含人脸检出）；workflow（工单 13）三交付物之一。

    Args:
        image_ids: 参与报告的图像 ID 列表；None = 全库。

    Returns:
        成功 {status:"ok", generated_at, album:{total, keep, review, reject},
        images:[{image_id, sha256, verdict, score, reasons, assess}], report_path}；
        失败 {status:"error", error}。
    """
    return assess_cap.selection_report_op(image_ids)


@mcp.tool()
def cluster_faces(image_ids: list[int] | None = None) -> dict[str, Any]:
    """人物聚类：ArcFace 嵌入回填 + 余弦阈值增量聚类（确定性单遍；已分配的脸不动）。

    跨图同一人自动聚成同一人物聚类（按人寻址的寻址单位）；
    person_type（女/男/童）自动初判，可被 rename_cluster 覆盖。

    Args:
        image_ids: 参与本次聚类的图像 ID 子集；None = 全库未分配人脸。

    Returns:
        成功 {status:"ok", backfilled, assigned, created, threshold, clusters}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return clusters_cap.cluster_faces_op(image_ids)


@mcp.tool()
def list_clusters() -> dict[str, Any]:
    """列出人物聚类：id / name / person_type / default_suit_id / face_count / image_ids。

    Returns:
        成功 {status:"ok", clusters:[...]}；失败 {status:"error", error}。
    """
    return clusters_cap.list_clusters_op()


@mcp.tool()
def rename_cluster(cluster_id: int, name: str | None = None, person_type: str | None = None) -> dict[str, Any]:
    """命名聚类并覆盖 person_type（自动初判的人工覆盖入口；命名后按名字寻址）。

    Args:
        cluster_id: 目标聚类 ID。
        name: 聚类名（如「新娘」）；与 person_type 至少给一个。
        person_type: female / male / child / unknown。

    Returns:
        成功 {status:"ok", id, name, person_type, default_suit_id}；
        失败 {status:"error", error}。
    """
    return clusters_cap.rename_cluster_op(cluster_id, name=name, person_type=person_type)


@mcp.tool()
def merge_clusters(source_ids: list[int], target_id: int) -> dict[str, Any]:
    """合并聚类：把被拆散的同一人并入 target；faces 重指、源行删除、默认方案继承。

    Args:
        source_ids: 被合并的聚类 ID 列表。
        target_id: 保留的目标聚类 ID。

    Returns:
        成功 {status:"ok", target_id, merged_sources, faces_moved, default_suit_inherited}；
        失败 {status:"error", error}。
    """
    return clusters_cap.merge_clusters_op(source_ids, target_id)


@mcp.tool()
def assign_face(face_id: int, cluster_id: int | None = None) -> dict[str, Any]:
    """重指派单张人脸：把误归的脸移入正确聚类；cluster_id=None 移出聚类待重分配。

    Args:
        face_id: faces 表行 ID。
        cluster_id: 目标聚类 ID 或 None。

    Returns:
        成功 {status:"ok", face_id, cluster_id}；失败 {status:"error", error}。
    """
    return clusters_cap.assign_face_op(face_id, cluster_id)


@mcp.tool()
def cutout(image_id: int) -> dict[str, Any]:
    """主体抠图（ISNet）：产出 alpha 蒙版；记录产物=RGBA PNG，alpha 双登记为独立产物。

    Args:
        image_id: 库内图像 ID（ingest 返回）。作用于当前态指针指向的产物；
            当前态已是 RGBA（抠过）则直接复用其 alpha（链组合，零推理）。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, alpha_sha256, alpha_path}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return cutout_cap.cutout_op(image_id)


@mcp.tool()
def bg_replace(
    image_id: int,
    color: list[int] | None = None,
    background_path: str | None = None,
    feather: float = 2.0,
    despill: bool = True,
) -> dict[str, Any]:
    """背景替换：主体抠图与纯色或指定背景图合成（边缘羽化+溢色处理）；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。当前态为 RGBA（已抠图）时直接复用其 alpha。
        color: 纯色背景 [r,g,b]，0..255；与 background_path 二选一。
        background_path: 背景图路径（绝对或相对项目根）；内容 sha256 自动进缓存键。
        feather: 边缘羽化半径（像素），0..10，默认 2。
        despill: 溢色处理（premultiplied 腐蚀采样），默认 true。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=羽化后 alpha）；失败 {status:"error", error}。
    """
    return background_cap.bg_replace_op(image_id, color=color, background_path=background_path,
                                        feather=feather, despill=despill)


@mcp.tool()
def bg_blur(
    image_id: int,
    strength: float = 0.6,
    max_blur: int = 31,
    focus: float = 0.9,
) -> dict[str, Any]:
    """背景虚化：Depth-Anything-V2 深度驱动的自然虚化；主体（depth≥focus）清晰度不变。

    Args:
        image_id: 库内图像 ID。
        strength: 虚化强度 (0,1]，默认 0.6。
        max_blur: 最大模糊核直径 px（自动取奇数），3..101，默认 31。
        focus: 主体保护深度阈值 (0,1]，默认 0.9。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=虚化权重图 t，t=0 处恒清晰）；
        失败 {status:"error", error}。
    """
    return background_cap.bg_blur_op(image_id, strength=strength, max_blur=max_blur, focus=focus)


@mcp.tool()
def color_grade(
    image_id: int,
    temperature: int = 0,
    tint: int = 0,
    wb_auto: bool = False,
    exposure: float = 0.0,
    contrast: int = 0,
    lut: str | None = None,
    lut_path: str | None = None,
) -> dict[str, Any]:
    """调色：白平衡 / 影调 / cube 3D LUT 三组参数可独立或组合为一次调用；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。RGBA 链态只调 RGB 通道，alpha 直通。
        temperature: 白平衡色温 -100..100（+暖 −冷）。
        tint: 白平衡色调 -100..100（+品红 −绿）。
        wb_auto: 灰世界自动白平衡。
        exposure: 曝光档数 -2.0..2.0 EV。
        contrast: 对比 -100..100（软 S 曲线，端点不动）。
        lut: 内置 LUT 名（一期：'bw-classic' 经典黑白基准配方）。
        lut_path: 外部 .cube 文件路径（与 lut 二选一；内容 sha256 自动进缓存键）。
        至少需要一项调整。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return color_cap.color_grade_op(
        image_id, temperature=temperature, tint=tint, wb_auto=wb_auto,
        exposure=exposure, contrast=contrast, lut=lut, lut_path=lut_path,
    )


@mcp.tool()
def super_resolve(image_id: int, scale: int = 4) -> dict[str, Any]:
    """超分增强（Real-ESRGAN）：×2/×4 放大，大图自动分块防 OOM、块缝不可见；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID（ingest 返回）。
        scale: 放大倍率，2 或 4（原生双模型）。
        RGBA 链态：RGB 走模型、alpha 以 lanczos 同步放大（纪律 8 几何增补）。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit, width, height}；
        失败 {status:"error", error}（结构化错误，不抛穿）。
        守卫：输出边长 >65500（JPEG 上限）或画布 >8GB → 结构化拒绝。
    """
    return superres_cap.super_resolve_op(image_id, scale=scale)


def main() -> None:
    log.info("pixretouch server starting (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
