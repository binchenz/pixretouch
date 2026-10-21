"""常驻 MCP 服务入口（ADR-0003）。

stdio 传输；stdout 是协议信道，任何日志只走 stderr。
基于 mcp 2.x：from mcp.server import MCPServer（v1 的 mcp.server.fastmcp 已移除）。
工具随能力工单逐个挂载：doctor（工单01）、blemish_remove（工单03）、
assess_image / detect_faces / selection_report（工单04）、
cluster_faces / list_clusters / rename_cluster / merge_clusters / assign_face（工单05）、
smooth_skin / whiten_teeth / brighten_eyes（工单06）、face_enhance（工单07）、
cutout / bg_replace / bg_blur（工单08）、color_grade（工单09）、
super_resolve（工单10）、apply_preset / list_presets / save_preset /
bind_cluster_suit（工单11）、ingest / ingest_album / set_verdicts /
contact_sheet / export_album（工单13 提前落）、rollback_to / history（工单12 质检回路补挂）。
挂载模式（后续能力照抄）：from .capabilities import <name> 触发注册副作用，
@mcp.tool() 只做薄包装，逻辑全部在能力模块的 <tool>_op() 里。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

from mcp.server import MCPServer  # mcp 2.x API（v1 的 mcp.server.fastmcp 已移除）

from . import album as album_mod  # 逻辑主体（工单 13：相册级 verdict/联络表/导出，非能力）
from . import chainops as chainops_mod  # 逻辑主体（记录链查询/回退，工单 12 质检回路补挂）
from . import doctor as doctor_mod
from . import intake as intake_mod  # 逻辑主体（工单 13 首步：入库工具，非能力）
from . import presets as presets_mod  # 逻辑主体（工单 11：预设方案库）
from .capabilities import assess as assess_cap  # 逻辑主体（工单 04：选片质检）
from .capabilities import background as background_cap  # 逻辑主体（工单 08：背景替换/虚化）
from .capabilities import clusters as clusters_cap  # 逻辑主体（工单 05：人物聚类与修正）
from .capabilities import color as color_cap  # 逻辑主体（工单 09：调色）
from .capabilities import cutout as cutout_cap  # 逻辑主体（工单 08：主体抠图）
from .capabilities import face_enhance as face_enhance_cap  # 逻辑主体（工单 07：人脸增强）
from .capabilities import inpaint as inpaint_cap  # noqa: F401  能力注册副作用（工单 03）
from .capabilities import preset_engine as preset_cap  # 逻辑主体（工单 11：预设方案事务化编排）
from .capabilities import skin as skin_cap  # 逻辑主体（工单 06：磨皮与牙眼）
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
def smooth_skin(image_id: int, cluster_id: int | None = None, strength: float = 0.6) -> dict[str, Any]:
    """皮肤磨皮（频率分离，保留纹理）：皮肤区域低频均匀化、高频纹理原样保留；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。RGBA 链态只调 RGB、alpha 直通（纪律 8）。
        cluster_id: 人物聚类 ID（按人寻址）；None = 作用于全图检出人脸。
        strength: 磨皮强度 (0,1]，默认 0.6。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=皮肤权重图）；失败 {status:"error", error}。
    """
    return skin_cap.smooth_skin_op(image_id, cluster_id=cluster_id, strength=strength)


@mcp.tool()
def whiten_teeth(image_id: int, cluster_id: int | None = None, amount: float = 0.5) -> dict[str, Any]:
    """牙齿美白：牙部 mask（唇内缩∩亮低饱和像素）内 HSV 提亮去黄，有上限防假牙感；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。RGBA 链态只调 RGB、alpha 直通（纪律 8）。
        cluster_id: 人物聚类 ID（按人寻址）；None = 作用于全图检出人脸。
        amount: 美白强度 (0,1]，默认 0.5。闭嘴照片 mask 近空 → 近恒等输出（记录照常追加）。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=牙部权重图）；失败 {status:"error", error}。
    """
    return skin_cap.whiten_teeth_op(image_id, cluster_id=cluster_id, amount=amount)


@mcp.tool()
def brighten_eyes(image_id: int, cluster_id: int | None = None, amount: float = 0.35) -> dict[str, Any]:
    """亮眼：眼部 mask 内暗像素优先提亮（巩膜增白、虹膜瞳孔温和），有上限；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。RGBA 链态只调 RGB、alpha 直通（纪律 8）。
        cluster_id: 人物聚类 ID（按人寻址）；None = 作用于全图检出人脸。
        amount: 亮眼强度 (0,1]，默认 0.35。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=眼部权重图）；失败 {status:"error", error}。
    """
    return skin_cap.brighten_eyes_op(image_id, cluster_id=cluster_id, amount=amount)


@mcp.tool()
def face_enhance(image_id: int, cluster_id: int | None = None, strength: float = 0.3) -> dict[str, Any]:
    """人脸增强（CodeFormer）：低清/模糊/老照片人脸修复，输出按强度与原图混合——默认 0.3 保守混合杜绝换脸感；追加记录链，可回滚。

    Args:
        image_id: 库内图像 ID。RGBA 链态只调 RGB、alpha 直通（纪律 8）。
        cluster_id: 人物聚类 ID（按人寻址）；None = 作用于全图检出人脸。
        strength: 混合强度 (0,1]，默认 0.3（保守上限；越高越接近模型全量输出）。

    Returns:
        成功 {status:"ok", record_id, output_sha256, output_path, cache_hit,
        mask_sha256, mask_path}（mask=脸部椭圆羽化权重图）；
        失败 {status:"error", error}（结构化错误，不抛穿）。
    """
    return face_enhance_cap.face_enhance_op(image_id, cluster_id=cluster_id, strength=strength)


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


@mcp.tool()
def apply_preset(image_id: int, suit: int | str, cluster_id: int | None = None) -> dict[str, Any]:
    """套用预设方案（事务化编排）：配方多能力=一段记录链，整体生效或整体无痕；单图事务（批量归 workflow fan-out）。

    图级应用时 person 域步骤按人物聚类自动路由：聚类绑定 default_suit_id 则用绑定方案
    的 person 域步骤，否则用请求方案；image 域步骤随请求方案全图一次。
    straight 方案=零步骤基准：输出内容恒等输入。

    Args:
        image_id: 库内图像 ID。
        suit: 方案名（取最新版）或 preset_suits.id。
        cluster_id: 单聚类覆盖应用（只执行该聚类 person 域步骤，image 域跳过）；None=图级。

    Returns:
        成功 {status:"ok", suit, suit_id, steps, cache_hits, record_ids,
        output_sha256, output_path, routing, mask_paths}；
        失败 {status:"error", error}（事务保证零记录残留）。
    """
    return preset_cap.apply_preset_op(image_id, suit, cluster_id=cluster_id)


@mcp.tool()
def list_presets() -> dict[str, Any]:
    """列出预设方案（含 5 套内置种子与自建方案，版本化）。

    Returns:
        成功 {status:"ok", presets:[{id, name, version, is_builtin, recipe_version,
        steps:[{capability, params, scope}]}]}；失败 {status:"error", error}。
    """
    return presets_mod.list_presets_op()


@mcp.tool()
def save_preset(name: str, steps: list, version: int | None = None) -> dict[str, Any]:
    """自建预设方案（如「我的韩式哑光」）：入库即可被 apply_preset 消费；同名再存=升版。

    Args:
        name: 方案名。
        steps: [{capability, params, scope:"person"|"image"}]——capability 须在能力目录，
            person 域限 smooth_skin/whiten_teeth/brighten_eyes/face_enhance。
        version: 缺省=同名最新版+1（版本化追加，旧版不改）。

    Returns:
        成功 {status:"ok", id, name, version, steps}；失败 {status:"error", error}。
    """
    return presets_mod.save_preset_op(name, steps, version=version)


@mcp.tool()
def bind_cluster_suit(cluster_id: int, suit_id: int | None = None) -> dict[str, Any]:
    """把人物聚类绑定到默认预设方案（None=解绑）；图级 apply_preset 按人路由的依据。

    Args:
        cluster_id: 人物聚类 ID。
        suit_id: preset_suits.id 或 None。

    Returns:
        成功 {status:"ok", cluster_id, default_suit_id}；失败 {status:"error", error}。
    """
    return presets_mod.bind_cluster_suit_op(cluster_id, suit_id)


@mcp.tool()
def history(image_id: int) -> dict[str, Any]:
    """记录链查询：根 → 当前态的有序记录（capability/params/产物哈希/cluster_id/时间）。

    质检回路处置路径第一步：定位要回退到的记录。

    Args:
        image_id: 库内图像 ID。

    Returns:
        成功 {status:"ok", image_id, current_record_id, records:[{id, parent_id,
        capability, params, output_sha256, cluster_id, created_at}]}；
        失败 {status:"error", error}。
    """
    return chainops_mod.history_op(image_id)


@mcp.tool()
def rollback_to(image_id: int, record_id: int) -> dict[str, Any]:
    """回滚 = 移动当前态指针到目标记录（记录链只追加不删除，目标必须属于同一图像）。

    用途：修图复检不过/伪影 → 回退到出问题记录之前 → 换参重做（skill 质检回路）；
    联络表淘汰单张重修（workflow 检查点 3）。回退后同参重调会因链态复原而缓存命中。

    Args:
        image_id: 库内图像 ID。
        record_id: 目标记录 ID（history 查得）。

    Returns:
        成功 {status:"ok", image_id, current:{id, capability, params, output_sha256,
        output_path, cluster_id}}；失败 {status:"error", error}（含归属校验）。
    """
    return chainops_mod.rollback_to_op(image_id, record_id)


def _exit_grace_reaper(grace_s: float = 3.0) -> None:
    """进程生命周期兜底（B 类挂起防护；lead 裁决 2026-10-21，案卷见 issue #5 评论）。

    背景：DSH 运维刷新（touch cordis.patch.yml 触发重组重连）以 stdin EOF 方式淘汰旧
    server；若解释器内残留第三方原生非 daemon 线程（历史案例：mediapipe
    SerialDispatcher executor 卡 atexit join——已由 f2ac6f2 容器化根治，实测 assess×3
    后 EOF 0.5s 自退 rc=0），正常退出会被卡住，留下孤儿进程持有模型内存。
    兜底：宽限期后仍未退出 → stderr flush + os._exit(0)。

    设计取舍（勿试图「改进」为独立 stdin 监视线程）：
    - stdout/stdin 是 MCP 协议信道，**不可双读**（会偷协议字节）——mcp.run() 返回
      就是唯一安全的 EOF 信号，本函数只在 run() 返回后启动计时；
    - DB 安全：data.apply/assess/clusters 每操作同步 commit，静止态无未决写事务；
      WAL 语义下未回放日志由下次打开自动恢复（工单02 崩溃恢复已测）→ os._exit 无
      数据风险；
    - 正常路径零影响：reaper 是 daemon 线程，解释器自然退出时随之消失。
    """

    def _reap() -> None:
        time.sleep(grace_s)
        log.warning("传输已关闭但解释器 %ss 未退出（第三方原生线程滞留？）——os._exit(0) 兜底", grace_s)
        try:
            logging.shutdown()  # stderr flush
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)

    threading.Thread(target=_reap, daemon=True, name="exit-grace-reaper").start()


@mcp.tool()
def ingest(path: str) -> dict[str, Any]:
    """单图入库：原图只读登记（哈希+元数据+根记录），幂等（sha256 去重）。

    Args:
        path: 图像文件路径（绝对，或相对项目根）。一期输入域 JPEG/TIFF。

    Returns:
        成功 {status:"ok", path, image_id, sha256, duplicate, record_id?}——image_id
        供全部后续工具寻址；重复入库 duplicate=true 且 image_id 不变；
        失败 {status:"error", error}（不存在/非 JPEG-TIFF/不可解码，结构化不抛穿）。
    """
    return intake_mod.ingest_op(path)


@mcp.tool()
def ingest_album(directory: str | None = None) -> dict[str, Any]:
    """相册入库：扫描 album-inbox/（或指定目录，递归）逐图入库，失败隔离不阻塞整本。

    Args:
        directory: 目标目录；None = 项目 album-inbox/ 投递目录。

    Returns:
        成功 {status:"ok", directory, total, ingested:[{path,image_id,sha256,duplicate}],
        failed:[{path,error}], skipped:[{path,reason}]}（确定性按路径排序；total=
        ingested+failed；skipped=非 JPEG/TIFF 候选）；目录不存在 {status:"error", error}。
    """
    return intake_mod.ingest_album_op(directory)


@mcp.tool()
def set_verdicts(items: list[dict[str, Any]], source: str = "human") -> dict[str, Any]:
    """选片结论覆写/持久化（单一写入口）：verdict 写入 images.meta['verdict']，批量原子。

    Args:
        items: [{image_id:int, verdict:"keep"|"review"|"reject", reason?:str}, ...]。
        source: "report"=选片报告落库（workflow S0 段在 selection_report 后调用）；
            "human"=用户自然语言推翻（默认；spec 用户故事3，agent 解析后调用）。
        fan-out/联络表/导出一律消费 meta 里的最终值；human 源语义上压过机器判定。

    Returns:
        成功 {status:"ok", count, updated:[{image_id, verdict}]}；
        失败 {status:"error", error}（整批无痕回滚，结构化不抛穿）。
    """
    return album_mod.set_verdicts_op(items, source=source)


@mcp.tool()
def contact_sheet(image_ids: list[int] | None = None, thumb: int = 256, per_row: int = 6) -> dict[str, Any]:
    """联络表（三交付物之二，HTML+PNG）：修图前后成对网格；review 打标、reject 单列。

    Args:
        image_ids: 参与图像；None = 全库。
        thumb: 缩略图边长 px（64..1024）。
        per_row: 每行单元格数（1..12）。

    Returns:
        成功 {status:"ok", cached, sheet_png, index_html, directory,
        stats:{total, shown, review_marked, rejected_listed}}——幂等：同输入
        （图集+参数+各图当前态哈希+verdict）返回同一目录；失败 {status:"error", error}。
    """
    return album_mod.contact_sheet_op(image_ids, thumb=thumb, per_row=per_row)


@mcp.tool()
def export_album(image_ids: list[int] | None = None, album_name: str | None = None) -> dict[str, Any]:
    """导出目录（三交付物之三）：当前态产物规整命名复制 + manifest.json；默认跳过 reject。

    Args:
        image_ids: 导出图像；None = 全库（除 verdict=reject）。
        album_name: 相册目录名（var/exports/<album_name>/）；None = album_<时间戳>。
        命名规则 {序号:03d}_{原文件名stem}{ext}；manifest 记 image_id/sha256/链路摘要/verdict。

    Returns:
        成功 {status:"ok", directory, manifest_path, count, skipped_rejected}；
        失败 {status:"error", error}。同名相册重导=覆盖刷新。
    """
    return album_mod.export_album_op(image_ids, album_name=album_name)


def main() -> None:
    log.info("pixretouch server starting (stdio)")
    mcp.run()
    # run() 返回 = stdin EOF / 传输层关闭 → 退出路径；挂兜底防第三方线程卡死退出
    _exit_grace_reaper()


if __name__ == "__main__":
    main()
