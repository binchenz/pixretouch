// ============================================================================
// PixRetouch S2 批量套用段（工单 13 / issue #14）—— DSH workflow 工具脚本正文
//
// 使用方式（引擎无仓库脚本发现机制，本文件=版本化正文）：对话 agent 按 #13 skill
// 指引，取本文件正文为 workflow 工具的 script 参数、下方 meta 为 meta 参数、
// 按 args 契约构造 args 提交。工具名在 DSH 侧前缀 mcp__pixretouch__。
//
// meta = {
//   "name": "pixretouch-album-apply",
//   "description": "PixRetouch S2 批量套用：预设方案批式 fan-out（默认批10）+ 逐图 assess 复检 + 失败隔离 + timeout 查库定论 + 联络表",
//   "whenToUse": "相册编排检查点2（命名与配方案）之后：按方案路由批量套用并产出联络表供检查点3终审",
//   "phases": [{"title": "apply"}, {"title": "contact-sheet"}]
// }
// args = {
//   image_ids:   [int, ...]        必填非空；应为非 reject 图（O3：reject→跳过）
//   routing:     {"<image_id>": "<预设方案名>"}  逐图方案路由；缺省 "natural"
//                                  （方案按名 fan-out，版本演进自动跟随——dev-b 契约）
//   batch_size:  10                批粒度 8..12（= 取消语义限界，见 P5）
// }
//
// 硬规则（与 tests/test_e2e_album.py S2 段、#13 skill 双向锁定；lead 三约束 2026-10-21）：
// P1 每图 apply_preset(suit 按名) → status=ok 后必须 assess_image 复检（质检回路硬规则）。
// P2 结构化错误含 "mediapipe inference timeout" → 同参重试一次（apply_preset 事务化：
//    失败零记录残留，重跑安全且命中已完成段缓存）。
// P3 客户端调用超时/中断类 → **超时≠失败≠回滚**：server 无调用中取消语义，孤儿渲染
//    仍会完成并单事务落库。处置=查库定论：先 history(image_id) 记链长基线再调用；
//    超时后查 history——链已增长=实际已套用，计 ok **禁止重跑**（防双倍渲染+双份记录）；
//    未增长=等 30s 复查一次，仍未增长才按同参重试一次，再失败→隔离。
// P4 其余错误 → 直接隔离记录 {image_id, error}，单图失败不阻塞批内其余图。
// P5 批粒度（默认 10）= 取消语义限界：单批最坏 ≈ 10×单图时长；5 套种子配方单图全部
//    <60s 量级（无 super_resolve 级重资产步骤），apply_preset 单图稳落 600s
//    toolCallTimeoutMs 内；批间独立 agent() 调用，job_kill 的取消面=当前批。
// P6 尾段 contact_sheet 只读当前态产物缩略、无重渲染（天然安全）；检查点3「淘汰单张
//    重修」= 对话直调 history/rollback_to（e604803 工具面）+ 能力工具，或对子集重跑
//    本脚本（幂等：缓存命中使重跑便宜）。
// P7 无人脸图（风景/静物）：图级 apply_preset 的 person 域步骤整体跳过 + image 域照常
//    执行 + 返回 person_skipped 审计字段（dev-b 裁决 2026-10-21）——计 ok，不是错误，
//    fan-out 无需为风景图做特殊隔离分支。显式 cluster_id 寻址无脸仍响亮报错（走 P4）。
//
// 返回（无损 JSON）：{total, applied, isolated[], batch_failed[], contact_sheet}
// batch_failed 仅在 agent()=null（子代理级崩溃）时出现——整批未处理，可原样重跑本
// 脚本（P6 幂等）。
// ============================================================================

phase("apply");

const ids = args.image_ids;
if (!Array.isArray(ids) || ids.length === 0) {
  throw new Error("args.image_ids 必须为非空数组");
}
const routing = args.routing || {};
const batchSize = Math.min(12, Math.max(8, args.batch_size || 10));
const batches = [];
for (let i = 0; i < ids.length; i += batchSize) batches.push(ids.slice(i, i + batchSize));
log(`fan-out：${ids.length} 图 / ${batches.length} 批（批粒度 ${batchSize}，取消语义限界=P5）`);

const batchPrompt = (batch) => `你在执行 PixRetouch 相册批量套用的一个批次（MCP 工具前缀 mcp__pixretouch__）。批内逐图**串行**处理，单图失败不得中断其余图。

每张图 image_id=<ID> 的步骤：
0. 先调 history(image_id=<ID>) 记下当前链长（基线，供超时定论用）。
1. 调 apply_preset(image_id=<ID>, suit=<该图方案名>)。
2. status=ok → 立即调 assess_image(image_id=<ID>) 复检（硬规则），复检摘要记入 assess 字段。

硬规则（严格遵守）：
- 返回 error 含 "mediapipe inference timeout" → 同参重试 apply_preset 一次（事务化、失败零残留、重跑安全），仍失败 → 隔离。
- 调用超时/被中断 → 不当失败：查 history(image_id) 对比基线——链已增长=实际已套用，计 ok，禁止重跑；未增长 → 等 30 秒复查一次，仍未增长 → 同参重试一次，再失败 → 隔离。
- 返回 ok 且含 person_skipped 字段 → 无人脸图（风景/静物）：person 段跳过、image 段已照常执行——计 ok，不是错误，不隔离。
- 其余错误 → 直接隔离（status="isolated"，error 记原文）。

批次清单：
${batch.map((id) => `- image_id=${id}：suit="${routing[id] || "natural"}"`).join("\n")}

返回 JSON：{results:[{image_id, status:"ok"|"isolated", suit, record_ids, attempts, assess, person_skipped?, error}]}——批内每张图都要有一条结果（含隔离的）。`;

const results = await pipeline(batches, async (_prev, batch, i) =>
  agent(batchPrompt(batch), {
    label: `apply-b${i}`,
    phase: "apply",
    schema: {
      type: "object",
      properties: {
        results: {
          type: "array",
          items: {
            type: "object",
            properties: {
              image_id: { type: "integer" },
              status: { type: "string", enum: ["ok", "isolated"] },
              suit: { type: "string" },
              record_ids: { type: "array", items: { type: "integer" } },
              attempts: { type: "integer" },
              assess: { type: "string" },
              person_skipped: { type: "string" },
              error: { type: "string" },
            },
            required: ["image_id", "status"],
            additionalProperties: true,
          },
        },
      },
      required: ["results"],
      additionalProperties: false,
    },
  })
);

phase("contact-sheet");

const sheet = await agent(
  `调用 PixRetouch MCP 工具 contact_sheet(image_ids=${JSON.stringify(ids)}) 产出联络表（只读当前态，无重渲染），将其 JSON 结果原样返回。`,
  {
    label: "contact-sheet",
    phase: "contact-sheet",
    schema: {
      type: "object",
      properties: {
        status: { type: "string" },
        cached: { type: "boolean" },
        sheet_png: { type: "string" },
        index_html: { type: "string" },
        directory: { type: "string" },
        stats: { type: "object", additionalProperties: true },
      },
      required: ["status"],
      additionalProperties: true,
    },
  }
);

const perImage = [];
const batchFailed = [];
results.forEach((r, i) => {
  if (r && Array.isArray(r.results)) perImage.push(...r.results);
  else batchFailed.push({ batch_index: i, image_ids: batches[i], note: "agent()=null 子代理级崩溃；整批未处理，可重跑（P6 幂等）" });
});

return {
  total: ids.length,
  applied: perImage.filter((r) => r.status === "ok").length,
  isolated: perImage.filter((r) => r.status !== "ok"),
  batch_failed: batchFailed,
  contact_sheet: sheet,
};
