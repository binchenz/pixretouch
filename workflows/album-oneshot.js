// ============================================================================
// PixRetouch 一发模式（工单 13 / issue #14）—— DSH workflow 工具脚本正文
//
// 零检查点全自动直通七阶段：ingest → 选片报告 → (检查点1 缺席=消费报告 verdict)
// → 自动聚类 → (检查点2 缺席=默认方案路由) → 批量套用 fan-out → 联络表
// → (检查点3 缺席=不终审) → 导出目录。三交付物齐出。
//
// 使用方式同 album-apply.js（正文=script、下方 meta=meta、args 按契约）。
//
// meta = {
//   "name": "pixretouch-album-oneshot",
//   "description": "PixRetouch 一发模式：相册七阶段零检查点全自动（入库→选片→聚类→批量套用→联络表→导出），三交付物齐出",
//   "whenToUse": "演示/回归验证/信任全自动取舍时；需要人工检查点请改用分段编排（S0 直调 + album-apply.js + S3 直调）",
//   "phases": [{"title":"intake"},{"title":"selection"},{"title":"cluster"},{"title":"apply"},{"title":"contact-sheet"},{"title":"export"}]
// }
// args = {
//   directory:    string|null   入库目录；缺省 null=项目 album-inbox/
//   suit_default: "natural"     图级默认方案（聚类已绑默认方案者 person 段自动按人路由）
//   batch_size:   10            fan-out 批粒度 8..12
//   album_name:   string|null   导出相册名；缺省 null=album_<时间戳>
// }
//
// 硬规则：与 album-apply.js P1~P7 完全一致（apply 段同一策略文本，双向锁定
// tests/test_e2e_album.py 的 MCP 序列复刻）。含 P7：无人脸图 person 段跳过 +
// person_skipped 审计字段 + image 段照常（计 ok 不隔离）。O3 裁决：自动取舍
// **不另设常数**，直接消费 selection_report 的 verdict（唯一事实源=工单04 assess
// 校准）：keep→套用、review→套用（联络表打标）、reject→跳过并列入返回报告。
//
// 返回（无损 JSON）：{ingest, selection, clusters, apply:{...}, contact_sheet, export}
// ============================================================================

phase("intake");
const intake = await agent(
  `调用 PixRetouch MCP 工具 ingest_album(${args.directory ? `directory="${args.directory}"` : ""}) 将相册入库（原图只读、幂等去重、失败隔离），将其 JSON 结果原样返回。`,
  {
    label: "intake", phase: "intake",
    schema: {
      type: "object",
      properties: {
        status: { type: "string" }, directory: { type: "string" }, total: { type: "integer" },
        ingested: { type: "array", items: { type: "object", properties: {
          path: { type: "string" }, image_id: { type: "integer" },
          sha256: { type: "string" }, duplicate: { type: "boolean" },
        }, required: ["image_id"], additionalProperties: true } },
        failed: { type: "array", items: { type: "object", additionalProperties: true } },
        skipped: { type: "array", items: { type: "object", additionalProperties: true } },
      },
      required: ["status", "ingested"], additionalProperties: true,
    },
  }
);
if (!intake || intake.status !== "ok" || !intake.ingested.length) {
  throw new Error(`入库失败或空相册：${intake ? intake.status : "agent null"}`);
}
const ids = intake.ingested.map((it) => it.image_id);
log(`入库 ${ids.length} 图（failed ${intake.failed ? intake.failed.length : 0}，skipped ${intake.skipped ? intake.skipped.length : 0}）`);

phase("selection");
const selection = await agent(
  `依次调用 PixRetouch MCP 工具：
1. selection_report()（相册级选片报告，未质检图自动补测）；
2. 将报告 images[] 的每张 {image_id, verdict, reasons} 组装为 items，调 set_verdicts(items, source="report") 持久化。
返回 JSON：{report_path, counts: 报告 album 字段, verdicts: [{"image_id":int,"verdict":"keep|review|reject"}...全量]}。`,
  {
    label: "selection", phase: "selection",
    schema: {
      type: "object",
      properties: {
        report_path: { type: "string" },
        counts: { type: "object", additionalProperties: true },
        verdicts: { type: "array", items: { type: "object", properties: {
          image_id: { type: "integer" }, verdict: { type: "string", enum: ["keep", "review", "reject"] },
        }, required: ["image_id", "verdict"], additionalProperties: true } },
      },
      required: ["verdicts"], additionalProperties: true,
    },
  }
);
if (!selection) throw new Error("选片段子代理失败");
// O3：reject→跳过；keep/review→套用（review 联络表自动打标）
const targets = selection.verdicts.filter((v) => v.verdict !== "reject").map((v) => v.image_id);
const rejected = selection.verdicts.filter((v) => v.verdict === "reject").map((v) => v.image_id);
log(`选片：套用 ${targets.length} / 跳过(reject) ${rejected.length}`);

phase("cluster");
const clusters = await agent(
  `依次调用 PixRetouch MCP 工具 cluster_faces()（全库未分配人脸增量聚类）与 list_clusters()，返回 JSON：{assigned, created, clusters:[{id,name,person_type,face_count,default_suit_id}]}。`,
  {
    label: "cluster", phase: "cluster",
    schema: {
      type: "object",
      properties: {
        assigned: { type: "integer" }, created: { type: "integer" },
        clusters: { type: "array", items: { type: "object", additionalProperties: true } },
      },
      required: ["clusters"], additionalProperties: true,
    },
  }
);
log(`聚类：${clusters ? clusters.clusters.length : 0} 个人物聚类（一发模式不做人工命名，person 段按聚类默认方案自动路由，未绑定→图级默认方案）`);

phase("apply");
const suitDefault = args.suit_default || "natural";
const batchSize = Math.min(12, Math.max(8, args.batch_size || 10));
const batches = [];
for (let i = 0; i < targets.length; i += batchSize) batches.push(targets.slice(i, i + batchSize));

const batchPrompt = (batch) => `你在执行 PixRetouch 一发模式的批量套用批次（MCP 工具前缀 mcp__pixretouch__）。批内逐图**串行**，单图失败不得中断其余图。

每张图 image_id=<ID> 的步骤：
0. 先调 history(image_id=<ID>) 记链长基线（供超时定论）。
1. 调 apply_preset(image_id=<ID>, suit="${suitDefault}")（图级应用；person 段按聚类绑定自动路由）。
2. status=ok → 立即调 assess_image(image_id=<ID>) 复检（硬规则），摘要记 assess 字段。

硬规则：
- error 含 "mediapipe inference timeout" → 同参重试一次（事务化零残留），仍失败 → 隔离。
- 调用超时/中断 → 不当失败：history 对比基线——链已增长=已套用，计 ok 禁止重跑；未增长 → 等 30 秒复查，仍未增长 → 同参重试一次，再失败 → 隔离。
- 返回 ok 且含 person_skipped 字段 → 无人脸图（风景/静物）：person 段跳过、image 段已照常执行——计 ok，不隔离。
- 其余错误 → 直接隔离。

批次清单：${batch.map((id) => `image_id=${id}`).join(", ")}
返回 JSON：{results:[{image_id, status:"ok"|"isolated", suit, record_ids, attempts, assess, error}]}（每张图一条）。`;

const applyResults = await pipeline(batches, async (_prev, batch, i) =>
  agent(batchPrompt(batch), {
    label: `apply-b${i}`, phase: "apply",
    schema: {
      type: "object",
      properties: {
        results: { type: "array", items: { type: "object", properties: {
          image_id: { type: "integer" }, status: { type: "string", enum: ["ok", "isolated"] },
          suit: { type: "string" }, record_ids: { type: "array", items: { type: "integer" } },
          attempts: { type: "integer" }, assess: { type: "string" }, person_skipped: { type: "string" },
          error: { type: "string" },
        }, required: ["image_id", "status"], additionalProperties: true } },
      },
      required: ["results"], additionalProperties: false,
    },
  })
);
const perImage = [];
const batchFailed = [];
applyResults.forEach((r, i) => {
  if (r && Array.isArray(r.results)) perImage.push(...r.results);
  else batchFailed.push({ batch_index: i, image_ids: batches[i] });
});

phase("contact-sheet");
const sheet = await agent(
  `调用 PixRetouch MCP 工具 contact_sheet()（全库；只读当前态，无重渲染），将其 JSON 结果原样返回。`,
  {
    label: "contact-sheet", phase: "contact-sheet",
    schema: {
      type: "object",
      properties: {
        status: { type: "string" }, sheet_png: { type: "string" }, index_html: { type: "string" },
        directory: { type: "string" }, stats: { type: "object", additionalProperties: true },
      },
      required: ["status"], additionalProperties: true,
    },
  }
);

phase("export");
const exported = await agent(
  `调用 PixRetouch MCP 工具 export_album(${args.album_name ? `album_name="${args.album_name}"` : ""})（当前态规整命名导出 + manifest；verdict=reject 自动跳过），将其 JSON 结果原样返回。`,
  {
    label: "export", phase: "export",
    schema: {
      type: "object",
      properties: {
        status: { type: "string" }, directory: { type: "string" }, manifest_path: { type: "string" },
        count: { type: "integer" }, skipped_rejected: { type: "integer" },
      },
      required: ["status"], additionalProperties: true,
    },
  }
);

return {
  ingest: { total: intake.total, ingested: ids.length,
            failed: intake.failed || [], skipped: intake.skipped || [] },
  selection: { counts: selection.counts || null, report_path: selection.report_path || null,
               rejected_skipped: rejected },
  clusters: clusters ? clusters.clusters : [],
  apply: { targets: targets.length, applied: perImage.filter((r) => r.status === "ok").length,
           isolated: perImage.filter((r) => r.status !== "ok"), batch_failed: batchFailed },
  contact_sheet: sheet,
  export: exported,
};
