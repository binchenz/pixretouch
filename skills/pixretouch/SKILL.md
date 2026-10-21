---
name: pixretouch
description: PixRetouch 修图决策树——用 pixretouch MCP 工具修人像照片。触发场景：修图/美化/磨皮/美白/亮眼/人脸修复；合影按人分别处理；相册选片出报告；换背景/背景虚化；去痘痘/去路人/去杂物；老照片修复/超分放大；套用或自建预设方案（风格）；整本相册出片交付。不处理视频与非照片图像。
---

# PixRetouch 修图决策树

你通过 pixretouch MCP server 修图，工具名前缀 `mcp__pixretouch__`（下文省略前缀）。
三条全局常识，先于一切决策：

1. **结构化返回**：每个工具返回 `{status:"ok", ...}` 或 `{status:"error", error:"类型: 消息"}`。
   error 时读消息修正调用（缺前置就补前置，参数非法就改参数）；只有一类例外允许原样重试，
   见「质检回路」第 4 条。
2. **记录链语义**：原图永远只读；每个修图调用 = 在记录链上追加一条记录，当前态指针定义
   生效状态。推论：同参数重复调用会叠加两次效果——调整效果的正确动作是回退后重调，
   或换参数在新链态上重调。
3. **词汇自足定义**（与仓库 GLOSSARY.md 同义；本文自足，不依赖任何外部文件）：
   - **记录链**：单图追加式编辑序列，每条记录=能力+参数+前驱；修图即追加，永不改原图。
   - **当前态指针**：指向当前生效记录的引用；回滚=移动指针，不删记录。
   - **预设方案**：命名、可版本化的能力参数组合，可整体套用到一张图或一个人。
   - **人物聚类**：人脸相似度自动聚合的「同一个人」样本组，按人寻址的寻址单位。
   - **按人寻址**：参数以人物聚类为单位定向应用——同一张合影不同人各持参数组。
   - **选片**：相册自动质量评估（清晰度/曝光/闭眼/人脸可检出）并产出取舍报告。
   - **联络表**：修图前后成对对比的网格图，验收交付物之一。
   - **检查点**：批量流程中的人工确认节点（取舍/命名与配方案/成片终审）。
   - **一发模式**：跳过全部检查点、端到端全自动的批量模式。

## 主干流程

每步的完成判据写在句尾，达标才进下一步。

1. **照片入库**：单张 `ingest(path)`（JPEG/TIFF，幂等去重）；用户已把照片丢进投递目录
   `album-inbox/` 时用 `ingest_album()` 整目录入库。→ 拿到 `image_id`（后续一切工具的寻址键）。
2. **质检先行**：`assess_image(image_id)`。→ 读到四维度（sharpness / exposure / 闭眼 EAR /
   人脸可检出）与每张脸的 bbox、EAR、eyes_closed；据此在下节判定照片类型。
3. **定路线**：单人像/风景式快修 → 「预设方案选择规则」；点名具体诉求（磨皮/去痘/换背景…）
   → 「任务→能力映射」；**合影或多人照片 → 先走「合影支线」**。→ 明确要调用的工具与参数。
4. **执行修图**：按定好的路线调用；多人照片的每个人差异化参数用 `cluster_id` 按人寻址。
   → 每次调用返回 ok 且拿到 `output_path`（当前态产物）。
5. **质检回路**：每次修图后立即复检（硬规则，见专节）。→ 复检通过，或已按失败路径处置完毕。
6. **交付**：向用户报告做了什么（用记录链的 capability 序列说话）、复检结论、产物路径。
   注意 `output_path` 是**内容寻址缓存文件，无扩展名**（形如 var/cache/xx/yy/<sha256>；
   内容为 JPEG 或 PNG，按文件头判别：前 8 字节为 PNG 魔数则是 PNG，否则 JPEG）。
   单图规范交付约定：复制并重命名为 `<原名 stem>-retouched.<jpg|png>`（扩展名按上述
   嗅探），复制后以 sha256 与当前态记录核对一致，把副本路径给用户；相册批量交付走
   `export_album`（板块五）。

## 板块一 · 任务→能力映射

| 用户诉求 | 工具 | 关键参数与提示 |
|---|---|---|
| 整张快修（人像） | `apply_preset` | `suit` 见板块三；一步到位的事务性套系 |
| 磨皮（保纹理） | `smooth_skin` | `strength` 0.4~0.6 起步；频率分离，纹理不丢 |
| 牙齿美白 | `whiten_teeth` | `amount` 0.4~0.5；闭嘴照近恒等（正常，非故障） |
| 亮眼 | `brighten_eyes` | `amount` 0.25~0.35 |
| 人脸修复（模糊脸/老照片脸） | `face_enhance` | `strength` **默认 0.3 就是上限起点**，保守优先 |
| 去痘痘/路人/杂物 | `blemish_remove` | `points=[[x,y],...]` 或 `boxes=[[x0,y0,x1,y1],...]`，`brush`≈污点半径 |
| 换纯色/指定图背景 | `cutout` → `bg_replace` | bg_replace 给 `color=[r,g,b]` 或 `background_path` 二选一；已 cutout 的链态自动复用 alpha |
| 背景虚化（保留原背景） | `bg_blur` | 深度驱动，无需先 cutout；`strength` 0.6、`focus` 0.9 保护主体 |
| 白平衡/影调/黑白 | `color_grade` | `temperature/tint`±100、`exposure`±2.0EV、`contrast`±100、`wb_auto`、`lut="bw-classic"` |
| 低分辨率放大 | `super_resolve` | `scale=2` 或 `4`；**独立单发**，纪律见板块二「super_resolve 独立纪律」 |
| 相册选片 | `assess_image`×N → `selection_report` | 报告含每张 keep/review/reject+理由，落盘 JSON |
| 合影按人修 | 见「合影支线」 | `cluster_id` 是差异化参数的钥匙 |
| 重检人脸 | `detect_faces` | 聚类前刷新；覆盖写人脸表 |
| 自建风格套系 | `save_preset` | 步骤表见板块三 |
| 回退某步修图 / 查做过什么 | `history` → `rollback_to` | 回滚=移动当前态指针；记录链只追加不删除 |
| 推翻选片结论（「那张逆光的留下」） | `set_verdicts` | `items=[{image_id, verdict, reason?}]`，verdict∈keep/review/reject，`source="human"` |
| 联络表（修图前后对比网格） | `contact_sheet` | 网格 PNG + 逐对比较 + index.html |
| 导出交付目录 | `export_album` | 当前态规整命名复制 + manifest.json；默认跳过 reject |
| 环境问题 | `doctor` | 依赖/权重/磁盘/镜像源一次体检 |

## 板块二 · 照片类型→参数起点

判定信号全部来自步骤 2 的 assess 维度，不要裸猜。
**类型可叠加，处理优先级固定：先抢救、再精修、最后统调**——
① 抢救段（逆光/暗光/低分辨率多病共存时，按 逆光→暗光→低分辨率 取主矛盾做一个抢救段）；
② 人像精修段（磨皮/牙/眼/人脸修复）；③ 全局影调（`color_grade` 永远收尾）。
「合影」是结构不是类型：先走合影支线定结构（聚类+寻址），每个人的段内按 ①→②→③ 叠加。
组合示例（合影+逆光）：支线聚类 → 逐聚类 `face_enhance` 轻抢救 → 图级 `color_grade` 提曝收尾。

**单人像**（face_count=1，max_face_side≥110）
- 标准三件：`smooth_skin(0.4~0.6)` → `brighten_eyes(0.25~0.35)` → 露齿加 `whiten_teeth(0.4~0.5)`。
- 全局影调：`color_grade` 轻调（exposure ±0.3 内、temperature ±10 内）。
- 脸部曝光看 `face_exposure_mean`（主信号），全局均值只作参考——白背景/暗背景会系统性带偏全局值。

**合影**（face_count≥2）→ 走「合影支线」。

**逆光**（face_exposure_mean < 80，常见高光溢出 clip_hi 高）
- 轻中度：`color_grade(exposure=+0.3~+0.8)`，色偏加 `wb_auto=true`。
- 重度（脸部 < 60 或已 reject）：`apply_preset("rescue")`。

**暗光/低清**（exposure_mean < 70，或 sharpness 低、噪点感）
- 顺序：`face_enhance(0.3~0.5)` 先修脸 → `smooth_skin(0.3)` 轻磨 → `color_grade(exposure=+0.3~+0.5)`。
- 脸修复对干净照片近恒等（不破坏好素材），可放心先行。

**低分辨率/老照片**（max_face_side < 110，或最长边 < 1024）
- 顺序：`face_enhance(0.4~0.6)` 修脸 → `super_resolve(scale=2 或 4)` 整体放大。
- 放大后如需继续修，重新 `assess_image`（维度已变）。
- **super_resolve 独立纪律**：它是单发操作，任何批量流程（板块五）都不含它。
  大图（>12MP）×4 前必须警示用户：耗时分钟级到几十分钟，且**中途不可取消**——
  DSH 工具 600s 超时 ≠ 失败，server 会继续渲染完成并落记录链；超时后先 `history`
  查当前态定论（记录在链上 = 已生效），勿盲目重跑（重跑 = 在已放大的图上再放大）。

**废片处置**（selection_report 判 reject 或 assess 维度触线）
- 模糊：`face_enhance(0.5)` 抢救一次，复检不过就建议换片——选片语义下废片的正解是淘汰。
- 闭眼（eyes_closed）：单人像建议换片；合影见支线。
- 分辨率不足：`super_resolve` + `face_enhance` 组合抢救，预期管理要诚实。

### 合影支线（按人寻址）

1. `cluster_faces(image_ids=[本图 image_id])` → 人物聚类。**单张修图必传 image_ids
   收窄到本图**（缺省 = 全库未分配人脸，会把库里其他照片卷进来）；相册批量才传整批
   或缺省。image_id 一律取自本次会话 `ingest` 的返回值，不假设库里已有旧 id。
   聚类增量、幂等：重跑不动已分配的脸。
2. `list_clusters()` → 核对聚类正确性：face_count 应等于图中人数；faces 明细
   （face_id/image_id/bbox/quality）对照原图逐脸核验（bbox 位置对得上人头）。
   **命名可选**：用户给了身份（「新娘」）→ `rename_cluster(cluster_id, name="新娘")`；
   没给身份可以不命名，直接用 cluster_id 走完全程；需要人类可读指代时按 faces 明细的
   bbox x 坐标从左到右兜底命名（「左一」「中」「右一」），或用 person_type（「男士1」）。
   person_type（female/male/child）自动初判不准就在 rename_cluster 覆盖。
3. 聚类错了：同一人被拆散 → `merge_clusters(source_ids, target_id)`；误归的脸 →
   `assign_face(face_id, cluster_id)`；漏检的脸先 `detect_faces(image_id)` 再重聚类。
4. 差异化修图。**默认分支（用户没提任何按人差异化诉求）：图级统一套系
   `apply_preset(image_id, suit)`——最常见路径；已绑定方案的聚类自动按人路由。**
   用户提出差异化时才升级：
   - 单人特判（「这张里的孩子磨皮轻一点」）：`smooth_skin(image_id, cluster_id=孩子, strength=0.3)`
     或 `apply_preset(image_id, suit, cluster_id=...)`（只跑该聚类的人像步骤）。
   - 固定风格跟人走：`bind_cluster_suit(cluster_id, suit_id)` 后图级 `apply_preset`——
     绑定者自动用自己的方案，未绑定者用请求方案。
5. 露齿补白：natural 等套系不一定含牙段——某聚类的脸露齿笑（看原图或产物）→
   `whiten_teeth(image_id, cluster_id=该聚类, amount=0.4~0.5)` 补一步（editorial 已含
   牙段无需补）。对闭嘴脸调用无害：mask 近空 → 近恒等，只多一条记录。
6. 合影闭眼判读：`assess.faces[].eyes_closed` 按脸看——合影总有人眨眼，整图 reject 不适用；
   同场景多张时用逐脸 EAR 挑全员睁眼的那张。

## 板块三 · 预设方案选择规则

预设方案 = 命名的、可版本化的能力参数组合，`apply_preset` 事务性套用（整套生效或整套无痕，
失败可安全重试）。内置 5 套（`list_presets()` 查完整配方）：

| 方案 | 配方（person 段 + image 段，数值=当前 v1，与 list_presets() 同步、冲突以工具返回为准） | 用它的时机 |
|---|---|---|
| `natural` | smooth_skin 0.4 + brighten_eyes 0.25 ＋ color_grade(temperature=4, exposure=0.1) | **日常默认**；拿不准就它 |
| `editorial` | face_enhance 0.25 + smooth_skin 0.55 + whiten_teeth 0.4 ＋ color_grade(contrast=18, temperature=-4) | 成片感/杂志感交付 |
| `bw-classic` | smooth_skin 0.4 ＋ color_grade(lut="bw-classic") | 黑白基准配方 |
| `rescue` | face_enhance 0.5 + smooth_skin 0.5 + brighten_eyes 0.3 ＋ color_grade(wb_auto=true, exposure=0.5) | 逆光/暗光/废片抢救 |
| `straight` | 零步骤 | 对照基准：输出恒等输入；验证引擎无损；用户要求原样 |

选择规则（判定分支，对号即走）：
- 用户没报风格 → `natural` 起步，复检后按需微调。
- 用户要成片感/「修得精致些」/杂志风 → `editorial`。
- 用户说「黑白」「单色」 → `bw-classic`。
- assess 判 reject 且原因是逆光/暗光（face_exposure_mean 低），或用户说「这张废片救一下」 → `rescue`。
- 用户要求「别动原片」/要对照基准/验证引擎无损 → `straight`。
- 用户描述了一个可复用的口味（「我的韩式哑光」）→ `save_preset(name, steps)` 自建：
  steps 为 `[{capability, params, scope}]`，scope=person 仅限
  smooth_skin/whiten_teeth/brighten_eyes/face_enhance，其余用 image；同名再存 = 升版，
  apply 默认取最新版。
- 套系里的步骤是链上一段连续记录：单步不满意不必推翻整套——回退到套系前记录，
  改用单能力手工组合。
- 无人脸的图（风景/静物混入相册）：含 person 步骤的方案照常可用——person 段自动跳过、
  image 段照常执行（返回体带 `person_skipped` 审计字段），无需为风景图换方案。
- 自建配方的两层校验口径：`save_preset` 只做结构校验（能力在册 / scope 合法 / person
  白名单 / LUT 参数互斥）；数值语义边界由单能力工具层把守——自建配方里的越界数值
  （如 strength>1）apply 时不响亮失败，渲染层会裁剪或恒等降级。写配方对照板块一参数域自查。
- 外部 LUT 进配方：params 写 `{"lut_path": "<绝对或相对项目根的 .cube 路径>"}`——引擎按
  素材规则下降为内容 sha256（路径不进缓存键；文件须在 apply 时存在，格式错误当场结构化报错）。

## 板块四 · 质检回路（硬规则）

1. **任何修图调用之后，必须立即 `assess_image(image_id)` 复检。** 渲染类工具全算：
   smooth_skin / whiten_teeth / brighten_eyes / face_enhance / blemish_remove / cutout /
   bg_replace / bg_blur / color_grade / super_resolve / apply_preset。每次修完即检，
   攒一批再检 = 违例（伪影会被后续步骤叠加掩盖，无法归因）。
2. **对比基准**：修前维度（步骤 2 已拿）vs 修后维度——
   sharpness 允许小幅下降（磨皮低频均匀化，≤20% 正常；>50% = 过度，处置）；
   face_count 不得减少（减少了 = 修图损伤人脸可检出性，处置）；
   不得新增 eyes_closed（修图把眼睛修闭了 = 严重伪影，处置）。
3. **处置路径**（复检不过或肉眼见伪影）：`history(image_id)` 查记录链 →
   `rollback_to(image_id, record_id)` 回退到出问题的记录之前 → 参数减半重调。
   同参数原样重调是无效动作（记录链只追加，重复 = 双倍效果）——重调前必须改变
   参数或回退位置二者之一。
4. **超时自愈契约（唯一允许原样重试的情形）**：error 消息含
   `mediapipe inference timeout` → 把同一调用原样重试一次（推理容器已自动换代自愈，
   实证低频），再失败才按真错误处置。
5. **肉眼终检**（维度看不出的主观伪影，看 output_path 产物）：塑料感皮肤 → strength 降到
   ≤0.3；假牙感 → amount ≤0.3；修复后「换脸感」→ face_enhance strength ≤0.2。
   保守参数是本管线的第一美学：默认值已经保守，加码永远比减码难救。

### 维度速查（assess_image 返回字段，判读的共同语言）

| 字段 | 含义 | 判读线 |
|---|---|---|
| `sharpness` | 无参考清晰度（拉普拉斯方差，统一 1024 工作尺度） | <60 模糊 reject；60~110 偏低复核 |
| `exposure_mean` | 全局平均灰度 | 仅参考——背景会系统性带偏 |
| `face_exposure_mean` | 脸部区域平均灰度（**曝光主信号**） | <45 或 >210 reject；45~70 / 195~210 复核 |
| `clip_hi` | 全局高光溢出比例 | >35% 且脸部同偏亮才复核（白背景棚拍单独高溢出不淘汰） |
| `face_count` / `faces[]` | 检出脸数与逐脸明细（bbox/quality/EAR/eyes_closed/wink） | 合影支线的分流依据 |
| `max_face_side` | 最大脸短边（原生像素） | <60 分辨率不足 reject；<110 修图收益有限 |
| `eyes_closed` / `wink` | 双眼 EAR<0.15 / 仅单眼低于 | 单人像闭眼→换片或抢救；合影逐脸判读 |
| `width` / `height` | 原生分辨率 | 最长边 <512 → 分辨率不足 reject |

### 常见错误 → 修正动作（error 消息对号）

| error 含 | 含义 | 修正动作 |
|---|---|---|
| `mediapipe inference timeout` | 推理容器自愈换代（低频） | **原样重试一次**（质检回路第 4 条），再失败按真错误 |
| `KeyError: image ... not found` | image_id 不在库 | 重新 `ingest`；批量前先 `ingest_album` |
| `未检出人脸，人像能力无目标` | 照片无可检出人脸 | 先 `assess_image` 看 face_count；确认是人脸照片再重试 |
| `聚类 ... 没有检出人脸` | cluster_id 与本图无交集 | `list_clusters` 核对 id；必要时 `detect_faces` + `cluster_faces` 刷新 |
| `ValueError`（参数越界/非法） | 参数超出合法域 | 对照板块一/二把参数收回合法域（strength/amount ∈ (0,1] 等） |
| `FileNotFoundError` / 权重缺失 | 模型权重未预取 | `doctor` 体检 → 按提示跑 `python -m pixretouch.prefetch` |
| `not a readable image` | 文件损坏或非图像 | 该图从批次剔除并记录（失败隔离），不重试 |

## 板块五 · 批量与 workflow 升级

**单张对话式修图（1~3 张）**：留在主干流程。
**≥4 张或整本相册 → 批量路径**（工具直调版，按序执行；检查点 = 向用户呈交并等指令）：

1. 入库：`ingest_album()`（默认扫投递目录 album-inbox/，递归、失败隔离、幂等去重）或逐张 `ingest`。
2. 选片：`selection_report(image_ids)`——未质检的图自动补测，直接产出相册级报告
   （每张 keep/review/reject + 可读理由，落盘 JSON）。呈交用户 = **检查点 1 取舍确认**。
3. 推翻与确认：用户自然语言改判（「那张逆光的留下」）→ `set_verdicts(items, source="human")`。
4. 多人相册：`cluster_faces()` → `rename_cluster`（「新娘」「伴娘」）→ 按人配方案
   `bind_cluster_suit`。呈交命名与配方案 = **检查点 2**。
5. 批量套用：对每张 keep/review 图 `apply_preset` → `assess_image` 复检（质检回路铁律，
   逐张成对）；单图失败隔离——记录错误继续下一张，批末汇总失败清单。
6. 终审：`contact_sheet()` 出联络表（修图前后对比网格）呈交 = **检查点 3 成片终审**；
   淘汰单张重修 = `history` 定位 → `rollback_to` → 换参重做 → 复检 → 重出联络表。
7. 交付：`export_album(album_name)`——当前态规整命名导出目录 + manifest.json（默认跳过 reject）。

**升级 workflow 相册编排**（工单 13 交付物，DSH workflow 脚本），满足任一即移交：
用户要「一口气跑完/演示」（一发模式，零检查点全自动）；相册规模大到需要 fan-out
自动重试与失败隔离编排；或要三检查点以对话指令暂停/恢复的完整体验。
移交交接物：image_id 清单 + selection_report 落盘路径 + 已命名/已绑定的人物聚类状态
（workflow 直接消费，不重做）。workflow 交付前，上述批量路径就是其语义等价物，照走不误。
