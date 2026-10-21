# pixretouch

Agent 原生修图管线：以 DSH 三件套（MCP server + skill + workflow）白盒复刻像素蛋糕（PixCake）的修图能力架构。

- 设计词汇：`GLOSSARY.md`
- 架构决策：`docs/adr/`（白盒边界 / 记录链+急切渲染 / 常驻 Python 引擎）
- 工作跟踪：GitHub Issues（Spec + tracer-bullet 工单）
- 隐私：`album-inbox/`、本地图像库、缓存与导出目录全部 gitignore，照片不入仓库、不出机器
- 修图经验决策树：`skills/pixretouch/SKILL.md`；相册编排：`workflows/album-{oneshot,apply}.js`

## 安装

```bash
# 1) venv（Python 3.11；本项目用 uv，无 pip）
uv venv .venv --python 3.11
# 2) 依赖：持锁 + 清华源（多人协作纪律）；lock 为权威版本面
until mkdir var/.piplock 2>/dev/null; do sleep 5; done
uv pip install --python .venv/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.lock.txt   # 含 -e .（editable）
rmdir var/.piplock
# 3) editable 安装走 stub 包重定向（本机守护进程反复给 .venv 文件打 UF_HIDDEN，
#    .pth 机制不可靠）：.venv/lib/python3.11/site-packages/pixretouch/__init__.py
#    将 __path__ 指向 <repo>/src/pixretouch。**重建 venv 后必须重建该 stub**，
#    否则注册 spawn 的 server 导入失败且 failOnStartupError=false 会掩盖它
#    （见 var/qa/stub_immunity.md；复原后务必 ps + 新会话复核工具面）。
# 4) 模型权重（~1.1GB，hf-mirror 优先、ModelScope 兜底 id 注册在案）
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m pixretouch.prefetch
# 5) 环境自检
.venv/bin/python -m pixretouch.cli doctor   # 或 MCP 工具 doctor
```

## 运行（DSH 注册）

home 层补丁 `~/.dsh/cordis.patch.yml`（优先于 GUI 管理的 profile 层
`~/.dsh/profiles/desktop/cordis.patch.yml`；删除即下线回落）：

```yaml
- insert:
    - id: mcp-pixretouch
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: pixretouch
        transport: stdio
        command: <repo>/.venv/bin/python
        args: ['-m', 'pixretouch.server']
        cwd: <repo>
        env:
          PIXRETOUCH_HOME: <repo>
          HF_ENDPOINT: https://hf-mirror.com
        toolCallTimeoutMs: 600000   # 首次调用含模型冷加载 5~15s + 推理；默认 60s 必超时
        failOnStartupError: false
```

- 工具名 = `mcp__pixretouch__<tool>`（工具面以 server.py docstring 清单为准）。
- **代码刷新语义**：常驻实例持有 spawn 时刻代码，src/ 提交不自动生效。验收/演示前必刷新：
  方法 B（首选）`kill <pixretouch.server pid>` → DSH 退避自动 respawn（实测 ~8s）；
  方法 A（备选）注册条目任一 config 值语义微扰（如 toolCallTimeoutMs ±1）→ ~20s 原地重连；
  **非语义保存（touch/加注释）不触发重连 = 设计行为**；兜底 = 重启 DSH。（var/qa/registration.md）
- 一发模式/分段编排：`workflows/album-oneshot.js`、`workflows/album-apply.js`
  （头注含 meta/args 契约，对话 agent 提交 workflow 工具）；交互协议见 `skills/pixretouch/`。

## 缓存清理策略

- `var/cache/` = 内容寻址渲染缓存（+`mask:`/`material:` 前缀条目），同时**兼作记录链
  当前态/中间态产物的存储**——已编辑图的当前版本就存在这里（`records.output_path` 直接
  指向 cache 文件，`var/renders/` 为空）。
  - **未编辑图**（当前态=album-inbox 原图）：整目录删除完全安全，后续编辑自动重渲染重建。
  - **已编辑图**（当前态=渲染产物）：删除后该图**当前版本不可恢复**，后续编辑/导出报
    `FileNotFoundError`（原图与记录链参数审计仍在，可 `rollback_to` 根记录（原图）后重做，
    或先 `export_album` 保留成片）。
  - **安全清理流程**：先把要保留的成片 `export_album` → 再删 `var/cache/`。
    （口径实测：var/qa/verify_readme_ops.py，case 5）
- `var/pixretouch.db(+wal/shm)` = 记录链与库，**不可删**（追加式设计 = 审计与回滚的根据）；
  重置库 = 删 db 三件后从 ingest_album 重建（#15 验收开场动作即此）。
- `var/renders/`、`var/scratch/`、`var/reports/`、`var/exports/` = 派生产物，可删可重生成。
- `models/` = 权重缓存，可删（prefetch 重拉）；磁盘吃紧时优先按上条安全流程清 var/cache，其次 models。

## 磁盘预算

- 一期预算 6~8GB（spec 供给链决策）；**实测权重现值 1.1GB**（models/ du：codeformer 367M、
  lama 196M、isnet 168M、arcface 166M、depth-anything-v2-small 94M、realesrgan x4/x2 各 64M、
  face_landmarker 3.6M、genderage 1.3M、face_detector 236K——registry 10/10 齐装，doctor 可盘点）。
- var/cache 增长模型：每次生效编辑 ≈ 一份输出产物（JPEG q95 或 PNG；参照例：40MP×2 超分
  输出 22.4MB）。自用规模（数百张相册多轮迭代）估算 <10GB；余量低于 10GB 时 doctor 判 degraded。

## 隐私声明

- 照片不入 git：album-inbox/、var/、models/ 全部 gitignore（.gitignore 首节即隐私红线）。
- 照片不出机器：全部推理本地（torch-MPS / onnxruntime-CPU / mediapipe），无遥测、无云 API。
- 唯一外发面 = 供给链下载：pip（清华源）、权重（hf-mirror，ModelScope 兜底）——只取不改。
- 白盒复刻边界（ADR-0001）：不含任何第三方受保护资产；权重全部为开源许可
  （apache-2.0/bsd-3-clause 系，逐源核对记录在各能力模块 docstring）。

## 故障排查 / 已知限制

1. **孤儿渲染（无调用中取消语义）**：client 超时/断开后 server 对进行中调用继续渲染至
   自然完成（实测孤儿 245% CPU 24min）。处置口径：**超时≠失败≠回滚**——server 完成
   后单事务落库，重试前必须「查库定论」（history 看链是否已增长；已增长 = 已生效，禁止
   重跑防双倍渲染+双份记录；实测：本链位置重跑 = 二次渲染，超分场景 425 瓦片/262MP 量级）。
   workflow fan-out 已内建此策略（P3）；超分大图建议独立单发并留意尺寸门槛（输出边长
   >65500 或画布 >8GB 会被守卫结构化拒绝）。
2. **输入域上限**：PIL DecompressionBomb 安全阈值（~89MP 警告 / ~178MP 硬错）先于产品
   守卫触发；不解除（zip-bomb 防线）。一期输入域 JPEG/TIFF。
3. **UF_HIDDEN/stub 坑**：见「安装」第 3 条；症状 = 注册后工具面缺失/doctor 报 import 错，
   处置 = 重建 stub + ps 复核 + 方法 B 刷新。
4. **进程退出**：EOF 后正常自退（<1s）；另有 grace-reaper 兜底（第三方原生线程卡退出时
   3s 后 os._exit(0)，DB 无风险——每操作同步 commit + WAL 恢复语义）。案卷：issue #5 评论。
5. **torch.jit.load FutureWarning**（LaMa）：2.14 可用，torch.export 迁移另行跟踪。
6. **无人脸图套含 person 步骤方案**：person 段跳过 + person_skipped 审计字段 + image 段
   照常（计 ok 非错误）；显式 cluster_id 寻址无脸仍响亮报错。
7. **单图交付形态**：工具返回 output_path = 内容寻址文件（无扩展名）；单图交付
   约定 = 复制重命名 + sha256 核对（export_album 属批量场景）。
