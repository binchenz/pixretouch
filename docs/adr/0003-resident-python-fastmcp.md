# 常驻 Python MCP server（stdio）作为插件引擎

引擎为单个**常驻 Python 3.11 进程**，经 MCP server（stdio；mcp 2.x `MCPServer`，v1 名为
FastMCP——上游已移除 v1 模块，c979365 迁移）暴露全部修图工具。注册走 DSH cordis 补丁双层：
home 层 `~/.dsh/cordis.patch.yml`（可选、更高优先）与 GUI 管理的 profile 层
`~/.dsh/profiles/desktop/cordis.patch.yml`；形式 = 顶层数组 insert 一条
`@deepseek-ai/dsh-mcp-client` 条目（裸条目=覆盖补丁、未知 id 跳过 warn）；工具名
`mcp__pixretouch__<tool>`；`toolCallTimeoutMs=600000`（首次调用含模型冷加载 5~15s + 推理，
默认 60s 必超时）；`failOnStartupError=false` 的掩盖风险以 ps + doctor 复核纪律对冲。
模型首次调用时加载、之后常驻内存。skill（修图决策树）与 workflow（相册编排）叠加其上，不承担像素计算。
（文件名保留 fastmcp 字样以稳定外链；术语以本修订为准。）

## Considered Options

- **Node 壳 + 每调用起 Python 子进程**：拒绝。GFPGAN/LaMa 冷加载 5~15s，相册批处理数百次调用会慢到不可用。
- **纯 skill + CLI 工具（无常驻进程）**：拒绝。同样的冷启动问题，且失去 MCP 的工具类型契约与结构化错误传播。

## Consequences

- 内存足迹：**按需常驻**（懒加载 + 进程内复用），热态足迹取决于工作集——10 权重盘上共 1.1GB，
  实测单能力峰值 RSS ~2GB 量级（40MP 超分 1.9GB 可作下界参照；原「全模型热态 3~6GB」为估算口径，
  2026-10-21 修订）；M2 Max / 32GB 可接受。进程生命周期由 DSH MCP 管理器托管，重启只丢热态不丢
  数据（全部状态在 SQLite + 内容寻址缓存中）。**无调用中取消语义**：client 超时/断开后在途调用
  继续渲染至完成并单事务落库——处置=查库定论（见 GLOSSARY；README 已知限制 1）。
- 这是对原版架构的有意对齐：原版 = 常驻 PixCook 引擎 + pixcake-mcp 桥接进程（QLocalSocket 长度前缀 JSON-RPC）；我们 = 常驻 Python 引擎 + MCP stdio。未来读者若质疑"为何不用更轻的形态"，答案在此。
