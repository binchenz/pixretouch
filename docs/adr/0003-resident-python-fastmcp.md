# 常驻 Python FastMCP 服务作为插件引擎

引擎为单个**常驻 Python 3.11 进程**，经 FastMCP（stdio）暴露全部修图工具，注册进 `~/.dsh/cordis.patch.yml`（与 rea MCP server 同机制）；模型首次调用时加载、之后常驻内存。skill（修图决策树）与 workflow（相册编排）叠加其上，不承担像素计算。

## Considered Options

- **Node 壳 + 每调用起 Python 子进程**：拒绝。GFPGAN/LaMa 冷加载 5~15s，相册批处理数百次调用会慢到不可用。
- **纯 skill + CLI 工具（无常驻进程）**：拒绝。同样的冷启动问题，且失去 MCP 的工具类型契约与结构化错误传播。

## Consequences

- 全模型热态内存足迹约 3~6GB（M2 Max / 32GB 可接受）；进程生命周期由 DSH MCP 管理器托管，重启只丢热态不丢数据（全部状态在 SQLite + 内容寻址缓存中）。
- 这是对原版架构的有意对齐：原版 = 常驻 PixCook 引擎 + pixcake-mcp 桥接进程（QLocalSocket 长度前缀 JSON-RPC）；我们 = 常驻 Python 引擎 + MCP stdio。未来读者若质疑"为何不用更轻的形态"，答案在此。
