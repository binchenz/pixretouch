# pixretouch

Agent 原生修图管线：以 DSH 三件套（MCP server + skill + workflow）白盒复刻像素蛋糕（PixCake）的修图能力架构。

- 设计词汇：`GLOSSARY.md`
- 架构决策：`docs/adr/`（白盒边界 / 记录链+急切渲染 / 常驻 Python 引擎）
- 工作跟踪：GitHub Issues（Spec + tracer-bullet 工单）
- 隐私：`album-inbox/`、本地图像库、缓存与导出目录全部 gitignore，照片不入仓库、不出机器
