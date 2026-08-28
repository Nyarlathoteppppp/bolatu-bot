# 2026-08-28 项目审计

审计时间：2026-08-28（服务器本地时间）  
范围：生产服务器、Docker 部署、SQLite/RAG 卫生、备份任务、依赖、全量测试、关键聊天/工具/管理热路径。  
结论：**生产可运行，但当前质量门不全绿；不应把现状描述为“无已知问题”。**

## 1. 运行快照

- `qq-social-agent-bot`：healthy，`/readyz` 返回 database、LLM、OneBot 均 ready。
- NapCat：运行中，后端已连接 Bot `1801507496`。
- 磁盘：40G 总量，已用约 17G，剩余约 22G。
- 内存：3.6G，总可用约 2.8G；Swap 使用约 205M / 2G，处于正常缓冲范围。
- SQLite：完整性与 quick check 均为 `ok`；数据库约 154MiB。
- RAG：活动 conversation 文档约 1.27 万、活动 summary 约 1118；最近 24h 检索无错误，平均约 0.65-0.77 秒。
- 备份：COS、历史归档、ntqq 渐进归档均有已部署的计划任务。

## 2. 验证结果

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| 后端就绪检查 | 通过 | 数据库、LLM、OneBot 均 ready |
| Python 编译 | 通过 | `compileall` 无语法错误 |
| 依赖一致性 | 通过 | `pip check` 无破损依赖 |
| 全量测试 | **失败** | 482 通过，2 失败，见 P1 |
| SQLite 完整性 | 通过 | integrity / quick check 均为 ok |
| RAG 垃圾检查 | 轻微待清理 | 21 条 nonactive embedding，自动体检会处理 |
| URL 读取 SSRF 防护 | 通过审阅 | 限 scheme/端口、禁凭据、DNS 解析后要求公网 IP、每次重定向重新校验 |
| 管理端公网暴露 | 未发现 | bot / NapCat 端口仅绑定 `127.0.0.1` |

## 3. 发现与待办

### P1-1：全量测试当前不全绿

**证据：** `python -m pytest -q` 结果为 482 passed、2 failed。

1. `tests/test_meme_library.py::test_group_meme_gate_respects_group_cooldown`
   - `PrivateMemeLibrary` 初始化契约会创建 `meme_library` 目录；测试随后再次调用 `mkdir(parents=True)` 且未指定 `exist_ok=True`，导致测试夹具失败。
   - 运行功能不受影响，但 CI / 发布质量门被破坏。
   - 修复：测试使用 `exist_ok=True`，并补充“库初始化创建目录”的明确断言。

2. `tests/test_memory_store.py::test_style_rules_merge_equivalent_group_rules`
   - 测试期望两条语义近似的群风格规则合并，但实现只在标准化后的 fingerprint 完全一致时合并。
   - 这说明“风格规则防重复”的设计目标尚未完全落地，长期会重新积累近义规则。
   - 修复：在精确 fingerprint 外增加可解释的近义候选判断；先以确定性关键词/相似度筛选候选，必要时对后台学习结果使用缓存 embedding，不得在每条实时群消息上新增 embedding 请求。

**验收：** 全量测试恢复全绿；新增“近义但不应合并”和“近义应合并”两类测试。

### P2-1：系统清理任务重复运行

**证据：**

- `/etc/cron.d/qq-social-agent-hygiene` 在每日 04:17 执行维护。
- 用户 crontab 在每日 04:20 再执行一次相同的 `system_hygiene.sh --apply`。

影响：通常不会损坏数据，但会重复 vacuum/checkpoint、Docker/日志清理和报告写入；长期不必要，并增加两个任务争用 SQLite/WAL 的机会。

修复：保留 `/etc/cron.d/qq-social-agent-hygiene` 作为唯一标准任务，删除用户 crontab 中的重复行；修改后查看两天日志确认每日只运行一次。

### P2-2：SearXNG 在 compose 中声明，但当前未运行

**证据：** `docker compose ps` 只有 bot 和 NapCat；bot 容器无法解析 `searxng` 主机名。

当前搜索仍由 Tavily / RSS / Web fallback 工作，因此不是线上中断。但 SearXNG 不能被宣传为正在使用的搜索主源。

修复选择二选一：

- 启动 SearXNG，确认 health、查询质量、资源占用和 fallback 后再启用；或
- 从生产 compose / 文档中降级为“实验性未部署”，避免维护者误判能力。

### P2-3：群聊主编排仍过度集中

**证据：**

- `qq_social_agent/plugin.py` 约 12090 行。
- `_handle_group_message_locked()` 约 1208 行。
- `_handle_private_message_scoped()` 约 566 行。
- `_handle_group_message_scoped()` 约 432 行。

影响：群聊任一小功能都可能误伤审批、工具、RAG、记忆或发送；回归测试定位困难。

修复：按 [`../module_boundaries.md`](../module_boundaries.md) 的顺序抽取会话编排、审批服务、主动/定时任务和发送协调。每次迁移一段并保留 Trace 事件名和黑盒测试，不做大爆炸重构。

### P2-4：管理端只有网络位置控制

管理端口只绑定 localhost，代码还检查请求来源为 loopback 或 Docker `172.*`，当前公网暴露风险低。残余风险是同 Docker 网络内的任意已入侵容器可访问 WebUI 编辑 Prompt / config。

修复：在不改变 SSH tunnel 使用方式的前提下增加一个独立 admin token 或仅允许明确 bot / SSH proxy 来源。此项应在新增第三方容器、反向代理或公网端口前完成。

### P3-1：中期回想未自动分层

当前约 1118 条 active summary 仍可用，实际注入有预算，不会直接撑爆 Prompt；但它们会持续占用 RAG 候选和索引空间。已在 [`../memory_lifecycle.md`](../memory_lifecycle.md) 记录热/温/冷分层方案。

修复：先做字段、WebUI 可视化和一周小批试跑，禁止直接批量归档全部历史。

### P3-2：少量进程内状态使用惰性淘汰

`addressed_event_times`、部分 lock / cooldown map 在重启前依赖后续同用户事件清理。由于群数和私聊白名单有限，当前内存风险很低；但未来开放更多群或私聊前，应加定期 prune 并记录 map 大小指标。

### P3-3：测试目录不随源码 bind mount

生产容器中的 Python 源码是宿主机实时挂载，`tests/` 不是。因此运行服务器当前代码的全量测试前必须 `docker cp tests/. qq-social-agent-bot:/app/tests/`。该步骤已写进工程运行手册。

## 4. 正向结论

- OneBot 接入、群消息结构化保存、引用关系、RAG、工具路由和 LLM fallback 已形成可观测链路。
- URL 深读实现了实际的 SSRF 保护，不是只靠提示词限制。
- 端口未直接暴露公网；敏感运行数据未出现在 Git 状态中。
- 数据库有 WAL、完整性检查、COS 快照、原文归档和 ntqq 渐进备份，具备一年运行的基础。
- provider 熔断已降低第三方模型超时对群聊热路径的影响。

## 5. 建议执行顺序

1. 让全量测试恢复全绿，先修 P1-1 的两个项目。
2. 删除重复 hygiene cron，观察两天。
3. 明确 SearXNG 是“部署并启用”还是“暂不维护”，避免半配置状态。
4. 按记忆生命周期方案做小批周回忆试跑。
5. 从 `plugin.py` 抽出会话编排服务，并以发送前二次校验作为第一块业务迁移。
6. 在引入任何额外 Docker sidecar 或反向代理前，为 admin WebUI 加二次认证。
