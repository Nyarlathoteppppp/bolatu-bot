# 模块边界与维护手册

最后更新：2026-09-24。

这份文档给在服务器上继续维护张风雪的开发者和 AI 使用。目标是让功能继续增长，但不再把所有事情塞进 `qq_social_agent/plugin.py`。

## 1. 当前结论

项目已经具备真实分层：消息结构化、决策、工具、上下文、记忆、RAG、审批、发送、观测、后台学习和本地插件都有独立模块。`plugin.py` 仍承担 NoneBot 入口、运行时单例装配、群聊编排、私聊阶段依赖装配、会话调度、审批命令和 Web 管理路由。群聊主循环已按话语解析、决策、上下文、工具执行、候选生成和审批交接分段；私聊主循环也已按轮次准备、工具执行、上下文检索和生成发送分段。

策略：**保留 `plugin.py` 作为适配器和组合根，不再向其中放业务规则；新增能力优先落到所属模块，再由主文件显式注册。** 不在缺少回归测试时做一次性大拆分。

## 2. 运行链路与依赖方向

```text
NapCat / OneBot Event
  -> plugin.py：事件适配、ChatMessage 持久化、按会话串行化
  -> message_segments + history_sync + reference_resolver
  -> decision_gate + rate_limiter + buffer
  -> group_discourse_flow：一次解析关系和记忆影响
  -> group_decision_flow + conversation_tool_routing：发言决策与工具计划
  -> group_generation_context：memory/RAG/context packet
  -> group_tool_execution + group_reply_generation：工具证据、候选生成与复核
  -> group_approval_dispatch：审批单和 PipelineState 交接
  -> approval_rules / approval models
  -> delivery + social_actions + onebot_gateway
  -> 私聊入口：private_turn_preparation -> private_tool_execution
     -> private_generation_context -> private_reply_delivery
  -> observability + background_learning + COS/归档
```

依赖只应从入口向下流动：

```text
entrypoint/plugin -> orchestration -> domain/storage/tools -> provider adapters
```

`memory.py`、`onebot_gateway.py`、`deepseek_client.py` 和工具实现都不能反向导入 `plugin.py`。固定人格 Prompt 放在 `prompts/zhangfengxue.yaml`；基于解析结果生成的上下文提示由所属模块格式化。

## 3. 模块地图

| 边界 | 主要文件 | 责任 | 不应承担 |
| --- | --- | --- | --- |
| QQ 接入 | `plugin.py`、`onebot_gateway.py`、`history_sync.py` | OneBot 事件、API、历史同步 | 人格回复判断 |
| 私聊轮次 | `private_message_types.py`、`private_turn_preparation.py` | 合并后的 `PrivateTurn`、去重与审批优先、媒体/语音/OCR/转发上下文、命令处理和用户消息入库 | 工具决策、模型 Prompt 拼装、QQ 回复发送 |
| 私聊工具与上下文 | `private_tool_execution.py`、`private_generation_context.py`、`conversation_tool_routing.py`、`tool_registry.py` | 共享工具路由、工具执行、并行 RAG 与记忆上下文，使用有类型的阶段结果交接 | 原始 OneBot 事件适配、审批状态 |
| 私聊生成与发送 | `private_reply_delivery.py`、`reply_splitter.py`、`meme_library.py` | 私聊模型调用、表情包选择、分段回复、首段引用及成功后的机器人消息入库 | 入口注册、会话 buffer 调度 |
| MessageChain | `message_segments.py`、`reference_resolver.py`、`media_context.py` | 原始 segment、引用/艾特/媒体事实 | 凭文本猜人物关系 |
| 前置筛选 | `decision_gate.py`、`rate_limiter.py` | 去重、低价值、频控、buffer | 社交氛围或搜索词 |
| 社交决策 | `decision_gate.py`、`group_decision_flow.py`、`pipeline_types.py`、`pipeline_stages.py` | channel、action、状态转移 | 最终回复正文 |
| 工具 | `tool_router.py`、`conversation_tool_routing.py`、`group_tool_execution.py`、`tool_registry.py`、`tools/` | 路由、执行、缓存、限流、结构化结果 | 客服式 fallback 文案 |
| 上下文 | `group_discourse_flow.py`、`group_generation_context.py`、`context_assembler.py`、`member_context.py`、`speaker_context.py` | 指代与记忆影响、画像、RAG、预算和输入拼装 | 数据库 schema |
| 候选与审批交接 | `group_reply_generation.py`、`group_approval_dispatch.py` | 草稿生成、复核、审批状态转移 | QQ 实际发送 |
| LLM | `deepseek_client.py`、`embedding_client.py`、`prompts.py` | provider、JSON、模型路由、用量 | QQ 发送与审批状态 |
| 记忆/RAG | `memory.py`、`memory_learning.py`、`rag_*.py` | 原文、画像、atoms、风格、索引 | QQ 生命周期 |
| 发送 | `delivery.py`、`reply_splitter.py`、`social_actions.py` | 拆分、艾特、表情、节流 | 写长期事实 |
| 管理/观测 | `admin_ui.py`、`observability.py`、`approval_rules.py` | WebUI、Trace、工具单 | 聊天热路径判断 |
| 安全输出 | `political_guard.py` | 输出脱敏和明确语义拦截 | 裸匹配消息 ID/QQ/时间戳 |

## 4. 插件化现状

`plugins/*/manifest.yaml` 是**声明式插件化**，不是动态执行插件。manifest 声明工具、命令、定时任务、Web 路由和权限；`LocalPluginRegistry` 读取它们，`plugin.py` 仍通过 allowlist 显式绑定本地实现。

新增插件必须：

1. 先加 manifest capability 和 permission。
2. 实现落在明确领域模块，不写进 manifest，也不把业务塞回主文件。
3. 注册前检查 capability 是否启用。
4. 让 `/admin/plugins` 和工具单可见状态。
5. 外部网络工具都具备 timeout、缓存、限流、Trace 和 `ToolResult` 失败结果。

## 5. 当前模块化评价

做得好的部分：

- `ToolRegistry` 已统一搜索、行情、网页读取的执行边界。
- `PipelineState` 已覆盖一次处理的阶段、决策、上下文、候选、审批和失败原因。
- MessageChain、引用/艾特解析和媒体上下文已从纯文本中分离。
- 后台学习与 RAG 索引不阻塞聊天热路径。
- WebUI 渲染已在 `admin_ui.py` 中独立。
- 54 个测试文件覆盖大多数领域模块。

主要技术债：

- `plugin.py` 同时承载生命周期、scheduler、审批、私聊/群聊编排、HTTP controller 和发送协调。
- `memory.py`、`deepseek_client.py`、`rag_store.py` 仍大，但数据契约密集，暂不宜粗暴拆分。
- manifest 能声明能力，但实际 handler 注册仍集中在主文件。

## 6. 下一步拆分顺序

群聊高频改动路径已按阶段拆出。`plugin.py` 装配当前 LLM、存储、日志和指标回调，并保留事件生命周期与阶段间早退。后续继续按功能边界迁移，保持 Trace 事件名与数据库写入不变。

1. **审批服务**：抽成 `approval_service.py`，负责待审批单、并发抢占、权限、反馈和候选发送。
2. **定时任务**：抽 `daily_review_service.py` 与 `proactive_chat_service.py`，连接回调只保留启动/停止 task。
3. **会话编排**：逐步整理 buffer、processing lock 和 followup window；群聊阶段函数已独立，私聊主流程尚未拆分。
4. **管理 controller**：抽 `admin_controller.py`，让 `admin_ui.py` 只做 HTML 渲染。
5. **发送协调**：抽 `message_delivery_service.py`，集中正文、表情、审批回写和 followup 记录。

不要优先拆 `memory.py` 或 `deepseek_client.py`。先补 repository/service 边界和表级测试，再动内部结构。

## 7. 数据与部署边界

热数据：SQLite。冷备份和历史归档：COS。不得提交 Git 的文件包括 `.env`、`data/`、`server-data/`、NapCat 登录态和私有表情包。

生产 compose 对源码、Prompt、config、plugins 和数据均使用 bind mount：

- Python、Prompt、普通 config 修改：`docker compose -p qq-social-agent -f docker-compose.server.yml up -d --no-deps --force-recreate bot`
- 依赖、Dockerfile 或镜像环境修改：`docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build --no-deps bot`
- 不要为后端代码修改重启 Docker daemon、执行 `compose down` 或重启 NapCat。这些会影响 QQ 登录态。
- 新 sidecar 如 SearXNG 只能单独 `up -d searxng`，先验证健康和网络，再切换 provider。

## 8. 维护检查单

```bash
cd /opt/qq-social-agent
git diff --check
PYTHONPATH=. python3 -m pytest -q <相关测试>
curl -fsS http://127.0.0.1:8080/healthz
git status --short
```

`/readyz` 还要求 OneBot 已连接，NapCat 未扫码时的 503 不代表 Python 后端错误。

当前已知事项：

- SearXNG 是 MVP。镜像或服务未健康时保留 Tavily/RSS fallback，不要强制切主 provider。
- 政治保护只能检查可见正文、引用和明确语义；消息 ID、QQ 号、时间戳必须剥离。
- 新功能先归属领域模块，`plugin.py` 仅负责装配和入口编排。
