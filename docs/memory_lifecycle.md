# 群聊记忆生命周期与分层方案

最后更新：2026-08-28  
状态：**设计已确认，尚未改动运行逻辑。**

本文档定义群聊长期记忆的存储边界、召回规则和未来迁移步骤。它面向后续接手项目的 AI / 开发者；修改 `memory.py`、RAG 或后台学习任务前必须先读本文档。

相关文档：

- [`../AI_PROJECT_GUIDE.md`](../AI_PROJECT_GUIDE.md)：完整消息流程和部署边界。
- [`group_chat_evolution.md`](group_chat_evolution.md)：群聊体验的其他演进项。
- [`storage_maintenance.md`](storage_maintenance.md)：SQLite、COS 与磁盘维护。

## 1. 当前机制

### 1.1 原始事实层

`messages` 是群聊消息的原始事实来源。每条消息保存发言人、昵称、时间、文本、OneBot MessageChain、原始事件和引用关系。日聊天原文会归档到 COS，因此 memory summary 不是唯一副本。

### 1.2 中期回想层

后台任务 `_maintain_group_learning()` 定期检查尚未总结的群消息：

- 至少有 24 条可用群友消息时才会调用 `mid_memory`。
- 单次最多处理 60 条消息，机器人自身发言不参与这份总结。
- `memory_summary_state.last_message_id` 标记已总结边界，防止同一段反复总结。
- 模型输出的 `summary` 和最多 5 个 `recall_cues` 写入 `memory_summaries`。
- 新 summary 默认 `status=active`、`locked=0`。

截至 2026-08-28，主群的 active summary 约 1105 条。当前问题不是它们会一次性塞进 prompt，而是所有历史段落都没有自动生命周期。

### 1.3 结构化长期事实层

同一次中期总结可以提取 `memory_atoms`：人物关系、身份、偏好、群内黑话、稳定事件等。这类内容与普通聊天摘要不同，应该长期保存，并且带来源、置信度、状态和人工纠错能力。

**原则：**

- 人物关系、黑话、明确偏好优先写入 `memory_atoms`。
- 普通对话气氛、临时玩笑、当天话题优先留在 `memory_summaries`。
- 不能把一句反串、互损或临时玩笑直接升级为长期人物事实。

### 1.4 当前召回

生成回复时，不会将全部 summary 注入模型：

- 混合 RAG 可在活动证据中召回少量相关原话、摘要、画像和 memory atom。
- RAG 无结果时，`relevant_memory_summaries()` 仅从最近 80 条 active summary 中按 `summary + recall_cues` 的文本相关度选取最多 4 条。
- 当前现场的最近群聊、引用链和发言人关系永远优先于旧回忆。

因此 1105 条不会直接造成 prompt 爆炸；真正风险是索引和候选集长期变大，以及旧碎片 summary 逐渐难以被正确管理。

## 2. 目标分层

### 热记忆：最近 14 天

- 保留每 24-60 条消息形成的细粒度 summary。
- `status=active`，正常参与 RAG 和兜底文本召回。
- 用于正在延续的群内事件、最近梗、临时关系变化和近期对话。

### 温记忆：14-90 天

- 以“群 + 自然周”为单位，将完整周内的细粒度 summary 压成一条周回忆。
- 周回忆只保留长期有价值内容：主要事件、稳定关系变化、延续梗、尚未结束的话题。
- 对应的细粒度 summary 标记为 `archived`，不再参与普通 RAG；周回忆替代它参与召回。
- 原细粒度 summary 不删除，仍可在 WebUI 查看、编辑、恢复。

### 冷记忆：90 天以上

- 细粒度 summary 继续保留于 SQLite，但标记 `archived`，并从活动 RAG 索引中移除。
- 每月保留一条月回忆供普通检索。
- 原始消息仍在 `messages` 与 COS 日归档中；确实需要追溯时，按日期、人物或关键词进行深度召回。

### 永久事实：memory atoms

- 不按 14/90 天自动过期。
- 必须可追溯来源；带 `locked` 的内容绝不自动改写、归档或删除。
- 若事实被反证，应标记过期或降置信度，而不是覆盖性改写历史。

## 3. 拟议数据模型

优先扩展现有 `memory_summaries`，不要新建第二套几乎相同的表。建议增加：

```text
summary_kind             segment / weekly_rollup / monthly_rollup
tier                     hot / warm / cold
rollup_bucket            例如 1026813421:2026-W35
source_summary_ids_json  被压缩的子 summary ID 列表
last_recalled_at         上次被实际注入上下文的时间
recall_count             实际召回次数
```

`status` 继续负责人工状态：`active`、`archived`、`expired`；`locked=1` 表示任何自动任务不得更改它。

RAG 文档必须保留 source summary ID：归档 segment 时使对应 RAG document 失活并移除 embedding；生成 rollup 时创建新的活动 RAG document。这样数据库保留原文，索引只保留真正需要的层级。

## 4. 后台迁移任务

任务应每天低峰运行一次，不进入实时聊天热路径。建议运行时点为北京时间凌晨，并避开现有 COS 备份窗口。

每个群依次执行：

1. 找出超过 14 天、尚未进入周回忆的完整自然周。
2. 读取该周 active segment summary；任何 locked summary 直接跳过，不能混入自动归档。
3. 调用一次 `memory_rollup` 流程生成周回忆，明确保留人物归属和不确定性。
4. 校验周回忆非空、结构可解析、来源列表完整。
5. 在单个 SQLite 事务内写入周回忆，再将来源 segment 改为 `archived`。
6. 异步更新 RAG：新周回忆入索引，已归档 segment 出活动索引。
7. 超过 90 天的完整周再进入月回忆；月回忆成功后，将周回忆改为 `archived`。

任务必须幂等：同一个 `rollup_bucket` 已存在时不再生成；模型超时、输出为空或校验失败时不改变任何来源数据。

## 5. 召回预算

普通群聊回复建议保持小上下文：

```text
短期现场消息：按现有 context_limit
热记忆：最多 4 条
温记忆：最多 1 条，且必须强相关
冷记忆：默认 0 条
memory atoms：按现有独立预算
```

只有在消息明确询问旧事件、旧人物关系或具体日期时，才允许触发“深度回忆”：先检索月/周回忆，再按来源 ID 打开 archived segment。不能让普通闲聊为了“可能有关”扫描冷记忆。

## 6. WebUI 与人工纠错

记忆审计页应支持：

- 按 tier、状态、群、日期和人物筛选。
- 查看 rollup 的来源 summary 和对应消息范围。
- 锁定、解锁、编辑、归档、恢复、过期。
- 一键将错误 summary 标为 `expired`，同时使对应 RAG 文档失活。
- 对单条 archived segment 执行“恢复热记忆并重新索引”。

中文 WebUI 是主要人工入口。不要依赖 QQ 私聊中的英文审批命令来维护记忆。

## 7. 不做的事

- 不删除原始聊天消息。
- 不把 COS 当成实时聊天数据库。
- 不让每条群消息多调用一次 LLM 来判断记忆层级。
- 不自动触碰 locked summary 或 locked atom。
- 不把审批“准奏/不准奏”当作当前群聊学习的必要输入；它只保留为可选人工反馈。

## 8. 实施顺序

1. 给 `memory_summaries` 增加层级字段和索引，默认所有既有记录为 `segment + hot`，不改变召回。
2. WebUI 增加 tier、来源和恢复操作，先让人工可看可控。
3. 实现周回忆生成与 RAG 索引切换，仅针对已超过 14 天的历史做一次小批量试跑。
4. 观察 7 天：核对人名归属、RAG 命中、Token 和延迟。
5. 再启用 90 天月回忆和长期自动调度。

任何阶段出现人物错位、来源缺失或 RAG 召回下降，应停止自动迁移，保留已生成 rollup，恢复来源 segment 为 active。
