# 张风雪 / QQ Social Agent

一个运行在 Ubuntu 服务器上的 QQ 群聊社交机器人。它不是问答客服，而是一个有固定人格、会看群聊氛围、知道何时插话和何时沉默的“群友型” Agent。

QQ 由 NapCat 接入，NoneBot2 接收 OneBot v11 事件；消息筛选、频率控制、记忆、RAG、审批、工具路由和运行监控都在本项目后端完成。LLM 只承担真正需要语义判断和表达生成的部分。

> 生产运行数据、QQ 登录态、密钥和群聊内容均不在仓库中。请先阅读“安全与数据”再部署。

## 文档阅读顺序

- [工程交接文档](AI_PROJECT_GUIDE.md)：运行、配置、Prompt、数据与常见排障。
- [工程运行手册](docs/engineering_runbook.md)：生产变更、测试、发布、巡检、备份恢复和故障处理规范。
- [最新项目审计](docs/audits/2026-08-28-project-audit.md)：2026-08-28 基线；其后搜索、记忆和政治守卫有独立提交，以本 README 和 git log 为准。
- [模块边界与维护手册](docs/module_boundaries.md)：当前模块职责、依赖方向、插件化现状与重构策略。
- [群聊质量演进路线](docs/group_chat_evolution.md)：发送前校验、短期对话线、话题冷却和可观测性等后续框架。
- [记忆生命周期与分层方案](docs/memory_lifecycle.md)：中期回想、memory atoms、RAG 召回与长期归档。
- [服务器部署说明](SERVER_DEPLOY.md)：Docker、SSH tunnel 和发布边界。
- [存储维护策略](docs/storage_maintenance.md)：SQLite、日志、COS 归档和容量巡检。

## 能做什么

- **像群友一样聊天**：先判断“现在插嘴有没有意思”，再决定普通接话、回答、认可、关心、吐槽、反问、艾特他人、点表情或沉默。连续缓冲消息只对最后触发者说话，不会把两个人认成同一个。
- **理解上下文关系**：完整保存 OneBot MessageChain、引用、艾特、图片、转发、语音和原始事件，尽量分清“谁在回复谁”“这句你在指谁”。
- **有可审计的长期记忆**：保存原始群聊语料、阶段回想、群友画像、关系/偏好记忆、群内黑话、审批反馈与撤回反馈；长期记忆带来源、状态和人工纠错入口。Builtin / 人工身份事实不会被中期摘要覆盖。
- **混合 RAG 检索**：SQLite FTS5 关键词检索与 `BAAI/bge-m3` embedding 语义检索并用，按群隔离召回历史原话。群聊生成时 RAG 与阶段回想合并，不再整段换掉中期摘要。
- **会搜，但不乱搜**：点名“搜一下”或回复需要最新事实时，走 SearXNG → Tavily / Bing 回退；成功后只从**本轮结果 URL**并行读最多 2 页。维基走 MediaWiki extracts API，不刮 HTML。禁止把空题目扩成热榜/排行。
- **支持图片和富媒体上下文**：图片可走 OCR/视觉摘要；文件、语音、转发和表情保留结构化安全元数据。
- **可人工审批，也可自动发送**：候选回复可以私聊审批人选择；审查开启时可配置一部分消息直接发送。
- **会主动但不刷屏**：按时间段概率主动聊天，话题来自静态词库并有冷却；主动聊不调搜索。群聊每日复盘当前关闭。
- **可维护地长期运行**：Web 管理台、链路 Trace、Token/模型用量、RAG 评测、SQLite 体检、Docker 清理和腾讯云 COS 冷备份。

## 架构

项目是“后端编排 + 小模型判断 + 大模型表达”的分层 Agent，而不是把所有聊天记录塞给一个模型直接回复。也不做通用 ReAct / 多轮搜索循环。

```text
QQ 客户端
  -> NapCatQQ
  -> OneBot v11 Reverse WebSocket   ws://bot:8080/onebot/v11/ws
  -> 接入层：MessageChain、原始事件、发言人和引用关系持久化
  -> 前置筛选：去重、buffer、频控、工作强度、禁言和低价值消息拦截
  -> 社交决策：是否插话 + social action
  -> 工具决策：搜索 / 行情 / 网页读取 / 不调用
  -> 搜索一跳：SearXNG（约 1.5s）→ Tavily / Bing；读本轮 URL（最多 2 成功 / 4 尝试）
  -> 上下文装配：短期群聊 + 引用链 + 记忆 + RAG + 画像 + 黑话 + 风格
  -> 表达生成：候选回复 / 直接回复 / 工具结果短评
  -> 发送层：人工审批或自动发送，附加表情/艾特；输出政治词打码
  -> 学习与观测：反馈、记忆、画像、风格、RAG、用量、Trace、备份
```

三个 Compose 服务必须同属项目名 `qq-social-agent`，否则 NapCat 找不到 `bot`：

| 容器 | 作用 | 改代码后 |
| --- | --- | --- |
| `qq-social-agent-bot` | Python 后端，bind mount 源码 | 只重启这个 |
| `qq-social-agent-searxng` | 内网搜索 `http://searxng:8080` | 一般不动 |
| `napcat` | QQ 登录 | **不要为普通代码重启** |

### 消息处理流程

#### 1. 接入与结构化保存：不靠模型猜消息关系

NapCat 把 QQ 群聊、私聊、引用、艾特、图片、语音、转发和通知事件通过 OneBot 推送给 NoneBot。后端先将原始 MessageChain、`message_id`、发言人、昵称、reply target 和发送时间写入 SQLite。

`message_segments.py`、`reference_resolver.py` 和 `message_segments_json` 共同解决“谁在回复谁”“这句你是在叫机器人还是在叫别人”的问题。模型看到的是经过整理的关系摘要，例如：

```text
A 说：……
B 回复 A：……
当前发言由 C 发出，未直接回复张风雪。
```

#### 2. 前置筛选与缓冲：把无意义调用挡在 LLM 之外

每条群消息先经过确定性后端逻辑：持久去重、消息 buffer、工作强度概率、群/用户频控、禁言状态、低价值短句、图片/富媒体可见性和连续回复保护。

这层的目标是降低 Token 和避免刷屏，不做人格表达判断。明确艾特、引用机器人、有效续聊窗口等强信号会绕过部分保守筛选，保证被叫到时不会像“掉线”。

#### 3. 社交决策：先判断值不值得插话

通过 `decision` + `timing gate`，LLM 输出结构化结果：

```text
should_reply: true / false
action: reply / answer / agree / care / tease / ask_back / at_someone / react / ignore
```

决策层只负责当前聊天局面的社会判断。它不生成最终正文，也不处理工具参数。

#### 4. 工具路由：需要新事实时才查

社交决策允许回复后，`decision_gate` 对点名“搜一下”可直接要求新鲜检索；否则 `tool_router` 再判断是否需要 `fresh_search`、`market_lookup`、`deep_url_reader`。近旁 URL 提示只看**当前句**，避免上一句的 GitHub 链接把闲聊带去读页。

搜索一跳（不是通用 Agent 循环）：

1. 压缩查询词；禁止把空题目扩成热榜/热门话题/排行。
2. SearXNG 硬顶约 1.5s，失败再走 Tavily / Bing。
3. 按 query 重叠和站点权重排序（维基 / GitHub / gov·edu / 国内新闻 / 知乎专栏问答）。
4. 并行读最多 2 个成功页，总尝试 ≤ 4；维基走 API extracts。
5. 知乎 hub 跳过，专栏/问答可试读；登录墙或 403 当没读到，只用本轮摘要。
6. 定义类问句且第一跳没正文时，可再搜一次「… 维基百科」。
7. `search_answer` 正文优先；有摘要就不说“没查到 / 你自己搜”。单条搜索回答不被最近机器人发言滤空。

工具返回的是可核查的背景，不是给群里播报的模板。可以自然提一个来源名；群友问出处再给链接。

#### 5. 上下文装配与回复生成：分层召回，不让旧记忆压过现场

只有确定要回复时才装配完整上下文：

- 最近短期群聊和当前引用链。
- 相关的 memory atoms、阶段回想和群友画像。
- 按群隔离的黑话、人工反馈和风格规则。
- 原文语料库中少量相关群友原话，只参考语气，不允许照搬。
- FTS5 + `bge-m3` 混合 RAG 召回的历史对话；与阶段回想拼在同一 `memory` 段，RAG 不再覆盖摘要。
- 搜索、行情或网页读取得到的最新背景。

`CHAT` 模式保留会话记忆。`SEARCH` / `MARKET` / `DEEP_URL` 只留匹配到的黑话，避免旧聊天盖住新网页。

根据场景选择 `reply_candidates`、`reply_direct` 或 `search_answer`。审查开启时生成多个候选交给审批人；直发时生成一个更完整的单候选。

#### 6. 发送、续聊与自我学习

发送层会执行文本拆分、输出政治词打码、艾特/表情限频和 OneBot 发信。输入侧不再用固定“冲塔”套话拦截；打码只处理明确整词（例如姓名/组织名缩写），避免把日常「64」「连任」误伤掉。

发送成功后打开续聊窗口。后台任务不阻塞热路径：

- 中期记忆总结和 memory atoms 提取；空摘要连续失败后跳过该窗口，避免卡住游标。
- 到期 atom 在学习循环中过期，不必等进程重启。
- 中期学习不会覆盖 `builtin_` / `manual` 身份事实；`jargon_candidate` 不进默认 atom 槽。
- 群友画像、风格规则、RAG 索引、主动聊天和 COS 归档。
- Token、模型、工具、拦截和审批的 Trace/指标写入数据库。

### LLM 与后端的边界

| 工作 | 主要承担者 | 原因 |
| --- | --- | --- |
| OneBot 解析、持久去重、buffer、频控、限流、审批状态 | 后端代码 | 可预测、低成本、可审计 |
| 谁回复谁、是否在说机器人 | 结构化消息 + 本地解析 | 保留事实，不让模型从纯文本瞎猜 |
| 是否插话、采用什么社交动作 | decision LLM | 需要理解群聊氛围和人格 |
| 是否查询、如何提取查询意图 | 本地硬路由 + tool_router LLM + 后端校验 | 点名搜索要稳，参数/限流必须可控 |
| 回复措辞、搜索后的短评 | reply / search_answer LLM | 需要自然表达和人格一致性 |
| 记忆、画像、风格规则 | 后台 LLM + SQLite 合并/淘汰 | LLM 负责提炼，后端负责证据和生命周期 |

核心原则：

1. `decision` 只负责社交判断，不负责搜索词、股票代码或回复正文。
2. 搜索、行情、网页读取由独立工具路由决定；失败信息进入回复上下文，不使用生硬的后端兜底台词。
3. 记忆不是“一坨摘要”：原文、结构化记忆、画像、黑话、风格规则和 RAG 各自保存、各自召回。
4. 普通消息优先经过后端的确定性筛选；点名、引用和续聊信号会放宽进入决策的条件。
5. 不做通用 ReAct。补读只来自本轮搜索 URL。

## 技术组成

| 模块 | 作用 |
| --- | --- |
| NapCatQQ + OneBot v11 | QQ 登录、收发消息、群历史、引用、图片等事件接入 |
| NoneBot2 | 事件入口和 Web 服务 |
| Python + SQLite WAL | 消息、审批、记忆、画像、指标、用量和 RAG 索引 |
| SiliconFlow / DeepSeek | 多 provider LLM 路由与 fallback；当前默认模型以 `config.yaml` 为准 |
| SearXNG + Tavily / Bing | 主搜索与回退；读页走本轮 URL，维基走 MediaWiki API |
| FTS5 + bge-m3 + NumPy | 混合 RAG 和向量化语义检索 |
| Docker Compose | bot / SearXNG / NapCat 三容器 |
| 腾讯云 COS | SQLite 快照、配置、聊天归档和冷备份 |

## 项目结构

```text
.
├── bot.py                         # NoneBot 入口
├── config.yaml                    # 群白名单、模型、频控、工具和定时任务
├── docker-compose.server.yml      # bot + searxng + napcat
├── prompts/zhangfengxue.yaml      # 人格、决策、回复、记忆、画像、学习 Prompt
├── qq_social_agent/
│   ├── plugin.py                  # 主流程编排和事件入口
│   ├── pipeline_stages.py         # 消息处理阶段
│   ├── decision_gate.py           # 点名/搜索等本地硬路由 + Timing Gate
│   ├── tool_router.py             # 搜索/行情/网页工具路由
│   ├── memory.py                  # SQLite 记忆、反馈、画像和指标
│   ├── memory_learning.py         # 中期摘要落地事实
│   ├── context_assembler.py       # CHAT vs 搜索的记忆预算
│   ├── rag_retriever.py           # 混合 RAG 检索
│   ├── political_guard.py         # 输出打码（输入不再硬拦）
│   ├── message_segments.py        # MessageChain 结构化保存
│   ├── reference_resolver.py      # 引用、艾特、指代关系解析
│   ├── media_context.py           # 图片、文件、语音上下文
│   ├── deepseek_client.py         # 多模型 LLM 客户端
│   ├── admin_ui.py                # 本地 Web 管理台
│   └── tools/
│       ├── fresh_context.py       # 搜索 + 本轮读页
│       ├── deep_content.py        # 深读
│       └── safe_url_reader.py     # SSRF/超时读页
├── plugins/                       # 搜索、行情、RAG、备份、主动聊天等本地插件
├── scripts/                       # 启动、体检、备份、归档、维护脚本
├── tests/                         # 单元和集成测试
├── AI_PROJECT_GUIDE.md            # 给后续开发者/AI 的工程交接文档
└── SERVER_DEPLOY.md               # 服务器部署细节
```

不要提交：`.env`、`data/`、`logs/`、`server-data/`、`data/meme_library/`。

## 快速部署

### 1. 准备环境变量

```bash
cd /opt/qq-social-agent
cp .env.example .env
```

按 `.env.example` 填写 LLM、搜索等环境变量。真实密钥只放 `.env` 或服务器的受限环境文件，绝不能提交到 Git。

### 2. 配置机器人

主要修改点：

- `config.yaml`：目标群、私聊白名单、模型路由、频控、主动聊天、搜索和 RAG。`fresh_search.provider` 默认 `searxng`。
- `prompts/zhangfengxue.yaml`：人格、决策、工具选择、回复、风格学习、记忆压缩。
- `plugins/*/manifest.yaml`：本地插件能力和开关。

NapCat 的 OneBot v11 反向 WebSocket 地址应为：

```text
ws://bot:8080/onebot/v11/ws
```

Compose 项目名必须是 `qq-social-agent`。

### 3. 启动

```bash
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build
```

检查状态：

```bash
curl -fsS http://127.0.0.1:8080/readyz
docker compose -p qq-social-agent -f docker-compose.server.yml ps
```

返回结果中的 `database_ready`、`llm_ready` 和 `onebot_ready` 均为 `true`，才表示机器人可以正常收发消息。

## 日常开发与更新

生产 Docker Compose 将源码、Prompt、配置和数据以 bind mount 挂进 bot 容器。因此普通 Python、Prompt、配置和插件改动不需要重建镜像：

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml restart bot
curl -fsS http://127.0.0.1:8080/readyz
```

只有修改 `pyproject.toml`、`Dockerfile.server`、系统依赖或镜像基础环境时，才需要：

```bash
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build --no-deps bot
```

不要为普通代码修改重启 NapCat。QQ 登录态可能失效或触发风控；只有 QQ 内核网络异常、账号掉线或登录状态损坏时才重启它。

查看日志：

```bash
docker logs -f qq-social-agent-bot
docker logs --since 20m qq-social-agent-bot 2>&1 | grep -E 'group decision|fresh context|empty_model_reply|skipped group reply'
```

常见空回复：`empty_model_reply_direct_single` 表示生成后候选被滤光，不是没搜到。

生产镜像通常不带 pytest。不要往正在跑的 bot 容器 `pip install`。一次性容器示例：

```bash
docker run --rm \
  -v /opt/qq-social-agent/qq_social_agent:/app/qq_social_agent \
  -v /opt/qq-social-agent/tests:/app/tests \
  -v /opt/qq-social-agent/prompts:/app/prompts \
  -w /app qq-social-agent-bot \
  python -c 'import subprocess,sys; subprocess.check_call([sys.executable,"-m","pip","install","-q","pytest","anyio"]); raise SystemExit(subprocess.call([sys.executable,"-m","pytest","tests/test_fresh_context.py","tests/test_memory_learning.py","-q","--tb=short"]))'
```

## 管理与调试

机器人提供两套操作面：

- **QQ 私聊工具单**：审批候选、开关群聊/审批、调整工作强度与模型、查看 Token、拦截原因、RAG、记忆和黑话。
- **本地 Web 管理台**：`/admin` 可查看审批队列、近期决策、链路 Trace、RAG 命中、记忆审计、画像、风格规则、模型状态和插件。

服务默认只绑定本机。远程访问 NapCat WebUI 或管理台请走 SSH tunnel，不要直接暴露到公网：

```bash
ssh -L 6099:127.0.0.1:6099 -L 8080:127.0.0.1:8080 qqbot-server
```

然后访问：

```text
http://127.0.0.1:6099/webui
http://127.0.0.1:8080/admin
```

## 长期运行与备份

这个项目的长期瓶颈通常是磁盘而不是 CPU/RAM：SQLite 数据库、NapCat 图片/媒体缓存、日志、Docker 构建缓存会持续增长。

- `scripts/db_hygiene.py`：检查 SQLite 完整性并清理失效 RAG/WAL。
- `scripts/system_hygiene.sh --dry-run`：预览维护动作。
- `scripts/system_hygiene.sh --apply`：清理过期备份、Docker cache、NapCat 临时文件和日志。
- `scripts/cos_backup.sh --apply`：上传 SQLite 快照、配置和归档到 COS。
- `scripts/daily_history_archive.sh`：按日归档聊天原文与历史回忆库。

COS 是冷备份，不是 SQLite 的实时存储盘。运行数据库必须留在服务器本地 SSD。

## 安全与数据

- 不提交 `.env`、`data/`、`logs/`、`server-data/`、COS 配置或 QQ 登录态。
- 不把 NapCat WebUI、OneBot 端口或 `/admin` 暴露到公网。
- 对数据库做操作前先保留 SQLite 快照；记忆和 RAG 采用状态标记，优先过期/归档而不是物理删除。
- 生产数据含真实群聊内容、昵称、QQ 号和审批记录，调试截图与日志脱敏后再分享。
- 不要在公开文档里写生产 IP、密钥或具体群号。

## 相关文档

- [AI_PROJECT_GUIDE.md](AI_PROJECT_GUIDE.md)：完整链路、模块边界、Prompt 流程和排障说明。
- [SERVER_DEPLOY.md](SERVER_DEPLOY.md)：Docker、分支、NapCat 和服务器操作。
- [docs/storage_maintenance.md](docs/storage_maintenance.md)：数据库、COS 和一年期运行维护策略。
- [docs/memory_lifecycle.md](docs/memory_lifecycle.md)：记忆分层；实现以 `memory.py` / `memory_learning.py` / `context_assembler.py` 为准。
