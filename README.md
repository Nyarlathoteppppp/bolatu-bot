# 张风雪 / QQ Social Agent

一个运行在 Ubuntu 服务器上的 QQ 群聊社交机器人。它不是问答客服，而是一个有固定人格、会看群聊氛围、知道何时插话和何时沉默的“群友型” Agent。

QQ 由 NapCat 接入，NoneBot2 接收 OneBot v11 事件；消息筛选、频率控制、记忆、RAG、审批、工具路由和运行监控都在本项目后端完成。LLM 只承担真正需要语义判断和表达生成的部分。

> 生产运行数据、QQ 登录态、密钥和群聊内容均不在仓库中。请先阅读“安全与数据”再部署。

## 能做什么

- **像群友一样聊天**：先判断“现在插嘴有没有意思”，再决定普通接话、回答、认可、关心、吐槽、反问、艾特他人、点表情或沉默。
- **理解上下文关系**：完整保存 OneBot MessageChain、引用、艾特、图片、转发、语音和原始事件，尽量分清“谁在回复谁”“这句你在指谁”。
- **有可审计的长期记忆**：保存原始群聊语料、阶段回想、群友画像、关系/偏好记忆、群内黑话、审批反馈与撤回反馈；所有长期记忆均带来源、状态和人工纠错入口。
- **混合 RAG 检索**：SQLite FTS5 关键词检索与 `BAAI/bge-m3` embedding 语义检索并用，按群隔离召回历史原话、回想、画像和知识库内容。
- **会用工具，但不乱搜**：模型在需要新信息时选择搜索、深度网页读取、美股/ETF/加密货币行情查询；工具结果先压缩为背景，再交给回复模型自然接话。
- **支持图片和富媒体上下文**：图片可走 OCR/视觉摘要；文件、语音、转发和表情保留结构化安全元数据，避免机器人“看不见还硬接”。
- **可人工审批，也可自动发送**：候选回复可以私聊审批人选择；审查开启时可配置一部分消息直接发送，减少人工负担。
- **会主动但不刷屏**：支持按时间段概率主动聊天、回复后的 90 秒续聊窗口、每日北京时间 24:00 复盘。
- **可维护地长期运行**：包含 Web 管理台、链路 Trace、Token/模型用量、RAG 评测、SQLite 体检、Docker 清理和腾讯云 COS 冷备份。

## 架构

```text
QQ 客户端
  -> NapCatQQ
  -> OneBot v11 Reverse WebSocket
  -> NoneBot2 / qq_social_agent
  -> 消息链保存、buffer、频控、工作强度抽样、引用/指代解析
  -> decision + timing gate（是否值得插话、采取什么动作）
  -> tool router（搜索 / 行情 / 网页 / 无工具）
  -> 记忆、RAG、画像、黑话、原文语料、风格规则组装上下文
  -> reply / search_answer 生成
  -> 审批或直发
  -> SQLite 指标、记忆、RAG 和 COS 备份
```

核心原则：

1. `decision` 只负责社交判断，不负责搜索词、股票代码或回复正文。
2. 搜索、行情、网页读取由独立工具路由决定；失败信息进入回复上下文，不使用生硬的后端兜底台词。
3. 记忆不是“一坨摘要”：原文、结构化记忆、画像、黑话、风格规则和 RAG 各自保存、各自召回。
4. 普通消息优先经过后端的确定性筛选，减少无意义 LLM 调用；点名、引用和续聊信号会放宽进入决策的条件。

## 技术组成

| 模块 | 作用 |
| --- | --- |
| NapCatQQ + OneBot v11 | QQ 登录、收发消息、群历史、引用、图片等事件接入 |
| NoneBot2 | 事件入口和 Web 服务 |
| Python + SQLite WAL | 消息、审批、记忆、画像、指标、用量和 RAG 索引 |
| SiliconFlow / DeepSeek | 多 provider LLM 路由与 fallback；当前默认模型以 `config.yaml` 为准 |
| Tavily + RSS fallback | 最新新闻和网页背景搜索 |
| FTS5 + bge-m3 + NumPy | 混合 RAG 和向量化语义检索 |
| Docker Compose | 服务器部署与进程隔离 |
| 腾讯云 COS | SQLite 快照、配置、聊天归档和冷备份 |

## 项目结构

```text
.
├── bot.py                         # NoneBot 入口
├── config.yaml                    # 群白名单、模型、频控、工具和定时任务
├── prompts/zhangfengxue.yaml      # 人格、决策、回复、记忆、画像、学习 Prompt
├── qq_social_agent/
│   ├── plugin.py                  # 主流程编排和事件入口
│   ├── pipeline_stages.py         # 消息处理阶段
│   ├── decision_gate.py           # 社交决策与 Timing Gate
│   ├── tool_router.py             # 搜索/行情/网页工具路由
│   ├── memory.py                  # SQLite 记忆、反馈、画像和指标
│   ├── rag_retriever.py           # 混合 RAG 检索
│   ├── message_segments.py        # MessageChain 结构化保存
│   ├── reference_resolver.py      # 引用、艾特、指代关系解析
│   ├── media_context.py           # 图片、文件、语音上下文
│   ├── deepseek_client.py         # 多模型 LLM 客户端
│   └── admin_ui.py                # 本地 Web 管理台
├── plugins/                       # 搜索、行情、RAG、备份、主动聊天等本地插件
├── scripts/                       # 启动、体检、备份、归档、维护脚本
├── tests/                         # 单元和集成测试
├── AI_PROJECT_GUIDE.md            # 给后续开发者/AI 的工程交接文档
└── SERVER_DEPLOY.md               # 服务器部署细节
```

## 快速部署

### 1. 准备环境变量

```bash
cd /opt/qq-social-agent
cp .env.example .env
```

按 `.env.example` 填写 LLM、搜索等环境变量。真实密钥只放 `.env` 或服务器的受限环境文件，绝不能提交到 Git。

### 2. 配置机器人

主要修改点：

- `config.yaml`：目标群、私聊白名单、模型路由、频控、主动聊天、搜索和 RAG。
- `prompts/zhangfengxue.yaml`：人格、决策、工具选择、回复、风格学习、记忆压缩和每日复盘。
- `plugins/*/manifest.yaml`：本地插件能力和开关。

NapCat 的 OneBot v11 反向 WebSocket 地址应为：

```text
ws://bot:8080/onebot/v11/ws
```

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
docker restart qq-social-agent-bot
```

只有修改 `pyproject.toml`、`Dockerfile.server`、系统依赖或镜像基础环境时，才需要：

```bash
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build --no-deps bot
```

不要为普通代码修改重启 NapCat。QQ 登录态可能失效或触发风控；只有 QQ 内核网络异常、账号掉线或登录状态损坏时才重启它。

查看日志：

```bash
docker logs -f qq-social-agent-bot
docker logs -f napcat
```

运行测试：

```bash
python -m pytest -q
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

## 相关文档

- [AI_PROJECT_GUIDE.md](AI_PROJECT_GUIDE.md)：完整链路、模块边界、Prompt 流程和排障说明。
- [SERVER_DEPLOY.md](SERVER_DEPLOY.md)：Docker、分支、NapCat 和服务器操作。
- [docs/storage_maintenance.md](docs/storage_maintenance.md)：数据库、COS 和一年期运行维护策略。
