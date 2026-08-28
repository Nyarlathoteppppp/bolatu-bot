# 工程运行手册

最后更新：2026-08-28  
适用环境：腾讯云服务器 `/opt/qq-social-agent`  
维护原则：**服务器是生产环境；GitHub `main` 是代码与文档真源；SQLite / COS 是运行数据真源。**

本文档供长期维护本项目的开发者和 AI 使用。它定义变更流程、验证标准、数据边界、日常巡检和故障处理。阅读本文件不能替代 [`../AI_PROJECT_GUIDE.md`](../AI_PROJECT_GUIDE.md)，但任何生产改动都应遵守本文件。

## 1. 文档地图

| 需求 | 先读 |
| --- | --- |
| 了解消息、模型、Prompt 和部署全链路 | [`../AI_PROJECT_GUIDE.md`](../AI_PROJECT_GUIDE.md) |
| 了解模块职责与拆分顺序 | [`module_boundaries.md`](module_boundaries.md) |
| 改群聊体验或决策行为 | [`group_chat_evolution.md`](group_chat_evolution.md) |
| 改记忆、RAG、画像或归档 | [`memory_lifecycle.md`](memory_lifecycle.md) |
| 改备份、COS、磁盘或数据库维护 | [`storage_maintenance.md`](storage_maintenance.md) |
| 查看当前工程风险与待办 | [`audits/2026-08-28-project-audit.md`](audits/2026-08-28-project-audit.md) |

## 2. 生产不变量

1. 只在服务器目录 `/opt/qq-social-agent` 部署生产改动；不得将本机运行数据覆盖服务器。
2. `.env`、`data/`、`server-data/`、NapCat 登录态和 `data/meme_library/` 不提交 Git。
3. 运行中普通代码、Prompt、插件和 `config.yaml` 通过 bind mount 注入 bot 容器；生效只重启 `qq-social-agent-bot`。
4. 不因 Python/Prompt 改动重启 NapCat，不执行 `docker compose down`，不删除 `server-data/ntqq`。
5. 不把 COS 当作实时数据库；它只承担归档、快照和恢复来源。
6. 所有长期记忆必须能回溯来源；人工锁定的 memory atom / summary 不得被自动任务改写。

## 3. 每次变更流程

### 3.1 改动前

```bash
cd /opt/qq-social-agent
git status --short
curl -fsS http://127.0.0.1:8080/readyz
```

- 先确认当前是否已有用户未提交的源码变更；只忽略已知运行目录 `data/meme_library/`。
- 涉及 `data/bot.sqlite3` 的人工修复，先创建本地快照并确认 COS 备份最近一次成功。
- 改 Prompt、模型或频率前，先查看最近 Trace / Token / 拦截指标，不以单句聊天印象直接改配置。

### 3.2 开发与验证

源码在宿主机，测试目录不是 bind mount。使用容器跑服务器当前源码时，先同步测试：

```bash
cd /opt/qq-social-agent
docker cp tests/. qq-social-agent-bot:/app/tests/
docker exec qq-social-agent-bot python -m pytest -q
docker exec qq-social-agent-bot python -m compileall -q /app/qq_social_agent
docker exec qq-social-agent-bot python -m pip check
```

- 修改范围小：至少运行对应测试文件。
- 修改消息主流程、数据库、RAG、模型路由或发送：必须跑全量测试。
- 全量测试非全绿时，不得把“测试通过”写进提交说明；先在审计/待办中登记失败原因。
- 不在生产环境直接批量升级 Python 包或 Docker 基础镜像。依赖升级必须单独提交、测试、观察。

### 3.3 发布

普通 Python、Prompt、插件或普通配置修改：

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml restart bot
curl -fsS http://127.0.0.1:8080/readyz
```

只有改动 `pyproject.toml`、`Dockerfile.server` 或系统依赖时才重建 bot：

```bash
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build --no-deps bot
```

发布后确认：

- `/readyz` 内 `database_ready`、`llm_ready`、`onebot_ready` 都为 `true`。
- `docker logs --tail 80 qq-social-agent-bot` 没有启动循环、数据库迁移失败或 OneBot 重连风暴。
- NapCat 保持原容器和登录态。

### 3.4 收尾

```bash
git diff --check
git add <源码和文档>
git commit -m "<清晰中文或英文说明>"
git push origin main
git status --short
```

- 提交必须包含相关测试或明确说明为什么没有测试。
- 更新受影响的架构文档、Prompt 文档或运行手册。
- 不提交数据库、日志、备份、二维码、账号信息、密钥或私聊内容。

## 4. 日常巡检

### 每周

- 查看 `/opt/qq-social-agent/reports/server_health_latest.md`。
- 检查 `/readyz`、容器状态、磁盘、内存、Swap、最近 bot 错误日志。
- 看 `bot_metric_events` 与 `llm_usage_events`：decision 通过率、发送率、工具失败、provider fallback 和 Token 异常。
- 打开 WebUI 的 RAG / 记忆页，抽查是否存在错人、把反串写成事实或明显重复风格规则。

### 每月

- 运行 `scripts/system_hygiene.sh --dry-run`，关注数据库完整性、RAG 垃圾、WAL、备份和媒体缓存。
- 检查 COS 最近的 SQLite 快照、项目元数据、聊天原文和 ntqq 渐进归档均有成功记录。
- 做一次**非生产路径**的恢复演练：将一份 COS SQLite 快照恢复到临时目录，用只读方式检查表和消息数量；不覆盖在线数据库。
- 审查 `memory_summaries`、`memory_atoms`、`style_rules` 的增长，按 `memory_lifecycle.md` 推进分层，而不是无限增加 active 数据。

### 每季度

- 复核私聊白名单、工具管理员、WebUI SSH tunnel 使用者。
- 更新依赖风险评估；只针对有明确安全/兼容性需求的包做小批升级。
- 审核最近三个月的审计报告，将完成项归档、未完成项重新排优先级。

## 5. 自动任务与日志

标准定时任务位于 `/etc/cron.d/qq-social-agent-*`：

- 00:20：昨日聊天原文与历史索引归档。
- 04:17：数据库、日志、备份、Docker、NapCat 临时文件体检与清理。
- 04:42：SQLite 与项目元数据 COS 备份。
- 05:42：NapCat `ntqq` 渐进 COS 备份。
- 每周一 06:12：健康报告。

日志目录：`/opt/qq-social-agent/logs/`。报告目录：`/opt/qq-social-agent/reports/`。

**注意：** 2026-08-28 已移除历史遗留的用户 crontab 清理任务。新增维护 cron 前先检查 `crontab -l` 与 `/etc/cron.d/qq-social-agent-*`，确保同类任务只有一份。

## 6. 故障处理

### 后端不可用

1. `curl -fsS http://127.0.0.1:8080/readyz`
2. `docker logs --tail 160 qq-social-agent-bot`
3. `docker compose -p qq-social-agent -f docker-compose.server.yml ps`
4. 仅重启 bot，再确认 OneBot 是否自动重连。

### QQ 不收发 / OneBot 未连接

先确认 backend 仍健康，再查看 NapCat 日志和 WebUI。只有明确掉线、登录态损坏或 QQ 内核失效时才重启 NapCat；重启前应告知操作者二维码可能需要重新扫码。

### 数据库问题

1. 停止对数据库的写入风险操作。
2. 运行 `scripts/system_hygiene.sh --dry-run`，确认完整性结果。
3. 从最新 COS SQLite 快照复制到临时路径检查，不直接覆盖在线 `data/bot.sqlite3`。
4. 恢复必须是单独的变更任务，记录恢复时间点、来源和验证结果。

### LLM Provider 超时

当前客户端有 provider 熔断：同一 provider 在 120 秒内连续失败 3 次，会暂时绕过 5 分钟并走 fallback；成功后自动恢复优先级。排查时看 `llm_usage_events` 和 bot 日志，不要为了短暂超时直接改 Prompt 或删 API key。

## 7. 质量门与待办管理

每个工程待办都要有：范围、优先级、复现方式、预期结果、测试、观测指标、回滚方式和对应文档。

优先级定义：

- P0：数据泄露、数据损坏、QQ 不可用、可被外部利用的漏洞。
- P1：测试不全绿、消息错误发送、长期数据持续错误写入、严重可靠性回退。
- P2：会增加维护成本或偶发行为异常的工程债。
- P3：体验、文案、低风险清理或未来扩展。

当前公开待办以 [`audits/2026-08-28-project-audit.md`](audits/2026-08-28-project-audit.md) 为准。完成任务时更新该报告或新增一份新日期审计，不要静默删除问题记录。
