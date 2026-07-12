# QQ Social Agent

运行在 Ubuntu 服务器上的 QQ 群聊人格机器人。NapCatQQ 负责 QQ 登录和 OneBot v11 事件转发，NoneBot2 接收消息，本地后端负责筛选、频控、记忆、工具调用和审批，外部 LLM 负责决策与回复生成。

详细工程说明见 [AI_PROJECT_GUIDE.md](AI_PROJECT_GUIDE.md)，服务器部署说明见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。

## 1. 项目路径和运行结构

主要开发与生产工作区：

```text
/opt/qq-social-agent
```

消息链路：

```text
QQ / NapCat
  -> OneBot v11 reverse WebSocket
  -> NoneBot2 (bot.py)
  -> qq_social_agent/plugin.py
  -> 本地筛选 / LLM / 搜索 / 行情 / 记忆 / 审批
```

服务器使用 Docker Compose，固定项目名为 `qq-social-agent`：

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml ps
```

关键服务：

- `bot` / `qq-social-agent-bot`：监听 `127.0.0.1:8080`
- `napcat`：WebUI 绑定 `127.0.0.1:6099`

## 2. 配置

复制环境变量样例并填写密钥：

```bash
cd /opt/qq-social-agent
cp .env.example .env
```

不要把真实密钥提交到仓库。NapCat 的 OneBot v11 反向 WebSocket 应配置为：

```text
ws://bot:8080/onebot/v11/ws
```

主要配置文件：

- `config.yaml`：群白名单、模型路由、频控和运行参数
- `prompts/zhangfengxue.yaml`：人格、决策、回复、记忆和学习 Prompt
- `data/bot.sqlite3`：生产记忆和运行状态，不要随意删除或提交

当前默认模型路由以 `config.yaml` 为准。运行时还可能通过 QQ 工具单设置模型覆盖，覆盖值保存在 SQLite；使用“模型状态”命令查看实际路由。

## 3. 启动与更新

首次启动或重新构建全部服务：

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build
```

只重新构建并启动 bot，不动 NapCat：

```bash
cd /opt/qq-social-agent
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --build --no-deps bot
```

日常操作也可使用：

```bash
scripts/status.sh
scripts/restart_bot.sh
scripts/stop_bot.sh
scripts/start_bot_daemon.sh
```

除非 QQ 登录或连接确实损坏，不要随便重启 NapCat；重启可能需要重新扫码并触发 QQ 风控。

查看日志：

```bash
docker compose -p qq-social-agent -f docker-compose.server.yml logs -f bot
docker compose -p qq-social-agent -f docker-compose.server.yml logs -f napcat
```

NapCat WebUI 只绑定服务器回环地址。通过 SSH tunnel 访问：

```bash
ssh -L 6099:127.0.0.1:6099 qqbot-server
```

然后打开 `http://127.0.0.1:6099/webui`。

## 4. 开发和测试

服务器若已安装开发依赖：

```bash
cd /opt/qq-social-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

生产镜像默认不包含 pytest 和 `tests/`，因此测试应在服务器开发虚拟环境或专门的测试镜像中执行。

人格或回复风格优先修改：

```text
/opt/qq-social-agent/prompts/zhangfengxue.yaml
```

Prompt 在后端启动时加载。修改 Python 代码后使用 `scripts/restart_bot.sh` 重新构建 bot；只修改 Prompt 时运行 `docker compose -p qq-social-agent -f docker-compose.server.yml restart bot` 即可，不需要重新构建镜像。

## 5. 群聊行为

- 被 @、回复或点名时优先响应；实际问题由后端保证进入回答流程，即使重复追问也不能用反问或调侃代替答案。
- 普通群消息先做持久去重，再经过 buffer、工作强度抽样、本地筛选和 LLM decision。
- bot 连接后会同步群资料/群成员，并补最近群历史；引用消息缺原文时会用 `get_msg` 补全上下文。
- 行情工具支持美股和加密货币；联网搜索支持 Tavily，并按类型回退 Google News 或 Bing Web RSS，结果保留可核查来源。需要细读时可受限抓取网页正文。
- decision 可以选择 `react`：只给当前消息点 QQ 表情，不发文字，并受后端限频控制。
- 普通图片由 SiliconFlow `deepseek-ai/DeepSeek-OCR` 做画面简述、文字转写和梗图含义；商城动画表情默认跳过 OCR。受限读取小型 txt/PDF/docx，语音仅在明确相关时转写，其他富媒体保留安全元数据。
- 长期记忆带来源、证据、时间、置信度和有效状态，可在私聊工具单中查看证据、纠正、反证或软删除。
- `GET /healthz`、`GET /readyz` 和 `GET /status` 分别用于存活、就绪和详细运行诊断；`GET /traces` 和 `GET /trace` 用于查看消息链路。服务只绑定本机端口。
- decision/reply/后台任务分别设置单次与总耗时预算，SDK 自动重试关闭，避免一次模型卡顿把群聊回复拖到近一分钟。
- 群聊通常生成 3 条候选；审查开启时先私聊审批人。

常用审批操作：

```text
A / B / C 或 1 / 2 / 3：选择候选
D / X / 取消：不发送
1!：发送并标记为优质
不准奏原因：xxx：记录负反馈
bot工具：查看工具单
```

群内管理命令：

```text
/bot status
/bot pause
/bot resume
/bot reset
/bot persona zhangxuefeng
```
