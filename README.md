# QQ Social Agent

Mac 本机可跑的 QQ 群聊 Agent：

- NapCatQQ 负责 QQ 登录和 OneBot v11 事件转发
- NoneBot2 接收群聊消息
- 本地 scorer 决定是否插话
- rate limiter 控制频率
- YAML persona skill 控制人格
- DeepSeek API 生成回复

## 1. 安装

```bash
cd /Users/ywbw/qq-social-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

编辑 `.env`，填入：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API key
SUPERUSERS=["你的QQ号"]
```

不要把真实 key 提交到仓库。

## 2. NapCat 配置

在 NapCat 的 OneBot v11 配置里添加反向 WebSocket：

```text
ws://127.0.0.1:8080/onebot/v11/ws
```

## 3. 日常启动

项目环境名：

```text
/Users/ywbw/qq-social-agent/.venv
```

启动 QQ/NapCat：

```bash
/Users/ywbw/qq-social-agent/scripts/start_napcat.sh
```

启动 bot：

```bash
/Users/ywbw/qq-social-agent/scripts/start_bot.sh
```

后台启动 bot：

```bash
/Users/ywbw/qq-social-agent/scripts/start_bot_daemon.sh
```

后台日志：

```text
/Users/ywbw/qq-social-agent/logs/bot-runtime.log
```

查看当前状态：

```bash
/Users/ywbw/qq-social-agent/scripts/status.sh
```

重启 bot：

```bash
/Users/ywbw/qq-social-agent/scripts/restart_bot.sh
```

`restart_bot.sh` 使用后台 daemon 方式启动，避免 macOS Terminal 前后台进程组导致 bot 被暂停。
后台 daemon 使用 macOS `launchctl`，服务标签：

```text
com.ywbw.qq-social-agent
```

当前 DeepSeek 模型在 `config.yaml`：

```yaml
deepseek:
  model: deepseek-v4-pro
  thinking: disabled
```

说明：DeepSeek V4 默认会开启 thinking。群聊人格默认关掉 thinking，让 `temperature` 生效，回复更像正常聊天；需要更强推理时再改成 `thinking: enabled`。

## 4. 群内命令

```text
/bot status
/bot pause
/bot resume
/bot reset
/bot persona zhangxuefeng
```

被 @ 或回复时优先响应。非 @ 消息会先过频控，再由 DeepSeek 返回 JSON 决策是否自然插话。

## 5. 调参

主要改 `config.yaml`：

- `min_interval_seconds` 控制最小发言间隔
- `max_replies_per_10min` 控制 10 分钟上限
- `max_replies_per_hour` 控制小时上限
- `context_limit` 控制取多少条最近聊天作为氛围上下文，当前默认 48；模型实际使用最近 40 条

人格改 `personas/*.yaml`。

## 6. 市场信息工具

群友聊美股或加密货币时，bot 会尝试补充实时市场信息：

- 美股数据源：Yahoo Finance，通过 `yfinance`
- 加密货币数据源：CoinGecko 公共 API
- 每条消息最多识别 2 个标的
- 外部行情查询全局限制：每 60 秒最多 2 次
- 60 秒内同一标的复用缓存，不重复查外部接口

支持示例：

```text
英伟达今天咋样
NVDA跌麻了
BTC多少了
ETH为啥跌
```

市场数据只作为聊天参考，不构成交易建议。

## 7. 最新背景搜索

群友讨论实时新闻、国际冲突、比赛赛果、政策、刚发布的产品等内容时，bot 会先由 DeepSeek 判断是否需要最新背景。只有决定要回复且确实需要最新信息时，才会调用搜索源。

当前搜索源配置在 `.env`：

```env
TAVILY_API_KEY=你的 Tavily key
FRESH_SEARCH_PROVIDER=auto
```

`auto` 规则：

- 有 `TAVILY_API_KEY`：优先 Tavily
- Tavily 不可用或没有 key：回退 Google News RSS
- 外部搜索限制：每 60 秒最多 2 次

低价值查询会被本地过滤，例如“今天周几”“几点”“测试”“随便搜搜”等，不会浪费搜索额度。
