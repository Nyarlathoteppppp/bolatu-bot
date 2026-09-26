# 文本模型入口

文本生成任务使用 `LLMTaskClient` 组织提示词和解析结果，再由 `LLMGateway` 统一发送 Chat Completions 请求。网关负责 provider 客户端、任务路由、超时、回退、熔断和用量记录。JEV 决策接口、OCR、embedding、语音使用各自协议，不进入文本聊天路由。

## 添加模型

在 `config.yaml` 的 `llm.providers` 添加 OpenAI Chat Completions 兼容端点和 `api_key_env`，在服务器 `/opt/qq-social-agent/.env` 设置该变量，再把 `provider/model-id` 加入 `model_catalog`。需要默认使用时，修改对应的 `reply_model`、`decision_model` 等任务路由。`fallback_models` 接受单个模型或有序列表。列表中第一个失败后按顺序尝试下一个，共用该任务的总超时预算。

当前回复顺序为 `mimo/mimo-v2.6-pro` → `deepseek/deepseek-flash` → `siliconflow/deepseek-ai/DeepSeek-V4-Flash`。MiMo 返回余额不足时，本次请求立即回退，进程内后续请求跳过 MiMo；重启后会重新尝试。工作日高峰时，原有 DeepSeek/SiliconFlow 价差路由仍适用于两个回退模型。其他任务仍使用原来的路由。

OpenRouter 的 `openai/gpt-6-luna` 已加入模型清单，provider 的推理强度配置为 `high`，没有成为默认路由。2026-09-27 从生产服务器实测返回 HTTP 403，表示该服务器出口地区暂不能使用它。

## QQ 私聊命令

仅主人号可用：

- `模型状态`：查看当前任务路由、回退顺序和候选模型。
- `测试模型`：逐个请求候选模型，显示可用、HTTP 状态、超时或缺少密钥。每次测试会调用模型并消耗少量额度。
- `测试模型 1`：按 `模型状态` 中的编号只测一个模型，失败不会被备用模型掩盖。也接受完整的 `provider/model` 名称。
- `切回复模型 1`：按相同编号覆盖回复路由；其他任务类似。也接受完整模型名称。覆盖保存到应用 KV，重启后仍生效。
- `清模型覆盖`：恢复 `config.yaml` 中的默认路由。

管理台已有相同的任务路由覆盖。切换只改变主模型，仍沿用该任务配置的回退链。
