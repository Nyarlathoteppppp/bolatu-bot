# VerySadai 接入（2026-10-03）

## 回复路由

默认回复：`verysadai/gpt-6.1-sol` → `lingsuan/gpt-6.1-sol` → `deepseek/deepseek-flash`。
两条 Sol 路由均使用 Responses API，`reasoning.effort=medium`、`store=false`。
每次请求沿用 `max_retries=0`。Sadai 主链路单次超时 30 秒、总预算 80 秒，给两个回退提供时间；沿用既有余额耗尽处理和熔断机制。

`llm.providers.verysadai.reply_fallback_models` 单独声明回退顺序；全局 fallback 保留 DS/SiliconFlow，因此已有“高峰 DeepSeek/SiliconFlow 组合”仍可选择。
搜索和图片识别保留 DS；其余工具文本、决策和后台模型继续使用原有配置。聊天列表仅新增 VerySadai Sol，没有加入 Astra 或生图模型。
密钥从 `.env` 的 `VERYSADAI_API_KEY` 读取。密钥、生成图不进入 Git。

## 生图工具

- 插件：`plugins/image_generation/manifest.yaml`，权限 `tool.image_generation`。
- 工具选择：JEV 识别 `image_generation`；原有 DS 工具参数生成器补齐画面描述，保留风格、构图和图中文字，不压成搜索关键词。
- 接口：`POST https://verysadai.com/v1/images/generations`。
- 默认模型：`gpt-image-2.5-sunburst`，`quality=low`、`n=1`、请求尺寸 `1024x1024`、PNG。
- 文件：项目 `.pi/generated-images/<随机文件名>.png`；生产 bind mount 到 `/opt/qq-social-agent/.pi/generated-images`，重建 bot 后保留。
- 传递：`ToolResult.generated_images` 专门携带图片，base64 不放在日志 metadata、LLM 上下文或候选正文里，模型不能指定本地文件路径。
- 群聊：成功后产出图片候选，直接走现有审批/直发入口，不再调用回复生成器；图片跟随首段发送，复用 DeliveryProgress 对成功或结果未知的发送防重。
- 私聊：图片从 PrivateToolStage 传到 PrivateGenerationContext，直接发送，无须模型另外写一句成功通知。
- 群问题转私聊：同样发送图片，记录私聊记忆，已成功投递的候选不会重复投递。
- API 失败：沿用 ToolRegistry 的错误结果，记录失败且不产生图片；失败状态交给正常回复链路。生图不自动重试或换模型。

使用：群里 `@风雪 帮我画一张水彩白猫，横向构图`；允许聊天的私聊里直接发送同样请求。搜索已有图片和解释已发图片不走生图。

## 已实测的服务限制

服务器直接调用 Sol medium 成功，角色 system 消息和 JSON 输出成功，约 5–6 秒。单次成功不保证渠道稳定。
当前字符串 content 可用；未转换为纯文本 content 数组。显式 `instructions` 字段请求返回 HTTP 403，响应仍附带上游默认 Codex 指令（约 4.2K input tokens，部分缓存）。目前角色 system 消息能影响回复，但无法保证上游指令对所有角色扮演任务没有影响。

Sunburst 生图成功，请求 1024×1024，实际返回 1254×1254。尺寸按服务端实际结果记录；希望横/竖构图时写入画面描述。
其他模型、思考档位、真实容量与价格不由此测试推断；Pi 费用元数据 0 不表示免费。
独立测试包保存在服务器 `/home/ubuntu/tools/verysadai-server`。聊天通道、图像通道分开验证。
