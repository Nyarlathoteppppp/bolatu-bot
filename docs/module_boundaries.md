# 模块边界与维护手册

最后更新：2026-10-04。

这份文档给在服务器上继续维护张风雪的开发者和 AI 使用。目标是让功能继续增长，但不再把所有事情塞进 `qq_social_agent/plugin.py`。

## 1. 当前结论

项目已经具备真实分层：消息结构化、决策、工具、上下文、记忆、RAG、审批、发送、观测、后台学习和本地插件都有独立模块。`plugin.py` 仍承担 NoneBot 入口、运行时单例装配、群聊编排和私聊阶段依赖装配；`group_session_service.py` 管理群聊锁、缓冲、flush 与点名排队，`group_post_send_service.py` 管理群回复后的记录、续聊窗口和附加动作；`private_session_service.py` 管理私聊缓冲与 followup 调度；三个 scheduler service 管理 daily review、weekly usage report 和 proactive chat 的后台 task 生命周期与周期策略；`daily_review_service.py` 和 `proactive_group_message_service.py` 管理定时群消息的生成与发送；`memory_maintenance_service.py` 管理中期记忆、群风格和成员画像的后台维护。群聊主循环已按话语解析、决策、上下文、工具执行、候选生成和审批交接分段；私聊主循环也已按轮次准备、工具执行、上下文检索和生成发送分段。

策略：**保留 `plugin.py` 作为适配器和组合根，不再向其中放业务规则；新增能力优先落到所属模块，再由主文件显式注册。** 不在缺少回归测试时做一次性大拆分。

## 2. 运行链路与依赖方向

```text
NapCat / OneBot Event
  -> plugin.py：NoneBot 事件入口和运行时装配
     -> 群聊：message_segments -> group_session_service -> decision_gate -> group_discourse_flow
        -> group_decision_flow + conversation_tool_routing
        -> group_generation_context -> group_tool_execution + group_reply_generation
        -> group_approval_dispatch -> approval_request_service
        -> private command: approval_command_service
           -> private_admin_command_service (operator tools)
           -> approval_state_service -> approved_reply_delivery -> group_post_send_service
     -> 私聊：private_session_service -> plugin.py 阶段协调
        -> private_turn_preparation -> private_tool_execution
        -> private_generation_context -> private_reply_delivery
  -> observability + background_learning + COS/归档
```

依赖只应从入口向下流动：

```text
entrypoint/plugin -> orchestration -> domain/storage/tools -> provider adapters
                         -> shared record types (llm_task_types / memory_models)
```

`memory.py`、`onebot_gateway.py`、`deepseek_client.py` 和工具实现都不能反向导入 `plugin.py`。固定人格 Prompt 放在 `prompts/zhangfengxue.yaml`；基于解析结果生成的上下文提示由所属模块格式化。

## 3. 模块地图

| 边界 | 主要文件 | 责任 | 不应承担 |
| --- | --- | --- | --- |
| QQ 接入 | `plugin.py`、`onebot_gateway.py`、`history_sync.py` | OneBot 事件、API、历史同步 | 人格回复判断 |
| 私聊轮次 | `private_message_types.py`、`private_turn_preparation.py` | 合并后的 `PrivateTurn`、去重与审批优先、媒体/语音/OCR/转发上下文、命令处理和用户消息入库 | 工具决策、模型 Prompt 拼装、QQ 回复发送 |
| 群聊会话 | `group_session_service.py`、`group_message_types.py` | 同群处理锁、消息缓冲、flush task、生成 inflight、已点名等待计数与接收序号；保持多人艾特 FIFO、同一人连续消息及原触发消息身份 | 开口判断、模型调用、OneBot 正文适配、角色 Prompt |
| 私聊会话 | `private_session_service.py` | 按用户缓冲与顺序 flush、processing lock、生成 inflight 状态、followup 计数/取消/延时/概率 | 读取 OneBot 事件正文、私聊模型上下文和审批状态 |
| 私聊工具与上下文 | `private_tool_execution.py`、`private_generation_context.py`、`conversation_tool_routing.py`、`tool_registry.py` | 共享工具路由、工具执行、并行 RAG 与记忆上下文，使用有类型的阶段结果交接 | 原始 OneBot 事件适配、审批状态 |
| 私聊生成与发送 | `private_reply_delivery.py`、`reply_splitter.py`、`meme_library.py` | 私聊模型调用、表情包选择、分段回复、首段引用及成功后的机器人消息入库 | 入口注册、会话 buffer 调度 |
| MessageChain | `message_segments.py`、`reference_resolver.py`、`media_context.py` | 原始 segment、引用/艾特/媒体事实 | 凭文本猜人物关系 |
| 转发聊天记录 | `forward_context.py` | 内联/远端转发读取、原发言人和时间格式化、逐条图片 OCR 与整批图片预算 | 群聊开口判断、转发者画像、运行时单例 |
| 输入内容摘要 | `message_summary.py` | 长消息和大型转发摘要、原文退化格式；通过显式客户端和 policy 保留已有阈值与文本 | 选择模型、QQ 发送、读取插件全局 |
| 图片识别状态 | `image_read_state.py`、`media_context.py` | 原消息接收事实、识图状态持久化、共享任务与原消息结果关联；只消费统一指代绑定 | 重新猜图片来源、插入迟到的用户消息、学习临时识别状态 |
| 前置筛选 | `decision_gate.py`、`rate_limiter.py` | 去重、低价值、频控、buffer | 社交氛围或搜索词 |
| 社交决策 | `decision_gate.py`、`group_decision_flow.py`、`timing_gate.py`、`pipeline_types.py`、`pipeline_stages.py` | channel、action、状态转移；`timing_gate.py` 根据 Jev 观察执行开口策略 | 最终回复正文 |
| 工具 | `tool_router.py`、`conversation_tool_routing.py`、`group_tool_execution.py`、`tool_registry.py`、`tools/` | 路由、执行、缓存、限流、结构化结果 | 客服式 fallback 文案 |
| 上下文 | `group_discourse_flow.py`、`group_generation_context.py`、`context_assembler.py`、`member_context.py`、`speaker_context.py` | 指代与记忆影响、画像、RAG、预算和输入拼装 | 数据库 schema |
| 候选与审批交接 | `group_reply_generation.py`、`group_approval_dispatch.py` | 草稿生成、复核、审批状态转移 | QQ 实际发送 |
| 审批命令与管理员工具 | `approval_command_service.py`、`private_admin_command_service.py`、`approval_rules.py` | 命令解析、角色权限和管理员工具命令分派 | 待审批单状态修改、群消息发送 |
| 审批状态与请求 | `approval_state_service.py`、`approval_request_service.py`、`approval_models.py` | 待审批单集合、串行候选选择、取消反馈、stale choice 冷却、审批请求和自动发送决策 | 群消息发送与 delivery progress 写回 |
| 群回复发送后处理 | `group_post_send_service.py` | 机器人发送回执索引、续聊窗口、成功正文后的表情包与附加反应；显式共享窗口和当前运行时依赖 | 改变审批投递顺序、正文分段、开口概率或 Prompt |
| 审批发送 | `approved_reply_delivery.py`、`delivery.py`、`approval_models.py` | 已批准消息发送、分段进度和数据库/Trace 回写；未知结果标记后阻止盲目重试 | 审批命令解析与工具权限 |
| 共享数据契约 | `llm_task_types.py`、`memory_models.py` | 模型任务 DTO 与消息/记忆/画像/回执记录，仅依赖标准库；原 owner 显式再导出同一类对象 | 模型请求、数据库读写、运行时配置、Prompt 和策略 |
| 文本模型 | `llm_gateway.py`、`deepseek_client.py`、`prompts.py` | gateway 统一 provider、任务路由、超时、回退和用量；task client 负责提示词与结果解析 | QQ 发送与审批状态 |
| 模型管理 | `model_route_service.py` | 模型目录/编号、覆盖持久化与启动应用、状态、探测、切换和重置；QQ 命令与管理台共享操作。`BackgroundModelSettings` 仅依赖后台配置与 KV | 改变生成路由算法、模型请求协议、聊天 Prompt、后台任务生命周期 |
| 专用模型接口 | `jev_client.py`、`embedding_client.py`、`siliconflow_ocr.py`、语音客户端 | 各自协议和模态的请求 | 文本聊天路由 |
| 短期互动 | `interaction_state.py` | 引用原消息与真实发送回执；消费唯一话语状态，按当前窗口整理回复链、字面反馈与结束标记 | 再次判断说话人；推测情绪、亲密度或校园经历；生成角色卡措辞 |
| 自身互动连续性 | `interaction_state.py` 的 `own_contributions()`、`self_interaction_context.py` | 从当前分支和原有窗口读取已确认发送的原话、表达动作、触发人、直接回应与后续明确反馈；仅喂普通聊天生成阶段 | 写入推测的内心或情绪；影响发言时机；用旧观点覆盖联网事实 |
| 记忆存储入口 | `memory.py`、`memory_models.py` | 单一 SQLite 连接、repository/Schema 装配、原 API 的显式委托；少量群开关、自定义词与 KV 读写 | 消息、摘要、成员、指标、私聊状态、表情包、atoms 和风格存储的实现 |
| 记忆领域存储 | `memory_*_repository.py` | 共用原连接的消息/反馈、群目录/成员画像与印象、摘要、指标/用量、私聊状态、素材、原子生命周期/审计/召回、风格学习/合并/召回 | 新建连接、修改事务边界、运行时单例、Prompt |
| 记忆 Schema | `memory_schema.py` | 原建表、逐列升级与历史回填的有序执行；显式接收原子过期、成员画像与印象回调 | 独立数据库连接、开口策略、后台调度 |
| 记忆学习与 RAG | `memory_learning.py`、`memory_maintenance_service.py`、`rag_*.py` | 学习草稿持久化、异步记忆总结、风格学习和成员画像更新、索引 | QQ 生命周期、聊天热路径 |
| 发送 | `delivery.py`、`reply_splitter.py`、`social_actions.py` | 拆分、艾特、表情、节流 | 写长期事实 |
| 定时任务调度 | `daily_review_scheduler_service.py`、`weekly_usage_report_scheduler_service.py`、`proactive_chat_scheduler_service.py` | 按 Bot 管理 task 去重、时间窗口/概率策略、周期 tick、异常记录和取消清理 | 群消息生成、周报格式化与投递 |
| 定时群消息 | `daily_review_service.py`、`proactive_group_message_service.py` | daily review 与 proactive 群消息生成、投递、成功后状态/记忆回写 | scheduler 生命周期；私聊小时任务共用的话题选择入口仍由 `plugin.py` 适配 |
| 后台记忆维护 | `background_learning.py`、`memory_maintenance_service.py`、`memory_learning.py` | 单 worker 协调与记忆/风格/画像维护；attempt/streak 状态归 service 所有 | OneBot 事件适配和群聊热路径 |
| 管理 HTTP | `admin_controller.py`、`admin_tools_controller.py`、`admin_edit_controller.py`、`admin_summaries_controller.py`、`admin_memory_controller.py`、`admin_http.py` | 本地管理路由、鉴权委托、表单适配、资源操作和重定向；编辑服务保留路径白名单、内容校验、备份与原子替换 | 群聊/私聊热路径，页面 HTML 拼装 |
| 运行观测 | `observability.py` | correlation scope、OneBot 连接与发送健康；保留共享状态和旧 Trace 导入入口 | Trace 事件归一化和 HTML 拼装 |
| Trace 数据与展示 | `trace_snapshot.py`、`trace_render.py` | 事件归一化、分阶段聚合、元数据脱敏、JSON 快照与 HTML 转义 | QQ 连接状态、运行时单例、聊天策略 |
| 搜索 | `tools/fresh_context.py`、`fresh_intent.py`、`fresh_providers.py`、`fresh_types.py`、`fresh_text.py` | 编排与缓存、意图/查询解析、Provider HTTP 和结果解析、共享记录与文本辅助 | 人格和开口策略；Provider 不能反向导入编排器 |
| 管理页面 | `admin_ui.py`、`approval_rules.py` | 管理页面渲染、工具单 | HTTP request parsing、管理业务规则、聊天热路径判断 |
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

- `plugin.py` 仍承载生命周期、群聊编排和私聊阶段依赖装配；26 条 `/admin` 路由已迁入资源 controller，`plugin.py` 只保留本地鉴权策略、运行时依赖组装和 `/status`、`/healthz`、`/readyz`、`/trace(s)` 观测路由。
- `memory.py`、`deepseek_client.py`、`rag_store.py` 仍大；消息/记忆记录与模型任务 DTO 已移入独立类型模块，部分数据库方法已按领域迁入共享连接的 repository；消息/摘要/成员查询与 Schema 已分别独立；剩余主要是显式兼容接口、少量设置存储，以及模型请求和解析的后续维护。
- manifest 能声明能力，但实际 handler 注册仍集中在主文件。

## 6. 下一步拆分顺序

群聊高频改动路径已按阶段拆出。`plugin.py` 装配当前 LLM、存储、日志和指标回调，并保留事件生命周期与阶段间早退。后续继续按功能边界迁移，保持 Trace 事件名与数据库写入不变。

1. **定时任务生命周期（第一批已完成）**：三个 scheduler service 管理 daily review、weekly usage report、proactive chat 的生命周期和 tick 策略。`plugin.py` 在 connect 时按 manifest/config 开关调用 `_ensure_*`，disconnect/shutdown 取消同一组 service task registry。
2. **管理 controller（已完成）**：26 条 `/admin` 路由由 `admin_controller.py` 聚合，并按 tools、editable files、summaries、memory/private-memory 划分资源 controller；`admin_ui.py` 只负责渲染。各 controller 通过窄依赖对象调用运行时，不反向导入 `plugin.py`。原有本地请求判定和注册顺序保留；状态与 Trace 路由仍由入口注册。编辑服务保留项目路径校验、YAML/Prompt/config 校验、保存前备份、临时文件替换及 Docker 单文件 bind mount 的原地写入处理。
3. **后台记忆维护（已完成）**：`memory_maintenance_service.py` 负责中期摘要、风格规则和成员画像维护；`plugin.py` 保留 task/coordinator 装配与兼容入口。attempt 时间、空摘要 streak、摘要游标推进、私聊早退、持久化和指标名沿用原逻辑；状态字典只有 service 一份，plugin 的旧名称是同对象别名。
4. **定时群消息（本批完成）**：`daily_review_service.py` 管理定时/手动复盘的目标群、共享发送锁、生成、逐段投递与成功后的 sent marker/学习回写；`proactive_group_message_service.py` 管理主动群消息的上下文、生成、逐段投递和成功后的 topic marker。`plugin.py` 保留运行时依赖装配与原函数入口；daily review 锁仍与手动/定时路径共用同一对象，proactive 使用群聊共享的 inflight 集合。私聊小时任务继续经 plugin 的 `_select_proactive_topic` 选择话题。周报格式化和私聊投递仍留在入口，后续按相同边界单独评估。
5. **模型管理与输入内容（2026-10-03 完成）**：`ModelRouteService` 统一 QQ 命令与管理台的模型选择、切换、重置及覆盖存储；`BackgroundModelSettings` 独立读取后台配置和 KV，查询 batch 模式不要求 LLM 客户端或完整 LLM 配置。`ForwardContextService` 保留转发原发言人、时间和逐条图片归属；`MessageSummaryService` 只处理内容摘要。入口按调用装配当前依赖，避免持有过期客户端或额外覆盖副本。原命令文案、目录顺序、探测并发、摘要参数和失败分支沿用原实现，Prompt/config 文件未改。`plugin.py` 本轮从 10,505 行降到 9,837 行。
6. **群聊会话（2026-10-03 完成）**：`GroupSessionState` 保存共享 registry；`GroupSessionService` 统一缓冲、锁、任务去重、等待计数、接收序号与 flush。入口只构造 OneBot 对应的 `BufferedGroupMessage` 并装配当前处理/调度/指标回调。旧 registry 名称是同一状态对象的适配别名，scheduler/admin/shutdown 仍共享原对象；工厂以当前 registry 引用创建服务视图，不复制队列。普通消息立即 flush、生成中的 1 秒等待、多用户点名 FIFO、同人连续消息、原消息/关联 ID 以及异常取消后的清理沿用现有逻辑。群缓冲文本格式迁入 `group_message_types.py`，措辞未改。
7. **群回复发送后处理（2026-10-03 完成）**：`GroupPostSendService` 接管机器人发送记录、续聊窗口、群表情包和审批后的附加反应。`approved_reply_delivery.py` 继续管理正文投递及原调用顺序；入口保留旧函数适配，按调用装配当前客户端、发送器、指标和共享窗口。表情包只有发送成功后才写入素材冷却、消息记录和互动回执；ActionFailed 与其他异常保留各自原分支，40 秒窗口内不滑动刷新；默认阈值和文案均未改。群聊会话与发送后处理拆分后，`plugin.py` 为 9,505 行。

8. **共享类型与导入依赖（2026-10-04 完成）**：从 `deepseek_client.py` 提取 11 个模型任务 DTO 到 `llm_task_types.py`，从 `memory.py` 提取 21 个记录类型到 `memory_models.py`；两个类型模块只依赖标准库。使用方直接导入类型，真正需要客户端/存储的模块仍依赖原实现。旧模块显式再导出同一类对象，保持原调用方和测试的公开导入兼容；字段、顺序、默认值、frozen 声明及 `MemoryAtom.evidence_source` 属性不变。决策门、工具路由与私聊消息类型的独立导入不再加载数据库或模型网关。动作集合、SQL、模型请求/解析和 Prompt 未改。`memory.py` 为 5,188 行，`deepseek_client.py` 为 2,494 行。

9. **搜索边界（2026-10-04 完成）**：`fresh_intent.py` 负责现有意图识别与查询整理，`fresh_providers.py` 负责各搜索 Provider 的 HTTP 请求、响应解析和结果质量/排序辅助，`fresh_types.py` 保存共享记录，`fresh_text.py` 保存共享清洗、URL host 和超时构造函数。`fresh_context.py` 保留多轮搜索编排、缓存、总 deadline、Provider dispatcher、正文跟读及 factpack 格式化；仍显式再导出原符号，原 dispatcher 的 fetch monkeypatch 入口可继续使用。纯查询消费者改为直接导入意图/类型模块，不再加载搜索编排器和 Provider 适配器。原搜索请求、Prompt、排序、缓存和回退逻辑未改。`fresh_context.py` 从 2,711 行降到 1,738 行。
10. **Trace 数据与展示（2026-10-04 完成）**：`trace_snapshot.py` 负责原事件归一化、关联分组、阶段聚合、脱敏和 JSON 快照；`trace_render.py` 只依赖快照层，负责原 HTML 渲染与转义。`observability.py` 从 1,157 行降到 306 行，继续拥有唯一的 ContextVar、连接集合与连接健康字典，以及所有运行时操作，显式再导出旧 Trace 符号。直接替换原状态字典的传输测试语义保持不变；未引入模块代理或状态副本。Trace schema、阶段顺序、限制值、脱敏规则及页面内容未改。

11. **记忆 repository 第一、二批（2026-10-04 完成）**：指标/模型用量、私聊会话状态、表情包素材、记忆原子与风格规则分别迁入五个 repository。`MemoryStore` 保留全部原方法签名与委托入口，五个 repository 均接收同一个 `self.conn`，构造函数只保存连接引用。原子的创建/纠正/反证/争议/过期与审计仍由同一 repository 在原事务中执行，SQL、commit/rollback、排序、评分和 TTL 未改。原子 repository 在 Schema 初始化前构造，满足初始化最后触发过期维护的现有调用；其他 repository 在 Schema 完成后构造。共享身份映射与账号展开迁到 `memory_identity.py`；清洗/行转换/相关度辅助迁到 `memory_text.py` 和 `memory_repository_utils.py`，原模块显式再导出旧符号，领域存储不反向导入 `memory.py`。46 个方法实现、40 个辅助函数及全部常量与迁移前 AST 一致。新增边界测试覆盖独立导入、连接身份、随主连接关闭和五个领域原有的共享提交行为。`memory.py` 从 5,188 行降到 3,421 行。

12. **记忆 repository 第三批与 Schema（2026-10-04 完成）**：`memory_message_repository.py` 管理消息、接收去重、上下文更新、原文例子、机器人已发送记录及反馈；`memory_member_repository.py` 管理群目录、成员画像/印象及成员总结；`memory_summary_repository.py` 管理摘要、游标、召回和管理台编辑。原文分类辅助统一归 `memory_corpus.py`。消息 repository 显式接收原 ImageReadStateStore 与成员更新回调，入库→图片登记→画像→印象→提交的原顺序不变。`memory_schema.py` 管理原建表、升级和回填，原子与成员 repository 在建表前只保存连接引用，Schema 通过显式回调执行原过期维护与回填；消息 repository 在 images 创建后装配。旧方法仍显式委托，不使用动态属性代理。64 个方法及 19 个辅助函数的实现、签名和全部 SQL 与迁移前 AST 一致；新库和含旧消息/摘要/原子的旧库回放，完整数据库 dump 与原实现一致，重开库也保持一致。`memory.py` 从 3,421 行降到 1,535 行，剩余主要是兼容 API、装配及少量群设置、词条和 KV 操作。

现阶段无需继续为减少 `memory.py` 行数拆它的委托接口。后续功能修改进入所属 repository，连接管理留在入口。



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

## 短期互动与角色卡组织（2026-09-30）

消息依据、分支边界、发送回执和生命周期见 [interaction_state.md](interaction_state.md)。角色正文与运行规则的加载方式见 [persona_prompt_organization.md](persona_prompt_organization.md)。角色 PList 保持原文，事实、能力和输出句子迁入运行规则；按用户偏好允许简短括号表情、心理反应或玩笑动作，critic 同步区分表达与现实事实。普通聊天生成 flow 将社交 action 作为表达建议，认真问题仍先答；自身互动证据在决策完成后注入，不加入工具答案的上下文。
