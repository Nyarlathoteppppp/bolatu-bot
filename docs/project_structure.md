# QQ 风雪：当前目录与代码阅读入口

更新日期：2026-10-02。目录由当前 `live-hotfix-20260917` 工作区的 Git 文件清单生成，供 Claude 阅读代码与规划改动。代码仓库和生产运行数据分开；不要用本地 SQLite 覆盖生产数据。

## 1. 位置与连接

- GitHub：`https://github.com/Nyarlathoteppppp/bolatu-bot`
- 活跃分支：`live-hotfix-20260917`。先核实 HEAD 和工作区，不按旧文档默认使用 main。
- 本地开发仓库：`/Users/ywbw/Documents/Codex/2026-09-23/ni-q/work/qqbot-git`
- 生产仓库：`/opt/qq-social-agent`
- 早期笔记目录：`/Users/ywbw/workplace/qqbot`，不能当作当前 Git 仓库。
- SSH 别名：`qqbot-server`，用户 `ubuntu`，服务器 `124.223.165.120`，本机私钥 `~/.ssh/xbwmac.pem`。

```bash
ssh qqbot-server
cd /opt/qq-social-agent
git branch --show-current
git rev-parse --short HEAD
git status --short
```

管理台本机隧道：

```bash
ssh -N -L 8080:127.0.0.1:8080 qqbot-server
```

管理台 `http://127.0.0.1:8080/admin`；健康检查 `http://127.0.0.1:8080/readyz`。NapCat WebUI 如确需访问，可另加 `-L 6099:127.0.0.1:6099`，地址 `http://127.0.0.1:6099/webui`。

## 2. 阅读顺序

1. `bot.py`、`config.yaml`：启动与配置。
2. `qq_social_agent/plugin.py`：OneBot 事件入口、运行时装配、群消息队列、阶段交接、命令和生命周期。
3. `group_discourse_flow.py` → `discourse_state.py` → `speaker_context.py`：解析谁在说、对谁说、指代和引用。
4. `group_decision_flow.py` → `timing_gate.py` / `jev_client.py`：是否开口、动作与工具计划。
5. `group_tool_execution.py`、`group_generation_context.py` → `group_reply_generation.py`：执行工具、拼上下文、生成与复核。
6. `group_approval_dispatch.py`、`approval_*`、`approved_reply_delivery.py`：审批与实际发送。
7. `deepseek_client.py`、`llm_gateway.py`、`prompts.py`、`prompts/zhangfengxue.yaml`：任务提示词、模型传输和角色。
8. `interaction_state.py`、`self_interaction_context.py`、`generation_message_context.py`：互动依据、自身发言连续性与消息选择。

上述是职责阅读顺序；实际调用和早退分支以入口函数为准。

## 3. 常改功能应该看哪里

| 功能 | 主要文件 | 分工 |
| --- | --- | --- |
| 开口时机、普通插话概率 | `timing_gate.py`、`jev_client.py`、`deepseek_client.py`、`group_decision_flow.py`、`config.yaml` | JEV 给观察值，本地策略路由；可选机会抽样后才调用 LLM 复核或直接生成 |
| 引用、他/她、这个/那个 | `discourse_state.py`、`reference_resolver.py`、`ellipsis_resolver.py`、`discourse_effects.py`、`speaker_context.py` | 唯一话语状态；已绑定对象不能被普通歧义检查推翻，明确纠正可使其失效 |
| 角色和口吻 | `prompts/zhangfengxue.yaml`、`persona.py`、`prompts.py` | 角色正文、运行规则和各生成 flow 位于同一个 YAML 的不同字段；保留原卡措辞，修改运行规则需核对生成链路 |
| 回复生成、复核 | `group_reply_generation.py`、`pre_send_critic.py`、`pronoun_guard.py`、`deepseek_client.py` | 生成器写正文；critic 比较草稿与已解析状态，人称检查只核对归属 |
| 上下文与延续性 | `generation_message_context.py`、`group_generation_context.py`、`interaction_state.py`、`self_interaction_context.py` | 动态选消息、固定引用证据、读取实际已发送内容；不把未发出的计划写成历史 |
| 模型添加和切换 | `llm_gateway.py`、`config.py`、`config.yaml`、`plugin.py` 的模型命令 | gateway 管 provider/API/任务路由；task client 管提示词和解析；QQ 切换结果存在运行数据库 |
| 搜索和其他工具 | `conversation_tool_routing.py`、`tool_router.py`、`tool_registry.py`、`group_tool_execution.py`、`tools/` | 路由、执行、观察结果与群聊/私聊共用入口 |
| 图片与附件 | `image_read_state.py`、`media_context.py`、`content_ingestion.py`、`siliconflow_ocr.py`、`qq_social_agent/tools/file_content_reader.py` | 原消息存在事实、识图任务状态和结果关联；不能因识别未完成就认为没发图 |
| 私聊 | `private_turn_preparation.py`、`private_session_service.py`、`private_tool_execution.py`、`private_generation_context.py`、`private_reply_delivery.py` | 准备、每用户队列、工具、上下文、生成发送与 followup |
| 审批和发送 | `approval_command_service.py`、`approval_state_service.py`、`approval_request_service.py`、`approved_reply_delivery.py`、`delivery.py` | 命令权限、待审批状态、请求、实际投递与发送回执分层 |
| 长期记忆、风格、RAG | `memory.py`、`memory_learning.py`、`memory_maintenance_service.py`、`rag_*.py`、`background_learning.py` | SQLite 原文与画像、学习、召回和维护；群友表达是参考，不强制风雪附和 |
| 后台批处理 | `openrouter_batch.py`、`memory_maintenance_service.py`、`llm_gateway.py` | 后台模型与实时回复路由分开；具体支持方式看代码和配置 |
| 定时任务 | `*_scheduler_service.py`、`daily_review_service.py`、`proactive_group_message_service.py` | 调度生命周期与生成投递分开；是否启用看 manifest/config/运行开关 |
| 管理台 | `admin_controller.py`、`admin_*_controller.py`、`admin_ui.py`、`admin_edit_controller.py` 的 `AdminEditableFileService` | HTTP 控制器、页面渲染、文件编辑服务 |
| 观测 | `observability.py`、`pipeline_types.py`、`pipeline_stages.py`、`plugin.py` 状态路由 | Trace、阶段状态、用量指标、健康检查 |
| 本地插件声明 | `plugins/*/manifest.yaml`、`plugin_runtime.py`、`plugin.py` | manifest 声明能力，实现显式注册；不是任意动态 Python 插件执行 |

## 4. 当前行为配置与运行数据

- 普通消息进入判断的工作强度仍默认 100%；明确艾特、回复和已确认接续保持原优先队列。
- JEV 弱机会直接沉默；实际问答和已确认接续沿原路由。可选社交机会统一按 `llm.interjection_probability=0.30` 抽样，高分社交机会也参与。抽中后，高把握直接生成，中间候选才交 LLM 判断。JEV 分数不送入 LLM。
- 不因一个代词没绑定姓名就反问；只有必要信息会改变答案时才澄清。人物画像和长期记忆仍要求可靠绑定。
- 已删除生产自定义黑话 `ai → 指代：有可能指张风雪`。仅提到 AI 不由这条词典映射成点名风雪。
- 回复默认 `lingsuan/gpt-6.1-sol`，请求 `medium`；判断和其他文本任务请求 `low`。灵算回退只到 DS 官方。
- 搜索仍 DS，配置当前为 `siliconflow/deepseek-ai/DeepSeek-V4-Flash`；图片识别仍 DS 官方及原读图回退链路。
- QQ 模型覆盖、工作强度、群启停和自定义黑话存在 SQLite，不能只看 YAML 判断实时状态。此次改动保留用户当前群启停状态。
- 角色仍是风雪、北大计算机大二女生、想申请 CMU；角色卡为彩羽风格的群聊适配，没有动漫人物关系。
- `data/bot.sqlite3` 是运行数据，`messages` 记录原消息，`bot_metric_events` 记录事件指标，`app_kv` 记录运行开关，`custom_jargon_entries` 记录用户维护的黑话。

本轮回放的限制：无必要的“他是谁”追问已经减少，但复杂引用包装仍出现过昵称被当成发言内容的误读；critic 对这段上下文返回 `UNCERTAIN`，未阻止发送。人物绑定的回归通过不等于所有语义解释都正确。Claude 可优先审查 `_format_message()`、引用包装、生成上下文及 critic 对身份标签与正文的区分；直接加一句昵称说明在此次回放中没有解决误读，因此未留下这条额外提示。

## 5. 测试与发布

本地：

```bash
cd /Users/ywbw/Documents/Codex/2026-09-23/ni-q/work/qqbot-git
/tmp/qqbot-context-venv/bin/python -m pytest -q --tb=short
git diff --check
```

生产 pytest 在宿主机执行，bot 容器没有 pytest：

```bash
cd /opt/qq-social-agent
PYTHONPATH=. python3 -m pytest -q --tb=short
```

源码通过 bind mount 生效：`/opt/qq-social-agent/qq_social_agent → /app/qq_social_agent`。普通 Python/Prompt/config 改动只重建 bot：

```bash
cd /opt/qq-social-agent
docker inspect -f '{{.State.StartedAt}}' napcat
docker compose -p qq-social-agent -f docker-compose.server.yml up -d --no-deps --force-recreate bot
curl -fsS http://127.0.0.1:8080/readyz
docker inspect -f '{{.State.StartedAt}}' napcat
```

NapCat 前后启动时间必须相同；不得执行 `compose down` 或重启 NapCat。依赖与镜像修改另行评估，不能把普通源码部署当成依赖更新。

`docs/module_boundaries.md` 可看职责边界；`docs/interaction_state.md`、`docs/persona_prompt_organization.md`、`docs/image_read_state.md`、`docs/llm_gateway.md` 可看领域说明。早期运行手册的 main 分支和容器 pytest 说明已过时，以当前仓库、宿主机验证和本说明为准。

## 6. 完整 Git 文件目录

下列清单只列 Git 源码与文档，不包含 `.git/`、`.env`、SQLite、媒体、模型密钥、缓存和 NapCat 登录态。

```text
qqbot-git/
├── docs/
│   ├── audits/
│   │   └── 2026-08-28-project-audit.md
│   ├── career/
│   │   └── quant_industry_guide_2026.md
│   ├── engineering_runbook.md
│   ├── group_chat_evolution.md
│   ├── group_participation.md
│   ├── image_read_state.md
│   ├── interaction_state.md
│   ├── llm_gateway.md
│   ├── memory_lifecycle.md
│   ├── module_boundaries.md
│   ├── persona_prompt_organization.md
│   ├── project_structure.md
│   └── storage_maintenance.md
├── plugins/
│   ├── admin_ui/
│   │   └── manifest.yaml
│   ├── cos_backup/
│   │   └── manifest.yaml
│   ├── daily_review/
│   │   └── manifest.yaml
│   ├── fresh_search/
│   │   └── manifest.yaml
│   ├── market_tools/
│   │   └── manifest.yaml
│   ├── message_chain/
│   │   └── manifest.yaml
│   ├── proactive_chat/
│   │   └── manifest.yaml
│   ├── probability_tools/
│   │   └── manifest.yaml
│   ├── rag_memory/
│   │   └── manifest.yaml
│   ├── social_actions/
│   │   └── manifest.yaml
│   └── README.md
├── prompts/
│   ├── zhangfengxue.yaml
│   └── zhangfengxue_rewrite_worksheet.md
├── qq_social_agent/
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── deep_content.py
│   │   ├── file_content_reader.py
│   │   ├── fresh_context.py
│   │   ├── market.py
│   │   ├── market_intent.py
│   │   ├── probability_tool.py
│   │   ├── safe_url_reader.py
│   │   └── voice_transcript.py
│   ├── __init__.py
│   ├── admin_controller.py
│   ├── admin_edit_controller.py
│   ├── admin_http.py
│   ├── admin_memory_controller.py
│   ├── admin_summaries_controller.py
│   ├── admin_tools_controller.py
│   ├── admin_ui.py
│   ├── approval_command_service.py
│   ├── approval_models.py
│   ├── approval_request_service.py
│   ├── approval_rules.py
│   ├── approval_state_service.py
│   ├── approved_reply_delivery.py
│   ├── background_learning.py
│   ├── config.py
│   ├── content_ingestion.py
│   ├── context_assembler.py
│   ├── conversation_tool_routing.py
│   ├── cue_patterns.py
│   ├── daily_review_scheduler_service.py
│   ├── daily_review_service.py
│   ├── decision_gate.py
│   ├── deepseek_client.py
│   ├── delivery.py
│   ├── discourse_effects.py
│   ├── discourse_state.py
│   ├── ellipsis_resolver.py
│   ├── embedding_client.py
│   ├── generation_message_context.py
│   ├── group_approval_dispatch.py
│   ├── group_decision_flow.py
│   ├── group_directory.py
│   ├── group_discourse_flow.py
│   ├── group_generation_context.py
│   ├── group_jargon.py
│   ├── group_reply_generation.py
│   ├── group_tool_execution.py
│   ├── history_sync.py
│   ├── image_read_state.py
│   ├── interaction_state.py
│   ├── jev_client.py
│   ├── jev_policy.py
│   ├── llm_gateway.py
│   ├── media_context.py
│   ├── member_context.py
│   ├── meme_library.py
│   ├── memory.py
│   ├── memory_learning.py
│   ├── memory_maintenance_service.py
│   ├── message_segments.py
│   ├── notice_events.py
│   ├── observability.py
│   ├── onebot_gateway.py
│   ├── openrouter_batch.py
│   ├── persona.py
│   ├── pipeline_stages.py
│   ├── pipeline_types.py
│   ├── plugin.py
│   ├── plugin_runtime.py
│   ├── political_guard.py
│   ├── pre_send_critic.py
│   ├── private_admin_command_service.py
│   ├── private_context_window.py
│   ├── private_generation_context.py
│   ├── private_message_types.py
│   ├── private_reply_delivery.py
│   ├── private_session_service.py
│   ├── private_tool_execution.py
│   ├── private_turn_preparation.py
│   ├── proactive_chat_scheduler_service.py
│   ├── proactive_group_message_service.py
│   ├── prompts.py
│   ├── pronoun_guard.py
│   ├── rag_admin.py
│   ├── rag_indexer.py
│   ├── rag_query.py
│   ├── rag_retriever.py
│   ├── rag_router.py
│   ├── rag_store.py
│   ├── rate_limiter.py
│   ├── reference_resolver.py
│   ├── reply_splitter.py
│   ├── resolver_result.py
│   ├── scorer.py
│   ├── self_interaction_context.py
│   ├── siliconflow_ocr.py
│   ├── social_actions.py
│   ├── speaker_context.py
│   ├── temporal_evidence.py
│   ├── timing_gate.py
│   ├── tool_observation.py
│   ├── tool_registry.py
│   ├── tool_router.py
│   └── weekly_usage_report_scheduler_service.py
├── scripts/
│   ├── codex_proxy_tunnel.sh
│   ├── cos-backup.env.example
│   ├── cos_backup.sh
│   ├── cos_ntqq_gradual_backup.sh
│   ├── daily_history_archive.py
│   ├── daily_history_archive.sh
│   ├── db_hygiene.py
│   ├── direct_http_proxy.py
│   ├── dirty_work_report.py
│   ├── discourse_replay.py
│   ├── eval_dialogue_routing.py
│   ├── import_owner_memes.py
│   ├── ingest_quant_knowledge.py
│   ├── install_coscli.sh
│   ├── install_server_maintenance_cron.sh
│   ├── restart_bot.sh
│   ├── server_health_report.sh
│   ├── start_bot.sh
│   ├── start_bot_daemon.sh
│   ├── start_napcat.sh
│   ├── status.sh
│   ├── stop_bot.sh
│   ├── system_hygiene.sh
│   └── unify_kedai_memory.py
├── tests/
│   ├── fixtures/
│   │   ├── dialogue_correctness_cases.json
│   │   └── dialogue_routing_cases.json
│   ├── test_admin_controller.py
│   ├── test_admin_ui.py
│   ├── test_approval_services.py
│   ├── test_config.py
│   ├── test_content_ingestion.py
│   ├── test_conversation_tool_routing.py
│   ├── test_cue_patterns.py
│   ├── test_daily_review_batch_service.py
│   ├── test_deep_content.py
│   ├── test_deepseek_client.py
│   ├── test_delivery.py
│   ├── test_dialogue_boundaries.py
│   ├── test_dialogue_correctness_cases.py
│   ├── test_discourse_effects.py
│   ├── test_discourse_state.py
│   ├── test_ellipsis_resolver.py
│   ├── test_file_content_reader.py
│   ├── test_first_batch_social_features.py
│   ├── test_fresh_context.py
│   ├── test_generation_context_loading.py
│   ├── test_generation_message_context.py
│   ├── test_group_direct_mention.py
│   ├── test_group_flow_stages.py
│   ├── test_group_jargon.py
│   ├── test_hardening_fixes.py
│   ├── test_image_history_link.py
│   ├── test_image_read_state.py
│   ├── test_interaction_state.py
│   ├── test_jev_client.py
│   ├── test_jev_failure_policy.py
│   ├── test_jev_policy.py
│   ├── test_jev_reliability_regressions.py
│   ├── test_llm_gateway.py
│   ├── test_market_intent.py
│   ├── test_market_tool.py
│   ├── test_meme_library.py
│   ├── test_memory_learning.py
│   ├── test_memory_maintenance_batch.py
│   ├── test_memory_store.py
│   ├── test_memory_v2.py
│   ├── test_message_segments_and_gateway.py
│   ├── test_notice_events.py
│   ├── test_observability.py
│   ├── test_openrouter_batch.py
│   ├── test_persona_prompt_organization.py
│   ├── test_pipeline_refactor.py
│   ├── test_plugin_fresh_context.py
│   ├── test_plugin_low_value.py
│   ├── test_plugin_runtime.py
│   ├── test_political_guard.py
│   ├── test_pre_send_critic.py
│   ├── test_private_context_window.py
│   ├── test_private_memory_state.py
│   ├── test_private_message_stages.py
│   ├── test_private_session_service.py
│   ├── test_pronoun_guard.py
│   ├── test_rag.py
│   ├── test_rag_admin.py
│   ├── test_rate_limiter.py
│   ├── test_reference_resolver.py
│   ├── test_reply_split.py
│   ├── test_safe_url_reader.py
│   ├── test_scorer.py
│   ├── test_search_deadlines.py
│   ├── test_self_memory_query.py
│   ├── test_social_actions_v2.py
│   ├── test_status_endpoints.py
│   ├── test_timing_hybrid.py
│   ├── test_trace_observability.py
│   ├── test_transport_reliability.py
│   └── test_voice_transcript.py
├── .dockerignore
├── .env.example
├── .gitignore
├── AI_PROJECT_GUIDE.md
├── Dockerfile.server
├── README.md
├── SERVER_DEPLOY.md
├── bot.py
├── config.yaml
├── docker-compose.server.yml
└── pyproject.toml
```
