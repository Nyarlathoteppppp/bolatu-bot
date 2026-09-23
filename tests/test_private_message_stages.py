"""Behavioral checks for the staged private-message pipeline."""

import asyncio
from types import SimpleNamespace

import nonebot

nonebot.init()

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from qq_social_agent import plugin
from qq_social_agent.deepseek_client import ReplyDecision
from qq_social_agent.memory import MemoryStore
from qq_social_agent.private_message_types import PrivateGenerationContext, PrivateTurn
from qq_social_agent.private_reply_delivery import PrivateReplyServices, generate_and_send_private_reply
from qq_social_agent.private_tool_execution import PrivateToolServices, plan_and_execute_private_tools
from qq_social_agent.private_turn_preparation import PrivateTurnServices, prepare_private_turn
from qq_social_agent.tool_router import ToolRoutePlan


class FakeContentIngestion:
    def __init__(self, text="", *, file_count=0, voice_count=0):
        self.result = SimpleNamespace(
            text=text,
            file_count=file_count,
            voice_count=voice_count,
            file_status="ok" if file_count else "none",
            voice_status="ok" if voice_count else "none",
        )

    async def context_for_event(self, *args, **kwargs):
        return self.result


class FakeRag:
    def __init__(self):
        self.sync_requests = 0

    def request_source_sync(self):
        self.sync_requests += 1

    async def retrieve(self, **kwargs):
        return SimpleNamespace(
            plan=SimpleNamespace(route="lexical"),
            hits=(),
            context="",
            elapsed_ms=1,
            lexical_count=0,
            semantic_count=0,
            error="",
        )


def _event(message_id, text, *, user_id=987654321, self_id=1801507496, message=None):
    return SimpleNamespace(
        message_id=message_id,
        user_id=user_id,
        self_id=self_id,
        time=1_800_000_000,
        message=message if message is not None else Message(text),
        sender=SimpleNamespace(nickname="奈亚子", card="", role="", title=""),
        raw_message=text,
        message_type="private",
        sub_type="friend",
    )


def _preparation_services(memory, *, rag=None, content=None, **overrides):
    async def empty_text(*args, **kwargs):
        return ""

    async def no_media(*args, **kwargs):
        return False

    async def no_ocr(*args, **kwargs):
        from qq_social_agent.media_context import ImageOcrContext

        return ImageOcrContext("", 0, 0)

    async def no_forward(*args, **kwargs):
        return ""

    async def identity_text(text, **kwargs):
        return text

    async def no_send(*args, **kwargs):
        return None

    async def no_approval(*args, **kwargs):
        return False

    values = dict(
        memory=memory,
        rag_service=rag or FakeRag(),
        content_ingestion=content or FakeContentIngestion(),
        approval_handler=no_approval,
        private_user_can_chat=lambda user_id: True,
        private_chat_id=plugin._private_chat_id,
        source_message_id=lambda event: str(event.message_id),
        private_nickname=lambda event: event.sender.nickname,
        file_metadata_context=empty_text,
        media_worth_reading=no_media,
        image_ocr_context_for_event=no_ocr,
        format_image_ocr_context=lambda result: f"[图片OCR: {result.text}]",
        message_has_forward_context=lambda event: False,
        forward_context_text=no_forward,
        force_obey_command_response=lambda user_id, text: None,
        extract_force_obey_once_text=lambda user_id, text: None,
        force_obey_context=lambda user_id, **kwargs: "",
        message_text_for_context=identity_text,
        event_message_storage_kwargs=lambda event, **kwargs: {"session_id": f"private:{event.user_id}"},
        schedule_private_memory_maintenance=lambda chat_id: None,
        send_private_message=no_send,
        record_metric_event=lambda *args, **kwargs: None,
        message_factory=Message,
        short_notice_text=plugin._short_notice_text,
        private_context_reset_commands=frozenset(plugin.PRIVATE_CONTEXT_RESET_COMMANDS),
        command_only_private_user_ids=frozenset(),
        long_message_summary_threshold=1000,
        logger=plugin.logger,
    )
    values.update(overrides)
    return PrivateTurnServices(**values)


def test_private_duplicate_is_claimed_once_and_approval_precedes_chat_permission(monkeypatch, tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    approval_calls = []
    permission_calls = []
    metrics = []

    async def intercept(bot, user_id, text):
        approval_calls.append((user_id, text))
        return True

    monkeypatch.setattr(plugin, "memory", memory)
    monkeypatch.setattr(plugin, "_message_context_text", lambda event, **kwargs: "A")
    monkeypatch.setattr(plugin, "_handle_group_approval_private", intercept)
    monkeypatch.setattr(
        plugin,
        "_private_user_can_chat",
        lambda user_id: permission_calls.append(user_id) or False,
    )
    monkeypatch.setattr(plugin, "_record_metric_event", lambda *args, **kwargs: metrics.append((args, kwargs)))
    bot = SimpleNamespace(self_id=1801507496)
    event = _event("duplicate-1", "A")

    async def run():
        await plugin._handle_private_message_scoped(bot, event, correlation_id="c1")
        await plugin._handle_private_message_scoped(bot, event, correlation_id="c2")

    asyncio.run(run())
    assert approval_calls == [(987654321, "A")]
    assert permission_calls == []
    assert metrics[0][0] == ("message_duplicate",)
    assert memory.recent_messages(plugin._private_chat_id(987654321), 10) == []


def test_private_media_context_is_prepared_and_memory_stays_in_synthetic_chat(tmp_path):
    from qq_social_agent.media_context import ImageOcrContext

    memory = MemoryStore(tmp_path / "bot.sqlite3")
    event = _event(
        "media-1",
        "看看",
        message=Message("看看") + MessageSegment.image(file="score.png"),
    )
    rag = FakeRag()
    metrics = []

    async def metadata(*args, **kwargs):
        return "[文件: 成绩单.pdf]"

    async def media_worth(**kwargs):
        return kwargs["kind"] == "ocr"

    async def ocr(*args, **kwargs):
        assert kwargs["group_id"] == plugin._private_chat_id(event.user_id)
        return ImageOcrContext("GPA 3.8", 1, 1)

    services = _preparation_services(
        memory,
        rag=rag,
        content=FakeContentIngestion("[语音转写: 我想问一下]", file_count=1, voice_count=1),
        file_metadata_context=metadata,
        media_worth_reading=media_worth,
        image_ocr_context_for_event=ocr,
        record_metric_event=lambda *args, **kwargs: metrics.append((args, kwargs)),
    )
    bot = SimpleNamespace(self_id=1801507496)
    turn = asyncio.run(prepare_private_turn(
        bot,
        event,
        correlation_id="media-correlation",
        received_text="看看",
        buffered_messages=None,
        services=services,
    ))

    assert turn is not None
    assert "[文件: 成绩单.pdf]" in turn.text
    assert "[语音转写: 我想问一下]" in turn.text
    assert "GPA 3.8" in turn.text
    assert rag.sync_requests == 1
    assert any(item[0][0] == "content_ingestion" for item in metrics)
    private_messages = memory.recent_messages(turn.chat_id, 10)
    assert [(message.user_id, message.text) for message in private_messages] == [(event.user_id, turn.text.replace("[对方连续发了多条私聊]\n", ""))]
    memory.add_message(42, event.user_id, "奈亚子", "群聊隔离", is_bot=False)
    assert [message.text for message in memory.recent_messages(42, 10)] == ["群聊隔离"]
    assert "群聊隔离" not in [message.text for message in memory.recent_messages(turn.chat_id, 10)]


def test_private_tool_stage_uses_the_shared_router_and_preserves_private_scope(monkeypatch, tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    chat_id = plugin._private_chat_id(987654321)
    memory.set_group_enabled(chat_id, True)
    turn = PrivateTurn(
        bot=SimpleNamespace(self_id=1801507496),
        user_id=987654321,
        chat_id=chat_id,
        self_id=1801507496,
        nickname="奈亚子",
        source_message_id="tool-1",
        correlation_id="tool-correlation",
        text="帮我看看 NVDA 最近怎么样",
        forced_once_context="",
        received_message_count=1,
    )
    observed = {}

    class Rate:
        def allow(self, chat_id, *, mentioned):
            return SimpleNamespace(allowed=True, reason="")

    class Client:
        async def route_tool_use(self, **kwargs):
            observed["speaker_context"] = kwargs["speaker_context"]
            observed["router_text"] = kwargs["current_text"]
            return SimpleNamespace(tool="none", query="", confidence=0.0, reason="no_tool", kind="web", symbols=())

    class Registry:
        async def execute(self, request):
            raise AssertionError(f"unexpected tool request: {request}")

    class Rag(FakeRag):
        async def retrieve(self, **kwargs):
            observed["rag_chat_id"] = kwargs["group_id"]
            return await super().retrieve(**kwargs)

    monkeypatch.setattr(plugin, "deepseek_client", Client())

    services = PrivateToolServices(
        memory=memory,
        rag_service=Rag(),
        rate_limiter=Rate(),
        personas=SimpleNamespace(get=lambda persona_id: SimpleNamespace(name="张风雪")),
        app_config=SimpleNamespace(default_persona="default"),
        get_deepseek_client=lambda: Client(),
        tool_registry=Registry(),
        route_tools=lambda text, **kwargs: ToolRoutePlan(),
        tool_plan_with_runtime_context=lambda plan, **kwargs: plan,
        apply_backend_tool_decision=lambda decision, **kwargs: decision,
        apply_tool_plan=lambda decision, plan: decision,
        is_explicit_market_lookup=lambda text: False,
        market_intents_from_decision=lambda *args, **kwargs: [],
        apply_tool_use_router=plugin._apply_tool_use_router,
        execute_fresh_tool_request=lambda *args, **kwargs: None,
        compact_search_query=lambda text: text,
        fresh_tool_failure_context=lambda *args, **kwargs: "",
        normalize_rag_query=lambda text: SimpleNamespace(current_utterance=text),
        detect_market_intents=lambda *args, **kwargs: [],
        detect_fresh_intent=lambda text: None,
        without_current_message=lambda messages, **kwargs: messages,
        combine_text_sections=lambda *sections: "\n".join(item for item in sections if item),
        private_conversation_state_context=lambda chat_id: "",
        private_priority_context=lambda user_id: "",
        member_label=lambda user_id, nickname: nickname,
        record_metric_event=lambda *args, **kwargs: None,
        logger=plugin.logger,
        private_context_limit=40,
        mid_memory_keep_summaries=4,
    )
    async def run():
        stage = await plan_and_execute_private_tools(turn, services=services)
        assert stage is not None
        await stage.rag_task
        return stage

    stage = asyncio.run(run())
    assert stage is not None
    assert observed["speaker_context"].startswith("当前是和奈亚子的一对一私聊")
    assert observed["router_text"] == turn.text
    assert observed["rag_chat_id"] == chat_id


def test_private_segment_send_failure_keeps_successful_prefix_only(monkeypatch, tmp_path):
    from nonebot.adapters.onebot.v11.exception import ActionFailed
    from qq_social_agent import private_reply_delivery

    memory = MemoryStore(tmp_path / "bot.sqlite3")
    turn = PrivateTurn(
        bot=SimpleNamespace(self_id=1801507496),
        user_id=987654321,
        chat_id=plugin._private_chat_id(987654321),
        self_id=1801507496,
        nickname="奈亚子",
        source_message_id="reply-1",
        correlation_id="reply-correlation",
        text="请回复",
        forced_once_context="",
        received_message_count=1,
    )
    context = PrivateGenerationContext(
        persona=SimpleNamespace(name="张风雪"),
        recent_messages=(),
        current_text=turn.text,
        current_nickname=turn.nickname,
        decision=ReplyDecision(should_reply=True, confidence=1.0, reason="test", mode="reply", action="answer"),
        market_context="",
        fresh_context="",
        memory_context="",
        member_context="",
        memory_atoms_context="",
        style_context="",
        raw_corpus_context="",
        jargon_context="",
        recall_feedback_context="",
        speaker_context="私聊",
        priority_context="",
    )

    class MemeLibrary:
        def turn_gate(self, user_id, *, received_messages):
            return SimpleNamespace(allowed=False, reason="frequency", messages_since_last_meme=1)

    class Client:
        async def reply(self, **kwargs):
            return "回复内容"

    monkeypatch.setattr(private_reply_delivery, "split_reply_messages", lambda *args, **kwargs: ["第一段", "第二段"])
    sent = []

    async def send(bot, *, user_id, message):
        sent.append(str(message))
        if len(sent) == 2:
            raise ActionFailed(status="failed", retcode=120, message="blocked")

    services = PrivateReplyServices(
        memory=memory,
        private_meme_library=MemeLibrary(),
        get_deepseek_client=lambda: Client(),
        send_private_message=send,
        message_with_reply_quote=lambda message, message_id: message,
        sanitize_generated_text=lambda text: text,
        private_meme_context_eligible=lambda **kwargs: False,
        action_failed_summary=lambda exc: str(exc),
        record_metric_event=lambda *args, **kwargs: None,
        logger=plugin.logger,
    )
    asyncio.run(generate_and_send_private_reply(turn, context, services=services))
    assert len(sent) == 2
    stored = memory.recent_messages(turn.chat_id, 10)
    assert [(message.text, message.is_bot) for message in stored] == [("第一段", True)]
