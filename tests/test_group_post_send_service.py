"""Direct post-send ownership tests. They do not import the plugin runtime."""

import asyncio
from types import SimpleNamespace

from qq_social_agent.approval_models import PendingApprovalCandidate, PendingGroupApproval
from qq_social_agent.group_post_send_service import (
    GroupFollowupPolicy,
    GroupPostSendService,
    GroupPostSendServices,
    group_meme_context_eligible,
)
from qq_social_agent import group_post_send_service as post_send
from qq_social_agent.pipeline_types import PipelineState


BASELINE_HARD_SECONDS = 15
BASELINE_SOFT_SECONDS = 40


def _approval(**overrides) -> PendingGroupApproval:
    candidate = overrides.pop("candidate", PendingApprovalCandidate(1, "回复", "reply", ""))
    values = dict(
        approval_id="ap-1",
        group_id=9,
        trigger_user_id=10001,
        trigger_nickname=" 甲 ",
        trigger_text="触发",
        persona_name="风雪",
        self_id=1801507496,
        candidates=(candidate,),
        mention_targets={10001: "甲"},
        created_at=1.0,
        correlation_id="corr-1",
        tool_evidence="",
        source_message_id="src-9",
        pipeline_state=None,
    )
    values.update(overrides)
    return PendingGroupApproval(**values)


def _pipeline(**overrides) -> PipelineState:
    values = dict(
        correlation_id="corr-1",
        group_id=9,
        user_id=10001,
        nickname="甲",
        text="触发",
        addressed=True,
        source_message_id="42",
        decision_action="tease",
        decision_side_reaction="laugh",
        interaction_context_at=1234.5,
    )
    values.update(overrides)
    return PipelineState(**values)


def _logger() -> SimpleNamespace:
    logger = SimpleNamespace(warnings=[], infos=[])
    logger.warning = lambda message: logger.warnings.append(message)
    logger.info = lambda message: logger.infos.append(message)
    return logger


def _memory(asset=SimpleNamespace(id=7, description="猫猫")) -> SimpleNamespace:
    memory = SimpleNamespace(asset=asset, messages=[], observed=[], asset_calls=[], bot_sent=[])

    def meme_asset(meme_id):
        memory.asset_calls.append(meme_id)
        return memory.asset

    memory.meme_asset = meme_asset
    memory.add_message = lambda *args, **kwargs: memory.messages.append((args, kwargs))
    memory.interactions = SimpleNamespace(
        observe_sent=lambda **kwargs: memory.observed.append(kwargs)
    )
    memory.add_bot_sent_message = lambda **kwargs: memory.bot_sent.append(kwargs)
    return memory


def _library(*, allowed=True, reason="eligible", candidates=None, image_ref="base64://meme"):
    library = SimpleNamespace(
        gate=SimpleNamespace(allowed=allowed, reason=reason),
        candidates=[SimpleNamespace(id=7, description="猫猫")] if candidates is None else candidates,
        image_ref=image_ref,
        gate_calls=[],
        queries=[],
        text_calls=[],
        image_calls=[],
        marked=[],
    )
    library.group_gate = lambda group_id: library.gate_calls.append(group_id) or library.gate
    library.group_candidates = lambda group_id, *, query: library.queries.append((group_id, query)) or library.candidates
    library.candidate_text = lambda assets: library.text_calls.append(assets) or "候选说明"
    library.image_base64_ref = lambda meme_id: library.image_calls.append(meme_id) or library.image_ref
    library.mark_group_sent = lambda group_id, meme_id: library.marked.append((group_id, meme_id)) or True
    return library


def _service(
    *,
    memory=None,
    client=None,
    library=None,
    social=None,
    windows=None,
    metrics=None,
    logger=None,
    send=None,
    memory_text=None,
    policy=None,
) -> GroupPostSendService:
    if send is None:
        async def send(*_args, **_kwargs):
            raise AssertionError("send_group_message should not be called")

    if social is None:
        social = SimpleNamespace(calls=[])

        async def react(*_args, **_kwargs):
            raise AssertionError("react_to_message should not be called")

        social.react_to_message = react
    return GroupPostSendService(GroupPostSendServices(
        memory=memory if memory is not None else _memory(),
        client=client,
        meme_library=library if library is not None else _library(),
        social_actions=social,
        followup_windows=windows if windows is not None else {},
        followup_policy=policy or GroupFollowupPolicy(
            hard_seconds=BASELINE_HARD_SECONDS,
            soft_seconds=BASELINE_SOFT_SECONDS,
        ),
        send_group_message=send,
        record_metric_event=metrics if metrics is not None else (lambda *_args, **_kwargs: None),
        memory_text_from_reply_part=memory_text if memory_text is not None else (lambda text, _targets: text),
        logger=logger if logger is not None else _logger(),
    ))


def _install_image_message(monkeypatch):
    def image(*, file):
        return SimpleNamespace(type="image", data={"file": file})

    def message(segment):
        return SimpleNamespace(segment=segment)

    monkeypatch.setattr(post_send, "Message", message)
    monkeypatch.setattr(post_send, "MessageSegment", SimpleNamespace(image=image))


def test_followup_window_does_not_slide_inside_soft_window(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(post_send.time, "time", lambda: clock["now"])
    windows: dict[tuple[int, int], float] = {}
    metrics = []
    service = _service(windows=windows, metrics=lambda name, **kwargs: metrics.append((name, kwargs)))

    service.record_followup_window(1, trigger_user_id=100, mention_user_id=200)
    assert windows is service.services.followup_windows
    assert windows == {(1, 100): 1000.0, (1, 200): 1000.0}
    assert metrics == [("followup_window_opened", {
        "group_id": 1,
        "user_id": 100,
        "stage": "send",
        "action": "post_reply",
        "target_user_ids": [100, 200],
        "opened_user_ids": [100, 200],
        "skipped_refresh_user_ids": [],
        "window_seconds": BASELINE_HARD_SECONDS,
        "soft_window_seconds": BASELINE_SOFT_SECONDS,
    })]

    # Past the hard window, still inside the soft window: do not slide.
    clock["now"] = 1015.0
    service.record_followup_window(1, trigger_user_id=100, mention_user_id=200)
    assert windows[(1, 100)] == 1000.0
    assert windows[(1, 200)] == 1000.0
    assert metrics[-1][1]["opened_user_ids"] == []
    assert metrics[-1][1]["skipped_refresh_user_ids"] == [100, 200]
    assert metrics[-1][1]["window_seconds"] == BASELINE_HARD_SECONDS
    assert metrics[-1][1]["soft_window_seconds"] == BASELINE_SOFT_SECONDS

    # Inclusive soft bound still belongs to the original open.
    clock["now"] = 1040.0
    service.record_followup_window(1, trigger_user_id=100)
    assert windows[(1, 100)] == 1000.0
    assert metrics[-1][1]["skipped_refresh_user_ids"] == [100]
    assert metrics[-1][1]["opened_user_ids"] == []

    clock["now"] = 1030.0
    service.record_followup_window(1, trigger_user_id=100, mention_user_id=300)
    assert windows[(1, 100)] == 1000.0
    assert windows[(1, 300)] == 1030.0
    assert metrics[-1][1]["opened_user_ids"] == [300]
    assert metrics[-1][1]["skipped_refresh_user_ids"] == [100]

    clock["now"] = 1040.1
    service.record_followup_window(1, trigger_user_id=100, mention_user_id=300)
    assert windows[(1, 100)] == 1040.1
    assert windows[(1, 300)] == 1030.0
    assert metrics[-1][1]["opened_user_ids"] == [100]
    assert metrics[-1][1]["skipped_refresh_user_ids"] == [300]

    metric_count = len(metrics)
    clock["now"] = 2000.0
    service.record_followup_window(1, trigger_user_id=100, mention_user_id=300, conversation_engaged=False)
    service.record_followup_window(1, trigger_user_id=0, mention_user_id=0)
    assert windows[(1, 100)] == 1040.1
    assert windows[(1, 300)] == 1030.0
    assert (1, 0) not in windows
    assert len(metrics) == metric_count


def test_meme_context_eligible_keeps_baseline_gates():
    approval = _approval()
    reply = PendingApprovalCandidate(1, "回复", "reply", "")
    assert group_meme_context_eligible(approval, reply)
    assert group_meme_context_eligible(approval, PendingApprovalCandidate(1, "x" * 150, "reply", ""))
    assert not group_meme_context_eligible(approval, PendingApprovalCandidate(1, "x" * 151, "reply", ""))
    assert not group_meme_context_eligible(approval, PendingApprovalCandidate(1, "   ", "reply", ""))
    assert not group_meme_context_eligible(approval, PendingApprovalCandidate(1, "回复", "market_check", ""))
    assert not group_meme_context_eligible(approval, PendingApprovalCandidate(1, "回复", "fresh_context", ""))
    assert not group_meme_context_eligible(_approval(tool_evidence="  依据  "), reply)
    assert group_meme_context_eligible(_approval(tool_evidence="   "), reply)


def test_meme_success_owns_sent_source_and_write_order(monkeypatch):
    _install_image_message(monkeypatch)
    events = []
    memory = _memory()
    library = _library()
    library.mark_group_sent = lambda group_id, meme_id: events.append(("mark", group_id, meme_id)) or True
    memory.meme_asset = lambda meme_id: events.append(("asset", meme_id)) or memory.asset
    memory.add_message = lambda *args, **kwargs: events.append(("memory", args, kwargs))
    memory.interactions.observe_sent = lambda **kwargs: events.append(("observe", kwargs))
    text_calls = []

    async def select(**kwargs):
        events.append(("select", kwargs))
        return SimpleNamespace(send=True, meme_id=7, reason="fits")

    async def send(_bot, group_id, message):
        events.append(("send", group_id, message.segment.data["file"]))
        return 55

    service = _service(
        memory=memory,
        client=SimpleNamespace(select_private_meme=select),
        library=library,
        send=send,
        memory_text=lambda text, targets: text_calls.append((text, targets)) or f"mem:{text}",
        metrics=lambda name, **kwargs: events.append(("metric", name, kwargs)),
    )
    approval = _approval(pipeline_state=_pipeline())
    candidate = approval.candidates[0]

    asyncio.run(service.maybe_send_meme(SimpleNamespace(name="bot"), approval, candidate))

    assert text_calls == [("回复", {10001: "甲"})]
    assert library.queries == [(9, "触发\n回复")]
    assert [item[0] for item in events] == ["select", "send", "mark", "asset", "memory", "observe", "metric"]
    assert events[0][1] == {
        "current_text": "触发",
        "reply_text": "mem:回复",
        "candidates": "候选说明",
    }
    assert events[1][1:] == (9, "base64://meme")
    assert events[2][1:] == (9, 7)
    assert events[4][1] == (9, 1801507496, "风雪", "[风雪附了一张已授权表情包：猫猫]")
    assert events[4][2] == {
        "is_bot": True,
        "source_message_id": 55,
        "source_kind": "live",
        "correlation_id": "corr-1",
    }
    assert events[5][1] == {
        "group_id": 9,
        "source_message_id": "55",
        "trigger_source_id": "src-9",
        "action": "meme",
        "context_at": 1234.5,
    }
    assert events[6][1:] == ("group_meme_selector", {
        "group_id": 9,
        "user_id": 10001,
        "stage": "delivery",
        "action": "sent",
        "meme_id": 7,
        "reason": "fits",
    })


def test_meme_without_message_id_still_writes_but_does_not_observe(monkeypatch):
    _install_image_message(monkeypatch)
    memory = _memory(asset=None)
    library = _library()

    async def select(**_kwargs):
        return SimpleNamespace(send=True, meme_id=7, reason="fits")

    async def send(*_args):
        return None

    metrics = []
    service = _service(
        memory=memory,
        client=SimpleNamespace(select_private_meme=select),
        library=library,
        send=send,
        metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    asyncio.run(service.maybe_send_meme(SimpleNamespace(), _approval(pipeline_state=_pipeline()), _approval().candidates[0]))

    assert library.marked == [(9, 7)]
    assert memory.messages[0][0][3] == "[风雪附了一张已授权表情包：7]"
    assert memory.messages[0][1]["source_message_id"] is None
    assert memory.observed == []
    assert metrics[-1][1]["action"] == "sent"


def test_meme_delivery_failure_does_not_take_success_writes(monkeypatch):
    _install_image_message(monkeypatch)
    memory = _memory()
    library = _library()
    logger = _logger()
    sent = []
    metrics = []
    failure = post_send.ActionFailed(status="failed", retcode=120, message="blocked")
    failure_summary = post_send.action_failed_summary(failure)

    async def select(**_kwargs):
        return SimpleNamespace(send=True, meme_id=7, reason="fits")

    async def send(_bot, group_id, message):
        sent.append((group_id, message.segment.data["file"]))
        raise failure

    service = _service(
        memory=memory,
        client=SimpleNamespace(select_private_meme=select),
        library=library,
        logger=logger,
        send=send,
        metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    asyncio.run(service.maybe_send_meme(SimpleNamespace(), _approval(), _approval().candidates[0]))

    assert sent == [(9, "base64://meme")]
    assert library.marked == []
    assert memory.asset_calls == []
    assert memory.messages == []
    assert memory.observed == []
    assert logger.warnings == [
        f"qq_social_agent failed sending group meme: group=9 meme=7 {failure_summary}"
    ]
    assert metrics == [("group_meme_selector", {
        "group_id": 9,
        "user_id": 10001,
        "stage": "delivery",
        "action": "failed",
        "meme_id": 7,
        "error": failure_summary,
    })]


def test_meme_pre_send_skips_own_no_success_writes():
    memory = _memory()
    library = _library()
    logger = _logger()
    metrics = []
    sends = []
    approval = _approval()
    candidate = approval.candidates[0]

    async def unused_send(*_args):
        sends.append("sent")
        return 1

    async def scenario():
        blocked = _service(memory=memory, client=None, library=library, logger=logger, send=unused_send, metrics=lambda *a, **k: metrics.append((a, k)))
        await blocked.maybe_send_meme(SimpleNamespace(), approval, candidate)
        await blocked.maybe_send_meme(SimpleNamespace(), approval, PendingApprovalCandidate(1, "回复", "market_check", ""))
        await blocked.maybe_send_meme(SimpleNamespace(), _approval(tool_evidence="依据"), candidate)
        assert library.gate_calls == []
        assert metrics == []

        library.gate = SimpleNamespace(allowed=False, reason="group_cooldown")
        gated = _service(
            memory=memory,
            client=SimpleNamespace(),
            library=library,
            logger=logger,
            send=unused_send,
            metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
        )
        await gated.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert library.queries == []
        assert metrics == [("group_meme_selector", {
            "group_id": 9,
            "user_id": 10001,
            "stage": "eligibility",
            "action": "skipped",
            "gate_reason": "group_cooldown",
        })]

        library.gate = SimpleNamespace(allowed=True, reason="eligible")
        library.candidates = []
        empty = _service(memory=memory, client=SimpleNamespace(), library=library, send=unused_send, metrics=lambda name, **kwargs: metrics.append((name, kwargs)))
        await empty.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert len(metrics) == 1
        assert library.image_calls == []

        library.candidates = [SimpleNamespace(id=7, description="猫猫")]

        async def explode(**_kwargs):
            raise RuntimeError("selector down")

        failed = _service(
            memory=memory,
            client=SimpleNamespace(select_private_meme=explode),
            library=library,
            logger=logger,
            send=unused_send,
            metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
        )
        await failed.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert logger.warnings == ["qq_social_agent group meme selector failed: group=9 error=selector down"]
        assert len(metrics) == 1

        async def decline(**_kwargs):
            return SimpleNamespace(send=False, meme_id=7, reason="not-fit")

        skipped = _service(
            memory=memory,
            client=SimpleNamespace(select_private_meme=decline),
            library=library,
            logger=logger,
            send=unused_send,
            metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
        )
        await skipped.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert library.image_calls == []
        assert metrics[-1] == ("group_meme_selector", {
            "group_id": 9,
            "user_id": 10001,
            "stage": "selection",
            "action": "skipped",
            "gate_reason": "eligible",
            "reason": "not-fit",
        })

        library.image_ref = ""

        async def choose_other(**_kwargs):
            return SimpleNamespace(send=True, meme_id=99, reason="missing")

        outsider = _service(
            memory=memory,
            client=SimpleNamespace(select_private_meme=choose_other),
            library=library,
            send=unused_send,
            metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
        )
        await outsider.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert metrics[-1][1]["reason"] == "missing"
        assert library.image_calls == []

        async def choose(**_kwargs):
            return SimpleNamespace(send=True, meme_id=7, reason="fits")

        no_image = _service(
            memory=memory,
            client=SimpleNamespace(select_private_meme=choose),
            library=library,
            send=unused_send,
            metrics=lambda name, **kwargs: metrics.append((name, kwargs)),
        )
        await no_image.maybe_send_meme(SimpleNamespace(), approval, candidate)
        assert library.image_calls == [7]
        assert sends == []
        assert library.marked == []
        assert memory.messages == []
        assert memory.observed == []

    asyncio.run(scenario())


def test_side_reaction_skips_missing_conditions_before_react():
    social = SimpleNamespace(calls=[], result=SimpleNamespace(sent=False, reason="user_cooldown", reaction="agree", emoji_id="76"))
    logger = _logger()
    metrics = []

    async def react(bot, **kwargs):
        social.calls.append((bot, kwargs))
        return social.result

    social.react_to_message = react
    service = _service(social=social, logger=logger, metrics=lambda name, **kwargs: metrics.append((name, kwargs)))
    bot = SimpleNamespace(name="bot")

    async def scenario():
        await service.execute_side_reaction(bot, _approval(pipeline_state=None))
        pipeline = _pipeline(decision_side_reaction="", source_message_id="42")
        approval = _approval(pipeline_state=pipeline)
        await service.execute_side_reaction(bot, approval)
        pipeline.decision_side_reaction = "   "
        await service.execute_side_reaction(bot, approval)
        assert social.calls == []
        assert logger.infos == []
        assert metrics == []

        pipeline.decision_side_reaction = "thumb"
        pipeline.decision_action = "tease"
        pipeline.source_message_id = "12a"
        await service.execute_side_reaction(bot, approval)
        assert social.calls == []
        assert metrics == []
        assert logger.infos == [
            "qq_social_agent approved side reaction skipped: group=9 reason=missing_message_id reaction=thumb"
        ]

        pipeline.source_message_id = " 42 "
        await service.execute_side_reaction(bot, approval)
        assert social.calls == [(bot, {
            "group_id": 9,
            "user_id": 10001,
            "message_id": "42",
            "reaction": "agree",
            "target_label": "甲[#10001]",
        })]
        assert metrics == [("social_action", {
            "group_id": 9,
            "user_id": 10001,
            "stage": "approved_side_reaction",
            "action": "react",
            "reaction": "agree",
            "reason": "user_cooldown",
            "emoji_id": "76",
            "sent": False,
            "approval_id": "ap-1",
        })]
        assert logger.infos[-1] == (
            "qq_social_agent approved side reaction skipped: group=9 message_id=42 reason=user_cooldown"
        )

        social.result = SimpleNamespace(sent=True, reason="sent", reaction="agree", emoji_id="76")
        await service.execute_side_reaction(bot, approval)
        assert logger.infos[-1] == (
            "qq_social_agent approved side reaction sent: "
            "group=9 message_id=42 reaction=agree emoji_id=76"
        )
        assert metrics[-1][1]["sent"] is True

    asyncio.run(scenario())


def test_side_reaction_error_traces_keep_exception_order():
    social = SimpleNamespace(calls=[], error=None)
    logger = _logger()
    metrics = []

    async def react(bot, **kwargs):
        social.calls.append((bot, kwargs))
        raise social.error

    social.react_to_message = react
    pipeline = _pipeline(source_message_id="42", decision_action="tease", decision_side_reaction="laugh")
    approval = _approval(pipeline_state=pipeline)
    service = _service(social=social, logger=logger, metrics=lambda name, **kwargs: metrics.append((name, kwargs)))
    long_text = "Z" * 180
    failure = post_send.ActionFailed(status="failed", retcode=120, message="blocked")
    failure_summary = post_send.action_failed_summary(failure)

    async def scenario():
        social.error = failure
        await service.execute_side_reaction(SimpleNamespace(), approval)
        social.error = RuntimeError(long_text)
        await service.execute_side_reaction(SimpleNamespace(), approval)

    asyncio.run(scenario())

    assert [call[1]["reaction"] for call in social.calls] == ["laugh", "laugh"]
    assert logger.warnings == [
        f"qq_social_agent approved side reaction failed: group=9 message_id=42 {failure_summary}",
        f"qq_social_agent approved side reaction failed: group=9 message_id=42 error={long_text}",
    ]
    assert " error=" not in logger.warnings[0]
    assert [item[0] for item in metrics] == ["social_action_failed", "social_action_failed"]
    assert metrics[0][1]["error"] == failure_summary
    assert metrics[0][1]["reaction"] == "laugh"
    assert metrics[1][1]["error"] == long_text[:160]
    assert metrics[1][1]["error"] != long_text
    assert metrics[1][1]["stage"] == "approved_side_reaction"
    assert metrics[1][1]["action"] == "react"


def test_bot_sent_missing_message_id_does_not_write():
    memory = _memory()
    logger = _logger()
    service = _service(memory=memory, logger=logger)
    payload = dict(
        group_id=9,
        bot_reply="回复",
        trigger_user_id=10001,
        trigger_nickname="甲",
        trigger_text="问",
        action="answer",
    )

    service.record_bot_sent_message(message_id=None, **payload)

    assert memory.bot_sent == []
    assert logger.warnings == ["qq_social_agent bot sent message missing message_id: group=9 action=answer"]

    service.record_bot_sent_message(message_id=55, **payload)

    assert memory.bot_sent == [{**payload, "message_id": 55}]
    assert len(logger.warnings) == 1
