from __future__ import annotations

from qq_social_agent.discourse_effects import (
    AmbiguityJudgement,
    MemoryCandidate,
    MemoryEffectJudgement,
    RepairJudgement,
    RepairTarget,
    AmbiguityResolution,
    RepairResolution,
    apply_jev_ambiguity_judgement,
    apply_jev_memory_judgement,
    apply_jev_repair_judgement,
    apply_memory_effect,
    apply_repair_to_reference,
    format_ambiguity_prompt_block,
    resolve_ordinal_item,
    should_ask_jev_ambiguity,
    should_ask_jev_repair,
)
from qq_social_agent.ellipsis_resolver import EllipsisResolution
from qq_social_agent.memory import ChatMessage, MemoryStore
from qq_social_agent.reference_resolver import ReferenceResolution


BIRD = 184589072
WANG = 2001


def _named(text: str):
    ids = []
    if "小鸟" in text:
        ids.append(BIRD)
    if "小王" in text:
        ids.append(WANG)
    return tuple(ids)


def _msg(text: str, *, uid: int = 7, nick: str = "甲", mid: str = "1") -> ChatMessage:
    return ChatMessage(1, uid, nick, text, False, 1.0, source_message_id=mid)


def test_repair_referent_overrides_old_binding() -> None:
    old = ReferenceResolution((BIRD,), reason="previous_named_member", confidence=0.9, kind="PERSON")
    assert should_ask_jev_repair("不是小鸟，是小王", reference=old, named_in_text=_named("不是小鸟，是小王"))
    targets = [RepairTarget(key="t_referent", kind="referent", summary="小鸟", user_id=BIRD)]
    judged = RepairJudgement(kind="REFERENT", target_key="t_referent", confidence=0.88)
    repair = apply_jev_repair_judgement(
        judged,
        targets,
        current_text="不是小鸟，是小王",
        reference=old,
        resolve_named_users=_named,
    )
    assert repair.kind == "REFERENT"
    assert repair.replacement_user_ids == (WANG,)
    updated = apply_repair_to_reference(old, repair)
    assert updated.user_ids == (WANG,)
    assert updated.unresolved is False


def test_repair_item_uses_existing_ordinal_resolver() -> None:
    messages = [_msg("第一套要重写缓存，第二套更稳但慢", mid="10")]
    item = resolve_ordinal_item("我说的是第二个", messages)
    assert "第二套" in item
    targets = [RepairTarget(key="t_previous", kind="message", summary="第一套要重写缓存，第二套更稳但慢")]
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="ITEM", target_key="t_previous", confidence=0.8),
        targets,
        current_text="我说的是第二个",
        recent_messages=messages,
    )
    assert repair.kind == "ITEM"
    assert "第二套" in repair.replacement_item
    assert repair.unresolved is False


def test_repair_intent_without_replacement_stays_unresolved() -> None:
    targets = [RepairTarget(key="t_bot", kind="intent", summary="风雪把这当成要查价格")]
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="INTENT", target_key="t_bot", confidence=0.7),
        targets,
        current_text="不是这个意思",
    )
    assert repair.kind == "INTENT"
    assert repair.unresolved is True
    assert repair.reason == "intent_unspecified"


def test_repair_retraction_marks_target() -> None:
    targets = [RepairTarget(key="t_previous", kind="message", summary="我要去杭州")]
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="RETRACTION", target_key="t_previous", confidence=0.9),
        targets,
        current_text="刚才那句当我没说",
    )
    assert repair.kind == "RETRACTION"
    assert repair.unresolved is False


def test_memory_refine(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="小鸟准备考研", source="test", subject_user_id=BIRD)
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=BIRD, content="小鸟准备考计算机研究生")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="REFINE", target_id=f"mem{old_id}", confidence=0.8),
        related,
        candidate=candidate,
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert applied.applied
    assert store.memory_atom(old_id).status == "superseded"
    assert store.memory_atom(applied.target_atom_id).content == "小鸟准备考计算机研究生"
    assert store.memory_atom(applied.target_atom_id).status == "active"


def test_memory_replace(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="准备去上海", source="test")
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=None, content="后来改杭州")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="REPLACE", target_id=f"mem{old_id}", confidence=0.82),
        related,
        candidate=candidate,
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert store.memory_atom(old_id).status == "superseded"
    assert "杭州" in store.memory_atom(applied.target_atom_id).content


def test_memory_negate(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="小鸟准备考研", source="test", subject_user_id=BIRD)
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=BIRD, content="小鸟不考了")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="NEGATE", target_id=f"mem{old_id}", confidence=0.85),
        related,
        candidate=candidate,
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert store.memory_atom(old_id).status == "expired"
    assert store.memory_atom(applied.target_atom_id).content == "小鸟不考了"


def test_memory_retract_is_not_negate(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="我要去杭州", source="test")
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=7, content="刚才说我要去杭州那句当我没说")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="RETRACT", target_id=f"mem{old_id}", confidence=0.9),
        related,
        candidate=candidate,
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert applied.applied
    assert store.memory_atom(old_id).status == "expired"
    assert applied.target_atom_id == old_id


def test_memory_confirm_does_not_duplicate(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="小鸟准备考研", source="test", subject_user_id=BIRD)
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=BIRD, content="小鸟准备考研")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="CONFIRM", target_id=f"mem{old_id}", confidence=0.7),
        related,
        candidate=candidate,
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert applied.applied
    assert store.memory_atom(old_id).status == "active"
    assert store.recent_memory_atoms(1, 10)[0].id == old_id


def test_memory_low_confidence_does_not_mutate(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(atom_type="fact", group_id=1, content="小鸟准备考研", source="test")
    related = [store.memory_atom(old_id)]
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="REPLACE", target_id=f"mem{old_id}", confidence=0.4),
        related,
        candidate=MemoryCandidate(subject_user_id=None, content="去杭州"),
    )
    applied = apply_memory_effect(
        store,
        judged,
        group_id=1,
        candidate=MemoryCandidate(subject_user_id=None, content="去杭州"),
    )
    assert applied.applied is False
    assert applied.unresolved is True
    assert store.memory_atom(old_id).status == "active"


def test_ambiguity_person_two_women() -> None:
    judged = apply_jev_ambiguity_judgement(
        AmbiguityJudgement(kind="PERSON", confidence=0.8),
        candidate_labels=["小鸟", "小王"],
    )
    assert judged.kind == "PERSON"
    block = format_ambiguity_prompt_block(judged)
    assert 'kind=PERSON' in block
    assert "小鸟" in block
    assert "你说的是" not in block


def test_ambiguity_item() -> None:
    judged = apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="ITEM", confidence=0.77))
    assert judged.kind == "ITEM"


def test_ambiguity_thread() -> None:
    judged = apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="THREAD", confidence=0.7))
    assert judged.kind == "THREAD"


def test_ambiguity_ellipsis_when_referent_known() -> None:
    judged = apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="ELLIPSIS", confidence=0.74))
    assert judged.kind == "ELLIPSIS"


def test_ambiguity_time_scope_intent_memory() -> None:
    assert apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="TIME", confidence=0.7)).kind == "TIME"
    assert apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="SCOPE", confidence=0.7)).kind == "SCOPE"
    assert apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="INTENT", confidence=0.7)).kind == "INTENT"
    assert apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="MEMORY_REFERENCE", confidence=0.7)).kind == "MEMORY_REFERENCE"


def test_ambiguity_trigger_from_unresolved_flags() -> None:
    assert should_ask_jev_ambiguity(reference=ReferenceResolution(unresolved=True, kind="PERSON"))
    assert should_ask_jev_ambiguity(ellipsis=EllipsisResolution(kind="SAME_PREDICATE", unresolved=True))
    assert should_ask_jev_ambiguity(speaking_action="clarify")


def test_non_person_this_does_not_ask_item_ambiguity() -> None:
    reference = ReferenceResolution(kind="NON_PERSON", reason="jev_non_person", confidence=0.49, status="RESOLVED")
    ellipsis = EllipsisResolution(kind="NONE", status="NOT_APPLICABLE")
    assert should_ask_jev_ambiguity(reference=reference, ellipsis=ellipsis) is False
    linked = EllipsisResolution(
        kind="ITEM_DEIXIS",
        source_text="1折",
        status="RESOLVED",
        confidence=0.86,
    )
    assert should_ask_jev_ambiguity(reference=reference, ellipsis=linked) is False


def test_ambiguity_does_not_fire_on_memory_unresolved() -> None:
    from qq_social_agent.discourse_effects import MemoryEffectResolution, RepairResolution

    assert should_ask_jev_ambiguity(
        memory_effect=MemoryEffectResolution(action="REPLACE", unresolved=True)
    ) is False
    assert should_ask_jev_ambiguity(
        repair=RepairResolution(kind="FACT", unresolved=True)
    ) is True


def test_repair_targets_prefer_reply_bot_and_related_not_last_unrelated() -> None:
    from qq_social_agent.discourse_effects import build_repair_targets
    from qq_social_agent.reference_resolver import ReplyHint

    messages = [
        _msg("小鸟以前准备考研", uid=7, nick="甲", mid="10"),
        _msg("今天食堂好挤", uid=8, nick="乙", mid="11"),
    ]
    reply = ReplyHint(exists=True, author_id=7, author_label="甲", text="小鸟以前准备考研", message_id="10")
    reference = ReferenceResolution((BIRD,), reason="previous_named_member", confidence=0.9, kind="PERSON")
    rows = build_repair_targets(
        recent_messages=messages,
        reply=reply,
        reference=reference,
        related_memories=[],
        recent_bot_text="你是说小鸟还考吗",
        current_text="不是小鸟，是小王",
    )
    keys = [row.key for row in rows]
    assert "t_reply" in keys
    assert "t_bot" in keys
    assert "t_referent" in keys
    assert "t_previous" not in keys
    summaries = " ".join(row.summary for row in rows)
    assert "食堂" not in summaries


def test_repair_low_confidence_does_not_change_reference() -> None:
    old = ReferenceResolution((BIRD,), reason="previous_named_member", confidence=0.9, kind="PERSON")
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="REFERENT", target_key="t_referent", confidence=0.4),
        [RepairTarget(key="t_referent", kind="referent", summary="小鸟", user_id=BIRD)],
        current_text="不是小鸟，是小王",
        reference=old,
        resolve_named_users=_named,
    )
    assert repair.status == "AMBIGUOUS"
    assert repair.unresolved is True
    updated = apply_repair_to_reference(old, repair)
    assert updated.user_ids == (BIRD,)
    assert updated.reason == "previous_named_member"


def test_repair_referent_invalidates_and_resets_ellipsis() -> None:
    from qq_social_agent.discourse_effects import repair_invalidates, reset_ellipsis_for_repair

    old = ReferenceResolution((BIRD,), reason="previous_named_member", confidence=0.9, kind="PERSON")
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="REFERENT", target_key="t_referent", confidence=0.88),
        [RepairTarget(key="t_referent", kind="referent", summary="小鸟", user_id=BIRD)],
        current_text="不是小鸟，是小王",
        reference=old,
        resolve_named_users=_named,
    )
    updated = apply_repair_to_reference(old, repair)
    assert updated.user_ids == (WANG,)
    assert updated.source == "repair"
    assert updated.status == "RESOLVED"
    assert repair_invalidates(repair, "ellipsis")
    ellipsis = EllipsisResolution(
        kind="SAME_PREDICATE",
        source_key="s10",
        source_text="小鸟以前准备考研",
        status="RESOLVED",
        source="jev",
    )
    reset = reset_ellipsis_for_repair(ellipsis, repair)
    assert reset.status == "NOT_APPLICABLE"
    assert reset.reason == "invalidated_by_repair"
    assert reset.source_text == ""


def test_jev_unavailable_is_not_not_applicable() -> None:
    repair = apply_jev_repair_judgement(None, [], current_text="不是小鸟")
    assert repair.status == "UNAVAILABLE"
    assert repair.status != "NOT_APPLICABLE"
    none_repair = apply_jev_repair_judgement(
        RepairJudgement(kind="NONE", target_key="NONE", confidence=0.9),
        [RepairTarget(key="t_reply", kind="message", summary="x")],
        current_text="哈哈",
    )
    assert none_repair.status == "NOT_APPLICABLE"
    ambiguity = apply_jev_ambiguity_judgement(AmbiguityJudgement(kind="PERSON", confidence=0.8))
    assert ambiguity.status == "AMBIGUOUS"
    unavail = apply_jev_ambiguity_judgement(None)
    assert unavail.status == "UNAVAILABLE"
    assert unavail.status != "AMBIGUOUS"
    unavailable_ref = ReferenceResolution(kind="PERSON", unresolved=True, reason="jev_unavailable")
    assert unavailable_ref.status == "UNAVAILABLE"
    assert should_ask_jev_ambiguity(reference=unavailable_ref) is False


def test_memory_commit_ignore_does_not_require_reply(tmp_path) -> None:
    from qq_social_agent.discourse_effects import memory_can_commit

    store = MemoryStore(tmp_path / "m.sqlite3")
    old_id = store.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="小鸟准备考研",
        source="test",
        subject_user_id=BIRD,
    )
    related = [store.memory_atom(old_id)]
    candidate = MemoryCandidate(subject_user_id=BIRD, content="小鸟不考了")
    judged = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="NEGATE", target_id=f"mem{old_id}", confidence=0.85),
        related,
        candidate=candidate,
    )
    reference = ReferenceResolution((BIRD,), reason="named_in_current", confidence=0.92, kind="PERSON")
    assert memory_can_commit(
        judged,
        candidate=candidate,
        reference=reference,
        repair=RepairResolution(),
        ambiguity=AmbiguityResolution(),
    )
    applied = apply_memory_effect(store, judged, group_id=1, candidate=candidate)
    assert applied.applied
    assert store.memory_atom(old_id).status == "expired"


def test_repair_invalidation_clears_dependent_states() -> None:
    from qq_social_agent.discourse_effects import (
        AmbiguityResolution,
        MemoryEffectResolution,
        apply_repair_invalidation,
        memory_can_commit,
    )

    old = ReferenceResolution((BIRD,), reason="previous_named_member", confidence=0.9, kind="PERSON")
    repair = apply_jev_repair_judgement(
        RepairJudgement(kind="REFERENT", target_key="t_referent", confidence=0.88),
        [RepairTarget(key="t_referent", kind="referent", summary="小鸟", user_id=BIRD)],
        current_text="不是小鸟，是小王",
        reference=old,
        resolve_named_users=_named,
    )
    ellipsis = EllipsisResolution(
        kind="SAME_PREDICATE",
        source_key="s10",
        source_text="小鸟以前准备考研",
        status="RESOLVED",
        source="jev",
    )
    memory_effect = MemoryEffectResolution(action="CREATE", status="RESOLVED", source="jev")
    ambiguity = AmbiguityResolution(kind="PERSON", status="AMBIGUOUS", source="jev")
    reference, ellipsis, memory_effect, ambiguity, invalidated = apply_repair_invalidation(
        repair=repair,
        reference=old,
        ellipsis=ellipsis,
        memory_effect=memory_effect,
        ambiguity=ambiguity,
    )
    assert reference.user_ids == (WANG,)
    assert "ellipsis" in invalidated
    assert "memory" in invalidated
    assert "ambiguity" in invalidated
    assert ellipsis.source_text == ""
    assert ellipsis.status == "NOT_APPLICABLE"
    assert memory_effect.action == "IGNORE"
    assert memory_can_commit(
        MemoryEffectResolution(action="CREATE", status="RESOLVED"),
        candidate=MemoryCandidate(subject_user_id=WANG, content="小王准备考研"),
        reference=reference,
        repair=repair,
        ambiguity=ambiguity,
        pending_recompute=invalidated,
    ) is False


def test_memory_does_not_commit_while_referent_unavailable() -> None:
    from qq_social_agent.discourse_effects import memory_can_commit

    create = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="CREATE", target_id="NONE", confidence=0.9),
        [],
        candidate=MemoryCandidate(subject_user_id=BIRD, content="小鸟准备考研"),
    )
    unavailable = ReferenceResolution(kind="PERSON", unresolved=True, reason="jev_unavailable")
    assert unavailable.status == "UNAVAILABLE"
    assert memory_can_commit(
        create,
        candidate=MemoryCandidate(subject_user_id=BIRD, content="小鸟准备考研"),
        reference=unavailable,
        repair=RepairResolution(),
        ambiguity=AmbiguityResolution(),
    ) is False


def test_memory_does_not_commit_joke_or_bad_referent() -> None:
    from qq_social_agent.discourse_effects import memory_can_commit

    candidate = MemoryCandidate(subject_user_id=BIRD, content="小鸟准备考研")
    ignore = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="IGNORE", target_id="NONE", confidence=0.9),
        [],
        candidate=candidate,
    )
    reference = ReferenceResolution((BIRD,), reason="named_in_current", confidence=0.92, kind="PERSON")
    assert memory_can_commit(
        ignore,
        candidate=candidate,
        reference=reference,
        repair=RepairResolution(),
        ambiguity=AmbiguityResolution(),
    ) is False
    create = apply_jev_memory_judgement(
        MemoryEffectJudgement(action="CREATE", target_id="NONE", confidence=0.9),
        [],
        candidate=candidate,
    )
    bad_ref = ReferenceResolution(kind="PERSON", unresolved=True, reason="jev_ambiguous")
    assert memory_can_commit(
        create,
        candidate=candidate,
        reference=bad_ref,
        repair=RepairResolution(),
        ambiguity=AmbiguityResolution(),
    ) is False
