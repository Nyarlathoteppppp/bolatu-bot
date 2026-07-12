import sqlite3

import pytest

from qq_social_agent.memory import MemoryStore


LEGACY_MEMORY_ATOMS_SCHEMA = """
create table memory_atoms (
  id integer primary key autoincrement,
  atom_type text not null,
  group_id integer not null,
  subject_user_id integer,
  object_user_id integer,
  content text not null,
  source text not null,
  confidence real not null default 0.7,
  importance real not null default 0.5,
  expires_at real,
  created_at real not null,
  updated_at real not null
)
"""


def test_memory_v2_migrates_legacy_atoms_without_rewriting_identity(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(LEGACY_MEMORY_ATOMS_SCHEMA)
    conn.execute(
        """
        insert into memory_atoms(
          atom_type, group_id, subject_user_id, content, source,
          confidence, importance, expires_at, created_at, updated_at
        )
        values ('relation', 1, 100, '旧记忆仍然保留', 'manual:100', 0.8, 0.9, null, 100, 110)
        """
    )
    conn.commit()
    conn.close()

    memory = MemoryStore(path)
    atom = memory.memory_atom(1)

    assert atom is not None
    assert atom.id == 1
    assert atom.content == "旧记忆仍然保留"
    assert atom.evidence_type == "manual"
    assert atom.observed_at == 100
    assert atom.valid_from == 100
    assert atom.status == "active"
    assert [event.action for event in memory.memory_atom_audit_trail(1)] == ["migrated"]

    memory.conn.close()
    reopened = MemoryStore(path)
    assert reopened.memory_atom(1).content == "旧记忆仍然保留"
    assert [event.action for event in reopened.memory_atom_audit_trail(1)] == ["migrated"]


def test_memory_atom_records_message_evidence_and_validity(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    atom_id = memory.add_memory_atom(
        atom_type="relation",
        group_id=1,
        subject_user_id=100,
        object_user_id=200,
        content="甲和乙是同事。",
        source="group_message",
        evidence_type="message",
        source_message_id=42,
        observed_at=1_000,
        valid_from=900,
        valid_to=2_000,
        confidence=0.85,
        importance=0.7,
    )

    atom = memory.memory_atom(atom_id)
    assert atom.evidence_source == "message"
    assert atom.source_message_id == "42"
    assert atom.observed_at == 1_000
    assert atom.valid_from == 900
    assert atom.valid_to == 2_000
    assert atom.expires_at == 2_000
    assert atom.status == "active"

    events = memory.memory_atom_events(atom_id)
    assert len(events) == 1
    assert events[0].action == "created"
    assert events[0].source_message_id == "42"


def test_counter_evidence_disputes_atom_and_is_auditable(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    atom_id = memory.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="甲现在住在南京。",
        source="message:10",
        evidence_type="message",
        source_message_id=10,
        observed_at=100,
    )

    event_id = memory.add_memory_counter_evidence(
        atom_id,
        content="甲说自己已经搬去杭州。",
        source="message:11",
        evidence_type="message",
        source_message_id=11,
        observed_at=200,
        confidence=0.95,
    )

    assert event_id > 0
    assert memory.memory_atom(atom_id).status == "disputed"
    assert memory.relevant_memory_atoms(1, "南京", now=300) == []
    event = memory.memory_atom_events(atom_id)[-1]
    assert event.action == "counter_evidence"
    assert event.detail == "甲说自己已经搬去杭州。"
    assert event.metadata == {"confidence": 0.95}


def test_manual_correction_supersedes_old_atom_and_preserves_lineage(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    old_id = memory.add_memory_atom(
        atom_type="preference",
        group_id=1,
        subject_user_id=100,
        content="甲不吃辣。",
        source="message:20",
        evidence_type="message",
        source_message_id=20,
        observed_at=100,
        confidence=0.7,
        importance=0.8,
    )

    new_id = memory.correct_memory_atom(
        old_id,
        content="甲现在可以吃微辣。",
        actor_user_id=999,
        source="manual:999",
        source_message_id=21,
        observed_at=300,
        reason="本人纠正",
        confidence=1.0,
    )

    old = memory.memory_atom(old_id)
    new = memory.memory_atom(new_id)
    assert old.status == "superseded"
    assert old.valid_to == 300
    assert new.status == "active"
    assert new.supersedes_id == old_id
    assert new.evidence_type == "manual"
    assert new.content == "甲现在可以吃微辣。"
    assert new.confidence == 1.0
    old_events = memory.memory_atom_events(old_id)
    assert [event.action for event in old_events] == ["created", "superseded"]
    assert old_events[-1].metadata == {"replacement_atom_id": new_id}
    assert memory.memory_atom_events(new_id)[0].action == "manual_correction"
    assert memory.memory_atom_events(new_id)[0].actor_user_id == 999
    assert memory.correct_memory_atom(old_id, content="不能从旧节点再次分叉") == 0


def test_expiry_is_soft_and_legacy_delete_api_still_hides_atom(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    atom_id = memory.upsert_memory_atom(
        atom_type="note",
        group_id=1,
        content="临时活动今晚结束。",
        source="manual:1",
        observed_at=50,
        expires_at=200,
    )

    assert memory.relevant_memory_atoms(1, "活动", now=100)
    assert memory.relevant_memory_atoms(1, "活动", now=201) == []
    assert memory.memory_atom(atom_id).status == "active"
    assert memory.expire_due_memory_atoms(now=201, group_id=1) == 1
    assert memory.memory_atom(atom_id).status == "expired"
    assert memory.memory_atom_events(atom_id)[-1].action == "expired"

    second_id = memory.upsert_memory_atom(
        atom_type="note",
        group_id=1,
        content="通过旧 API 删除。",
        source="manual:1",
    )
    assert memory.delete_memory_atom(second_id)
    assert memory.memory_atom(second_id).status == "expired"
    assert all(atom.id != second_id for atom in memory.recent_memory_atoms(1, 20))


def test_retrieval_scoring_covers_keyword_recency_people_importance_and_feedback(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    now = 1_000_000.0

    keyword_id = memory.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="小鸟最近喜欢爵士乐。",
        source="message:1",
        evidence_type="message",
        observed_at=100,
        importance=0.1,
    )
    memory.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="完全无关的旧信息。",
        source="message:2",
        evidence_type="message",
        observed_at=100,
        importance=0.1,
    )
    assert memory.relevant_memory_atoms(1, "爵士乐", now=now, limit=1)[0].id == keyword_id

    relation_id = memory.add_memory_atom(
        atom_type="relation",
        group_id=2,
        subject_user_id=184589072,
        object_user_id=100,
        content="两人是熟悉的群友。",
        source="manual",
        observed_at=100,
        importance=0.1,
    )
    memory.add_memory_atom(
        atom_type="note",
        group_id=2,
        content="普通背景。",
        source="manual",
        observed_at=now,
        importance=0.1,
    )
    assert memory.relevant_memory_atoms(2, "", speaker_user_id=184589072, now=now, limit=1)[0].id == relation_id

    important_id = memory.add_memory_atom(
        atom_type="note",
        group_id=3,
        content="高重要度背景。",
        source="manual",
        observed_at=100,
        importance=1.0,
    )
    recent_id = memory.add_memory_atom(
        atom_type="note",
        group_id=3,
        content="刚刚发生的背景。",
        source="manual",
        observed_at=now,
        importance=0.1,
    )
    ranked = memory.relevant_memory_atoms(3, "", now=now, limit=2)
    assert {atom.id for atom in ranked} == {important_id, recent_id}
    assert ranked[0].id == important_id

    feedback_id = memory.add_memory_atom(
        atom_type="feedback",
        group_id=4,
        content="不准奏反馈：不要客服腔。",
        source="approval_reject:999",
        evidence_type="event",
        observed_at=100,
        importance=0.1,
    )
    memory.add_memory_atom(
        atom_type="note",
        group_id=4,
        content="普通低权重记录。",
        source="event",
        evidence_type="event",
        observed_at=100,
        importance=0.1,
    )
    assert memory.relevant_memory_atoms(4, "", now=now, limit=1)[0].id == feedback_id
    assert memory.relevant_memory_atoms(4, "完全无关的查询", now=now) == []


def test_upsert_is_noop_without_new_evidence_and_infers_legacy_manual_source(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    atom_id = memory.upsert_memory_atom(
        atom_type="feedback",
        group_id=1,
        content="优质反馈：保持自然。",
        source="manual_positive:999",
        confidence=0.9,
        importance=0.8,
    )
    first = memory.memory_atom(atom_id)
    first_events = memory.memory_atom_events(atom_id)

    same_id = memory.upsert_memory_atom(
        atom_type="feedback",
        group_id=1,
        content="优质反馈：保持自然。",
        source="manual_positive:999",
        confidence=0.9,
        importance=0.8,
    )

    assert same_id == atom_id
    assert memory.memory_atom(atom_id).evidence_type == "manual"
    assert memory.memory_atom(atom_id).updated_at == first.updated_at
    assert memory.memory_atom_events(atom_id) == first_events


def test_memory_lifecycle_changes_rollback_if_audit_write_fails(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    atom_id = memory.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="原始事实。",
        source="manual",
        observed_at=100,
        confidence=0.7,
    )
    memory.conn.execute(
        """
        create trigger reject_memory_audit
        before insert on memory_atom_audit_events
        begin
          select raise(abort, 'audit blocked');
        end
        """
    )
    memory.conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="audit blocked"):
        memory.add_memory_counter_evidence(
            atom_id,
            content="反证。",
            source="message:2",
            source_message_id=2,
        )
    assert memory.memory_atom(atom_id).status == "active"

    with pytest.raises(sqlite3.IntegrityError, match="audit blocked"):
        memory.expire_memory_atom(atom_id, observed_at=200)
    assert memory.memory_atom(atom_id).status == "active"

    with pytest.raises(sqlite3.IntegrityError, match="audit blocked"):
        memory.upsert_memory_atom(
            atom_type="fact",
            group_id=1,
            content="原始事实。",
            source="manual",
            confidence=0.95,
        )
    assert memory.memory_atom(atom_id).confidence == 0.7

    count_before = memory.conn.execute("select count(*) from memory_atoms").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError, match="audit blocked"):
        memory.add_memory_atom(
            atom_type="fact",
            group_id=1,
            content="不应留下的无审计事实。",
            source="manual",
        )
    count_after = memory.conn.execute("select count(*) from memory_atoms").fetchone()[0]
    assert count_after == count_before
    assert not memory.conn.in_transaction


def test_lifecycle_rejects_invalid_lineage_and_time_windows(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    atom_id = memory.add_memory_atom(
        atom_type="fact",
        group_id=1,
        content="从时间 100 开始有效。",
        source="manual",
        observed_at=100,
    )
    other_group_id = memory.add_memory_atom(
        atom_type="fact",
        group_id=2,
        content="另一个群的事实。",
        source="manual",
        observed_at=100,
    )

    with pytest.raises(ValueError, match="cannot predate"):
        memory.correct_memory_atom(atom_id, content="错误回溯纠正。", observed_at=50)
    with pytest.raises(ValueError, match="cannot predate"):
        memory.expire_memory_atom(atom_id, observed_at=50)
    with pytest.raises(ValueError, match="valid_to"):
        memory.upsert_memory_atom(
            atom_type="fact",
            group_id=1,
            content="从时间 100 开始有效。",
            source="manual",
            expires_at=50,
        )
    with pytest.raises(ValueError, match="same group"):
        memory.add_memory_atom(
            atom_type="fact",
            group_id=1,
            content="错误跨群替换。",
            source="manual",
            supersedes_id=other_group_id,
        )
    assert memory.memory_atom(atom_id).status == "active"
    assert memory.memory_atom(other_group_id).status == "active"
