import json
import time

from qq_social_agent.deepseek_client import DailyReviewDraft, MemoryFactDraft, MidMemoryDraft
from qq_social_agent.memory import ChatMessage, MemoryStore
from qq_social_agent.memory_learning import persist_daily_review_learning, persist_mid_memory_learning


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upsert_memory_atom(self, **kwargs: object) -> int:
        self.calls.append(kwargs)
        return len(self.calls)


def test_mid_learning_requires_valid_evidence_and_confidence() -> None:
    memory = FakeMemory()
    messages = [ChatMessage(1, 100, "甲", "我最近喜欢画画", False, 1000.0, id=41)]
    draft = MidMemoryDraft(
        "摘要",
        (),
        facts=(
            MemoryFactDraft(
                "preference",
                "甲最近喜欢画画",
                subject_user_id=100,
                evidence_message_ids=(41,),
                confidence=0.9,
                importance=0.8,
                valid_for_days=30,
            ),
            MemoryFactDraft("fact", "没有证据", confidence=0.9),
            MemoryFactDraft("fact", "置信度太低", evidence_message_ids=(41,), confidence=0.2),
        ),
    )

    ids = persist_mid_memory_learning(  # type: ignore[arg-type]
        memory,
        group_id=1,
        draft=draft,
        messages=messages,
    )

    assert ids == (1,)
    assert len(memory.calls) == 1
    call = memory.calls[0]
    assert call["atom_type"] == "preference"
    assert call["evidence_type"] == "message"
    assert call["source_message_id"] == "db:41"
    assert call["observed_at"] == 1000.0
    assert call["valid_to"] == 1000.0 + 30 * 24 * 60 * 60


def test_daily_review_persists_evidence_facts_and_event_lessons(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    observed_at = time.time()
    messages = [ChatMessage(1, 100, "甲", "今天换了新工作", False, observed_at, id=71)]
    draft = DailyReviewDraft(
        public_reply="今天群里挺热闹。",
        events=(
            MemoryFactDraft(
                "event",
                "甲今天换了新工作",
                subject_user_id=100,
                evidence_message_ids=(71,),
                confidence=0.9,
                importance=0.8,
                valid_for_days=90,
            ),
        ),
        feedback_lessons=(
            MemoryFactDraft(
                "feedback_lesson",
                "遇到认真求助时少用客服腔",
                confidence=0.95,
                importance=0.9,
            ),
        ),
    )

    ids = persist_daily_review_learning(
        memory,
        group_id=1,
        review_label="2026-07-13",
        draft=draft,
        messages=messages,
    )

    assert len(ids) == 2
    atoms = memory.recent_memory_atoms(1, 10)
    assert {atom.atom_type for atom in atoms} == {"event", "feedback"}
    event = next(atom for atom in atoms if atom.atom_type == "event")
    feedback = next(atom for atom in atoms if atom.atom_type == "feedback")
    assert event.source_message_id == "db:71"
    assert event.evidence_type == "message"
    assert feedback.evidence_type == "event"
    snapshot = json.loads(memory.app_kv_get("daily_review_structured:1:2026-07-13"))
    assert snapshot["public_reply"] == "今天群里挺热闹。"
    assert snapshot["memory_atom_ids"] == list(ids)
