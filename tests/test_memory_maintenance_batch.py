from __future__ import annotations

import asyncio
from types import SimpleNamespace

from qq_social_agent.deepseek_client import MidMemoryDraft
from qq_social_agent.memory import MemoryStore
from qq_social_agent.memory_maintenance_service import (
    MemoryMaintenancePolicy,
    MemoryMaintenanceService,
)


def _policy() -> MemoryMaintenancePolicy:
    return MemoryMaintenancePolicy(
        group_context_limit=0,
        private_context_limit=0,
        private_chat_offset=10_000_000,
        mid_memory_batch_size=20,
        mid_memory_min_batch=2,
        mid_memory_retry_interval_seconds=1,
        mid_memory_empty_skip_streak=3,
        style_learn_interval_seconds=10_000,
        style_learn_message_limit=20,
        style_learn_candidate_limit=100,
        style_learn_per_user_limit=5,
        style_learn_min_messages=100,
        member_profile_interval_seconds=10_000,
        member_profile_lookback_seconds=10_000,
        member_profile_active_limit=0,
        member_profile_min_messages=100,
        member_profile_message_limit=100,
        member_profile_min_chars=100,
    )


class _BatchService:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object]] = {}
        self.submitted: list[str] = []
        self.acknowledged: list[str] = []

    async def submit(self, key, body, *, task, metadata):
        self.submitted.append(key)
        self.jobs[key] = {
            "body": body,
            "task": task,
            "metadata": metadata,
            "status": "queued",
            "content": None,
            "error": None,
        }
        return "batch-1"

    async def poll(self, key):
        job = self.jobs[key]
        return SimpleNamespace(
            status=job["status"],
            content=job["content"],
            error=job["error"],
        )

    def acknowledge(self, key):
        self.acknowledged.append(key)
        self.jobs.pop(key, None)


class _Client:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.explicit_model_calls = 0
        self.parse_messages = []

    def build_mid_memory_request(self, *, messages, chat_label):
        return {
            "messages": [
                {"role": "user", "content": "|".join(item.text for item in messages)}
            ],
            "chat_label": chat_label,
        }

    def parse_mid_memory_response(self, content, *, messages):
        self.parse_messages = list(messages)
        return MidMemoryDraft(summary=content, recall_cues=("batch",))

    async def summarize_mid_memory(self, **kwargs):
        self.sync_calls += 1
        raise AssertionError("batch selection must not use the synchronous path")

    async def complete_on_model(self, **kwargs):
        self.explicit_model_calls += 1
        raise AssertionError("a pending batch must be polled after a model switch")


class _SyncClient(_Client):
    async def complete_on_model(self, **kwargs):
        self.explicit_model_calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="同步模型摘要"))]
        )


def _service(memory, client, batch, *, batch_enabled, memory_sync_model=None):
    events = []
    service = MemoryMaintenanceService(
        memory_provider=lambda: memory,
        client_provider=lambda: client,
        policy_provider=_policy,
        record_metric_event=lambda event, **payload: events.append((event, payload)),
        member_label=lambda user_id, nickname: nickname,
        useful_style_rule=lambda *_args: True,
        batch_service=batch,
        batch_mode_for_task=lambda task: batch_enabled and task == "memory",
        memory_sync_model=memory_sync_model,
    )
    return service, events


def test_mid_memory_batch_submits_without_waiting_then_resumes_after_restart(tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(5):
        memory.add_message(77, 100 + index, f"u{index}", f"message-{index}", created_at=100 + index)

    batch = _BatchService()
    client = _Client()
    service, events = _service(memory, client, batch, batch_enabled=True)

    asyncio.run(service.maintain_group(77))

    assert len(batch.submitted) == 1
    job_key = batch.submitted[0]
    assert memory.recent_memory_summaries(77, 5) == []
    assert memory.app_kv_get(service._mid_memory_batch_pointer_key(77))
    assert client.sync_calls == 0
    assert events == []

    restarted_service, restarted_events = _service(
        memory,
        client,
        batch,
        batch_enabled=False,
    )

    asyncio.run(restarted_service.maintain_group(77))

    assert memory.recent_memory_summaries(77, 5) == []
    assert memory.app_kv_get(service._mid_memory_batch_pointer_key(77))
    assert client.sync_calls == 0
    assert client.explicit_model_calls == 0
    assert batch.submitted == [job_key]

    batch.jobs[job_key]["status"] = "completed"
    batch.jobs[job_key]["content"] = "已完成的群聊摘要"
    asyncio.run(restarted_service.maintain_group(77))

    summaries = memory.recent_memory_summaries(77, 5)
    assert [item.summary for item in summaries] == ["已完成的群聊摘要"]
    assert [item.id for item in client.parse_messages] == [1, 2, 3, 4, 5]
    assert client.sync_calls == 0
    assert client.explicit_model_calls == 0
    assert batch.acknowledged == [job_key]
    assert memory.app_kv_get(service._mid_memory_batch_pointer_key(77)) == ""
    assert len(restarted_events) == 1
    assert restarted_events[0][0] == "mid_memory_learning"
    assert restarted_events[0][1]["action"] == "persisted"


def test_failed_mid_memory_batch_clears_pointer_without_ack(tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(5):
        memory.add_message(99, 300 + index, f"u{index}", f"message-{index}", created_at=300 + index)

    batch = _BatchService()
    client = _Client()
    service, _events = _service(memory, client, batch, batch_enabled=True)
    asyncio.run(service.maintain_group(99))
    job_key = batch.submitted[0]
    batch.jobs[job_key]["status"] = "failed"
    batch.jobs[job_key]["error"] = "remote failure"

    asyncio.run(service.maintain_group(99))

    assert memory.recent_memory_summaries(99, 5) == []
    assert memory.app_kv_get(service._mid_memory_batch_pointer_key(99)) == ""
    assert batch.acknowledged == []


def test_mid_memory_sync_selection_uses_selected_route(tmp_path):
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(5):
        memory.add_message(88, 200 + index, f"u{index}", f"message-{index}", created_at=200 + index)

    client = _SyncClient()
    service, _events = _service(
        memory,
        client,
        _BatchService(),
        batch_enabled=False,
        memory_sync_model=lambda: object(),
    )

    asyncio.run(service.maintain_group(88))

    assert client.explicit_model_calls == 1
    assert client.sync_calls == 0
    assert [item.summary for item in memory.recent_memory_summaries(88, 5)] == [
        "同步模型摘要"
    ]
