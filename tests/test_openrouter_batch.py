import asyncio
import json

import httpx
import pytest

from qq_social_agent.memory import MemoryStore
from qq_social_agent.openrouter_batch import (
    OpenRouterBatchError,
    OpenRouterBatchService,
)


def test_batch_submit_poll_persists_result_until_acknowledged(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload["endpoint"] == "/v1/chat/completions"
            assert payload["model"] == "z-ai/glm-5.3-flash"
            assert payload["requests"][0]["custom_id"] == "memory:12:20-30"
            assert payload["requests"][0]["body"]["model"] == "z-ai/glm-5.3-flash"
            return httpx.Response(200, json={"id": "batch-123", "status": "validating"})
        if len(calls) == 2:
            return httpx.Response(200, json={"id": "batch-123", "status": "in_progress"})
        return httpx.Response(
            200,
            json={
                "id": "batch-123",
                "status": "completed",
                "results": [
                    {
                        "custom_id": "memory:12:20-30",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {"message": {"content": "总结完成"}}
                                ]
                            },
                        },
                    }
                ],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = OpenRouterBatchService(memory, api_key="test-key", http_client=client)
            body = {"messages": [{"role": "user", "content": "summarize"}], "max_tokens": 100}
            metadata = {"group_id": 12, "start_id": 20, "end_id": 30}

            assert await service.request(
                "memory:12:20-30", body=body, task="mid_memory", metadata=metadata
            ) is None
            assert calls == [("POST", "/api/v1/batches")]

            assert await service.request(
                "memory:12:20-30", body=body, task="mid_memory", metadata=metadata
            ) is None

            result = await service.poll("memory:12:20-30")
            assert result.status == "completed"
            assert result.content == "总结完成"
            assert result.metadata == metadata
            assert await service.request(
                "memory:12:20-30", body=body, task="mid_memory", metadata=metadata
            ) == "总结完成"

            snapshot = service.status_snapshot(task="mid_memory")
            assert snapshot["counts"] == {"completed": 1}
            assert service.pending_jobs(task="mid_memory") == []
            service.acknowledge("memory:12:20-30")
            assert service.status_snapshot(task="mid_memory")["counts"] == {"acknowledged": 1}
            assert (await service.poll("memory:12:20-30")).status == "acknowledged"

    asyncio.run(run())


def test_recreated_service_resumes_existing_batch_without_resubmitting(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    calls: list[str] = []

    def submit_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, json={"id": "batch-resume", "status": "queued"})

    def poll_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, json={"id": "batch-resume", "status": "in_progress"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(submit_handler)) as first_client:
            first = OpenRouterBatchService(memory, api_key="test-key", http_client=first_client)
            assert await first.submit(
                "daily_review:100:2026-09-27",
                {"messages": [{"role": "user", "content": "review"}]},
                task="daily_review",
                metadata={"group_id": 100, "label": "2026-09-27"},
            ) == "batch-resume"

        async with httpx.AsyncClient(transport=httpx.MockTransport(poll_handler)) as second_client:
            restarted = OpenRouterBatchService(memory, api_key="test-key", http_client=second_client)
            same_id = await restarted.submit(
                "daily_review:100:2026-09-27",
                {"messages": [{"role": "user", "content": "ignored after first submit"}]},
                task="daily_review",
            )
            assert same_id == "batch-resume"
            result = await restarted.poll("daily_review:100:2026-09-27")
            assert result.status == "in_progress"
            assert result.metadata == {"group_id": 100, "label": "2026-09-27"}

        assert calls == ["POST", "GET"]

    asyncio.run(run())


def test_ambiguous_post_is_not_submitted_twice(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("response lost after server accepted request")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = OpenRouterBatchService(memory, api_key="test-key", http_client=client)
            with pytest.raises(OpenRouterBatchError) as first_error:
                await service.submit("daily_review:100:today", {"messages": []})
            assert first_error.value.ambiguous is True
            assert first_error.value.job_status == "submission_uncertain"

            with pytest.raises(OpenRouterBatchError, match="refusing to submit a duplicate"):
                await service.submit("daily_review:100:today", {"messages": []})
            assert calls == 1
            assert service.status_snapshot()["counts"] == {"submission_uncertain": 1}

    asyncio.run(run())


def test_probe_uses_a_stable_batch_key_and_model_normalization(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            assert request.url.path == "/api/v1/batches/probe-batch"
            return httpx.Response(200, json={"id": "probe-batch", "status": "validating"})
        assert request.url.path == "/api/v1/batches"
        payload = json.loads(request.content)
        assert payload["model"] == "z-ai/glm-5.3-flash"
        assert payload["requests"][0]["body"]["max_tokens"] == 16
        return httpx.Response(200, json={"id": "probe-batch", "status": "validating"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = OpenRouterBatchService(
                memory,
                api_key="test-key",
                model="openrouter/z-ai/glm-5.3-flash:batch",
                http_client=client,
            )
            first = await service.probe()
            second = await service.probe()
            assert service.model == "z-ai/glm-5.3-flash"
            assert first.key == second.key
            assert first.key.startswith("batch_probe:glm_flash:")
            assert first.batch_id == second.batch_id == "probe-batch"
            assert first.status == second.status == "queued"
            assert calls == ["POST", "GET"]

    asyncio.run(run())


def test_completed_probe_is_refreshed_on_next_test(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json={"id": f"batch-{len(calls)}", "status": "queued"})
        key = memory.app_kv_get("openrouter_batch_probe_active:batch_probe:glm_flash")
        return httpx.Response(200, json={
            "id": "batch-1", "status": "completed",
            "results": [{"custom_id": key, "response": {"status_code": 200, "body": {
                "choices": [{"message": {"content": "batch ok"}}]
            }}}],
        })

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = OpenRouterBatchService(memory, api_key="test-key", http_client=client)
            first = await service.probe()
            completed = await service.probe()
            refreshed = await service.probe()
            assert completed.status == "completed"
            assert refreshed.key != first.key
            assert calls == ["POST", "GET", "POST"]

    asyncio.run(run())


def test_failed_job_can_be_acknowledged_after_sync_fallback(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "batch-failed", "status": "queued"})
        return httpx.Response(200, json={"id": "batch-failed", "status": "failed", "error": "provider error"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = OpenRouterBatchService(memory, api_key="test-key", http_client=client)
            await service.submit("review:failed", {"messages": []})
            assert (await service.poll("review:failed")).status == "failed"
            service.acknowledge("review:failed")
            service.acknowledge("review:failed")
            assert (await service.poll("review:failed")).status == "acknowledged"

    asyncio.run(run())
