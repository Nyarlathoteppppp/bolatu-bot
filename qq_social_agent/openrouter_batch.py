from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx


OPENROUTER_BATCH_MODEL = "z-ai/glm-5.3-flash"
OPENROUTER_BATCH_BASE_URL = "https://openrouter.ai/api/v1"
_TABLE = "openrouter_batch_jobs"
_ACTIVE_STATUSES = ("submitting", "queued", "in_progress", "submission_uncertain")
_TERMINAL_STATUSES = ("completed", "failed", "expired", "cancelled", "acknowledged")


class OpenRouterBatchError(RuntimeError):
    """An OpenRouter batch request failed or has an ambiguous submission state."""

    def __init__(
        self,
        message: str,
        *,
        job_status: str | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.job_status = job_status
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class BatchPollResult:
    key: str
    task: str
    status: str
    batch_id: str | None
    content: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    updated_at: float = 0.0


class OpenRouterBatchService:
    """Persistent single-request batches for background LLM work.

    A stable caller key identifies one logical unit of work. The key and batch
    id live in SQLite, so polling resumes after a process restart. Completed
    output remains available until the caller applies it and calls
    :meth:`acknowledge`.
    """

    def __init__(
        self,
        memory: Any,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BATCH_BASE_URL,
        model: str = OPENROUTER_BATCH_MODEL,
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.memory = memory
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = self.normalize_model(model)
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client
        self._key_lock = asyncio.Lock()
        self._initialize_table()

    @staticmethod
    def normalize_model(model: str) -> str:
        cleaned = str(model or "").strip()
        if cleaned.startswith("openrouter/"):
            cleaned = cleaned[len("openrouter/"):]
        if cleaned.endswith(":batch"):
            cleaned = cleaned[:-len(":batch")]
        if not cleaned:
            raise ValueError("A batch model is required.")
        return cleaned

    def _initialize_table(self) -> None:
        self.memory.conn.execute(
            f"""
            create table if not exists {_TABLE} (
              job_key text primary key,
              task text not null,
              model text not null,
              batch_id text,
              status text not null,
              request_json text,
              content text,
              error text,
              metadata_json text,
              response_json text,
              created_at real not null,
              updated_at real not null
            )
            """
        )
        self.memory.conn.execute(
            f"create index if not exists idx_{_TABLE}_status_task on {_TABLE}(status, task)"
        )
        self.memory.conn.commit()

    def _require_key(self) -> str:
        if not self.api_key:
            raise OpenRouterBatchError("OPENROUTER_API_KEY is not configured.")
        return self.api_key

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _row(self, key: str) -> Any | None:
        return self.memory.conn.execute(
            f"select * from {_TABLE} where job_key = ?", (key,)
        ).fetchone()

    @staticmethod
    def _as_result(row: Any) -> BatchPollResult:
        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        except (TypeError, json.JSONDecodeError):
            metadata = None
        return BatchPollResult(
            key=str(row["job_key"]),
            task=str(row["task"]),
            status=str(row["status"]),
            batch_id=str(row["batch_id"]) if row["batch_id"] else None,
            content=str(row["content"]) if row["content"] is not None else None,
            error=str(row["error"]) if row["error"] else None,
            metadata=metadata if isinstance(metadata, dict) else None,
            updated_at=float(row["updated_at"]),
        )

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
        }
        try:
            if self.http_client is not None:
                response = await self.http_client.request(
                    method, url, headers=headers, json=json_body,
                    timeout=self.timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.request(method, url, headers=headers, json=json_body)
        except httpx.HTTPError as exc:
            raise OpenRouterBatchError(f"OpenRouter batch transport error: {exc.__class__.__name__}.") from exc
        if not 200 <= response.status_code < 300:
            detail = ""
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    error = parsed.get("error")
                    if isinstance(error, dict):
                        detail = str(error.get("message") or "")
                    elif error:
                        detail = str(error)
            except (ValueError, json.JSONDecodeError):
                pass
            if not detail:
                detail = f"HTTP {response.status_code}"
            raise _BatchHTTPError(response.status_code, detail)
        try:
            parsed = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise OpenRouterBatchError("OpenRouter returned invalid JSON for a batch operation.") from exc
        if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
            parsed = parsed["data"]
        if not isinstance(parsed, dict):
            raise OpenRouterBatchError("OpenRouter returned an unexpected batch response.")
        return parsed

    async def submit(
        self,
        key: str,
        body: Mapping[str, Any],
        *,
        model: str | None = None,
        task: str = "background",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Submit once for ``key`` and return the durable OpenRouter batch id."""
        clean_key = str(key or "").strip()
        if not clean_key:
            raise ValueError("A stable batch job key is required.")
        self._require_key()
        if not isinstance(body, Mapping):
            raise TypeError("Batch chat request body must be a mapping.")
        selected_model = self.normalize_model(model or self.model)
        clean_task = str(task or "background").strip() or "background"
        clean_metadata = dict(metadata or {})
        payload_body = dict(body)
        payload_body.pop("stream", None)
        payload_body["model"] = selected_model
        try:
            serialized_body = self._json(payload_body)
            serialized_metadata = self._json(clean_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Batch request body and metadata must be JSON serializable.") from exc

        async with self._key_lock:
            prior = self._row(clean_key)
            if prior is not None:
                status = str(prior["status"])
                batch_id = str(prior["batch_id"]) if prior["batch_id"] else None
                if batch_id:
                    if status == "acknowledged":
                        raise OpenRouterBatchError(f"Batch job {clean_key!r} was already acknowledged.")
                    return batch_id
                if status in {"submitting", "submission_uncertain"}:
                    raise OpenRouterBatchError(
                        f"Batch submission for {clean_key!r} is ambiguous; refusing to submit a duplicate."
                    )
                raise OpenRouterBatchError(
                    f"Batch job {clean_key!r} is already {status}: {prior['error'] or 'no batch id'}"
                )

            now = time.time()
            self.memory.conn.execute(
                f"""
                insert into {_TABLE}(
                  job_key, task, model, batch_id, status, request_json, content,
                  error, metadata_json, response_json, created_at, updated_at
                ) values (?, ?, ?, null, 'submitting', ?, null, null, ?, null, ?, ?)
                """,
                (clean_key, clean_task, selected_model, serialized_body, serialized_metadata, now, now),
            )
            self.memory.conn.commit()

        request = {
            "endpoint": "/v1/chat/completions",
            "model": selected_model,
            "requests": [{"custom_id": clean_key, "body": payload_body}],
        }
        try:
            response = await self._request_json("POST", f"{self.base_url}/batches", json_body=request)
        except _BatchHTTPError as exc:
            status = "submission_uncertain" if exc.status_code >= 500 else "failed"
            self._update(
                clean_key, status=status, error=exc.detail,
                response_json=None,
            )
            raise OpenRouterBatchError(
                f"OpenRouter batch submit failed: {exc.detail}",
                job_status=status,
                ambiguous=status == "submission_uncertain",
            ) from exc
        except OpenRouterBatchError as exc:
            self._update(clean_key, status="submission_uncertain", error=str(exc), response_json=None)
            raise OpenRouterBatchError(
                str(exc), job_status="submission_uncertain", ambiguous=True
            ) from exc

        batch_id = str(response.get("id") or "").strip()
        if not batch_id:
            self._update(
                clean_key,
                status="submission_uncertain",
                error="OpenRouter accepted no parseable batch id; refusing automatic resubmission.",
                response_json=self._json(response),
            )
            raise OpenRouterBatchError(
                "OpenRouter batch response did not include an id.",
                job_status="submission_uncertain",
                ambiguous=True,
            )
        remote_status = self._normalize_status(str(response.get("status") or "queued"))
        self._update(
            clean_key,
            status=remote_status,
            batch_id=batch_id,
            request_json=None,
            error=None,
            response_json=self._json(response),
        )
        return batch_id

    async def poll(self, key: str) -> BatchPollResult:
        """Poll one persisted job; network errors leave its status retryable."""
        clean_key = str(key or "").strip()
        row = self._row(clean_key)
        if row is None:
            raise KeyError(f"Unknown OpenRouter batch job: {clean_key}")
        current = self._as_result(row)
        if current.status in _TERMINAL_STATUSES or current.status == "submission_uncertain":
            return current
        if not current.batch_id:
            if current.status == "submitting":
                self._update(
                    clean_key,
                    status="submission_uncertain",
                    error="Submission was interrupted before its batch id was saved; refusing duplicate submit.",
                )
                return BatchPollResult(
                    **{**current.__dict__, "status": "submission_uncertain",
                       "error": "Submission is still recorded as in progress; refusing duplicate submit."}
                )
            return current

        url = f"{self.base_url}/batches/{quote(current.batch_id, safe='')}"
        response = await self._request_json("GET", url)
        remote_status = self._normalize_status(str(response.get("status") or "queued"))
        content: str | None = None
        error: str | None = None
        if remote_status == "completed":
            content, error = self._extract_result(response, current.key)
            if error:
                remote_status = "failed"
        elif remote_status in {"failed", "expired", "cancelled"}:
            error = self._remote_error(response) or f"OpenRouter batch ended with status {remote_status}."

        self._update(
            clean_key,
            status=remote_status,
            content=content,
            error=error,
            response_json=self._json(response),
        )
        refreshed = self._row(clean_key)
        return self._as_result(refreshed)

    async def request(
        self,
        key: str,
        *,
        body: Mapping[str, Any],
        model: str | None = None,
        task: str = "background",
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Submit on first call, then poll on later calls; ``None`` means pending."""
        clean_key = str(key or "").strip()
        if self._row(clean_key) is None:
            await self.submit(clean_key, body, model=model, task=task, metadata=metadata)
            return None
        result = await self.poll(clean_key)
        if result.status == "completed":
            return result.content
        if result.status in {"failed", "expired", "cancelled", "submission_uncertain"}:
            raise OpenRouterBatchError(
                f"Batch job {clean_key!r} ended with status {result.status}: {result.error or 'unknown error'}",
                job_status=result.status,
                ambiguous=result.status == "submission_uncertain",
            )
        return None

    async def probe(self, key: str = "batch_probe:glm_flash") -> BatchPollResult:
        """Poll the current probe; start a fresh one after reporting its result."""
        probe_name = str(key or "").strip()
        active_key_name = f"openrouter_batch_probe_active:{probe_name}"
        reported_key_name = f"openrouter_batch_probe_reported:{probe_name}"
        clean_key = str(self.memory.app_kv_get(active_key_name) or "")
        reported_key = str(self.memory.app_kv_get(reported_key_name) or "")
        if not clean_key or clean_key == reported_key:
            clean_key = f"{probe_name}:{time.time_ns()}"
            self.memory.app_kv_set(active_key_name, clean_key)
        if self._row(clean_key) is None:
            await self.submit(
                clean_key,
                {
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: batch ok"}
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                },
                model=self.model,
                task="probe",
                metadata={"kind": "connectivity_probe"},
            )
            result = self._as_result(self._row(clean_key))
        else:
            result = await self.poll(clean_key)
        if result.status in (*_TERMINAL_STATUSES, "submission_uncertain"):
            self.memory.app_kv_set(reported_key_name, clean_key)
        return result

    def acknowledge(self, key: str) -> None:
        """Mark a terminal job handled after its result or fallback is durable."""
        row = self._row(str(key or "").strip())
        if row is None:
            raise KeyError(f"Unknown OpenRouter batch job: {key}")
        if str(row["status"]) == "acknowledged":
            return
        if str(row["status"]) not in {"completed", "failed", "expired", "cancelled", "submission_uncertain", "submitting"}:
            raise OpenRouterBatchError(f"Cannot acknowledge batch job in status {row['status']}.")
        self._update(str(key).strip(), status="acknowledged", content=None)

    def pending_jobs(self, task: str | None = None) -> list[BatchPollResult]:
        statuses = ",".join("?" for _ in _ACTIVE_STATUSES)
        args: list[Any] = list(_ACTIVE_STATUSES)
        query = f"select * from {_TABLE} where status in ({statuses})"
        if task is not None:
            query += " and task = ?"
            args.append(task)
        query += " order by created_at"
        return [self._as_result(row) for row in self.memory.conn.execute(query, args).fetchall()]

    def status_snapshot(self, task: str | None = None) -> dict[str, Any]:
        query = f"select status, count(*) as count from {_TABLE}"
        args: tuple[Any, ...] = ()
        if task is not None:
            query += " where task = ?"
            args = (task,)
        query += " group by status order by status"
        counts = {str(row["status"]): int(row["count"]) for row in self.memory.conn.execute(query, args)}
        jobs_query = f"select job_key, task, status, batch_id, error, updated_at from {_TABLE}"
        jobs_args: tuple[Any, ...] = ()
        if task is not None:
            jobs_query += " where task = ?"
            jobs_args = (task,)
        jobs_query += " order by updated_at desc limit 50"
        jobs = [
            {
                "key": str(row["job_key"]),
                "task": str(row["task"]),
                "status": str(row["status"]),
                "batch_id": str(row["batch_id"]) if row["batch_id"] else None,
                "error": str(row["error"]) if row["error"] else None,
                "updated_at": float(row["updated_at"]),
            }
            for row in self.memory.conn.execute(jobs_query, jobs_args).fetchall()
        ]
        return {"model": self.model, "task": task, "counts": counts, "jobs": jobs}

    def _update(self, key: str, **changes: Any) -> None:
        allowed = {
            "status", "batch_id", "request_json", "content", "error", "response_json",
        }
        fields = [(name, value) for name, value in changes.items() if name in allowed]
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name, _ in fields)
        values = [value for _, value in fields]
        values.extend([time.time(), key])
        self.memory.conn.execute(
            f"update {_TABLE} set {assignments}, updated_at = ? where job_key = ?",
            values,
        )
        self.memory.conn.commit()

    @staticmethod
    def _normalize_status(status: str) -> str:
        cleaned = status.strip().lower().replace("-", "_")
        if cleaned in {"completed", "failed", "expired", "cancelled", "acknowledged"}:
            return cleaned
        if cleaned in {"in_progress", "processing", "running", "executing"}:
            return "in_progress"
        if cleaned in {"validating", "queued", "pending", "created", "submitted"}:
            return "queued"
        return "queued"

    @classmethod
    def _extract_result(cls, response: Mapping[str, Any], key: str) -> tuple[str | None, str | None]:
        results = response.get("results")
        if isinstance(results, dict):
            result_rows = list(results.values())
            for custom_id, row in results.items():
                if str(custom_id) == key and isinstance(row, dict):
                    return cls._extract_content(row)
        elif isinstance(results, list):
            result_rows = results
            selected = next(
                (row for row in result_rows if isinstance(row, dict) and str(row.get("custom_id") or "") == key),
                None,
            )
            if selected is None and len(result_rows) == 1 and isinstance(result_rows[0], dict):
                selected = result_rows[0]
            if selected is not None:
                return cls._extract_content(selected)
        else:
            result_rows = []
        if len(result_rows) == 1 and isinstance(result_rows[0], dict):
            return cls._extract_content(result_rows[0])
        return None, "Completed OpenRouter batch did not include this request's result."

    @classmethod
    def _extract_content(cls, result: Mapping[str, Any]) -> tuple[str | None, str | None]:
        result_error = result.get("error")
        response = result.get("response")
        if isinstance(response, dict):
            status_code = response.get("status_code") or response.get("status")
            if isinstance(status_code, int) and status_code >= 400:
                return None, cls._remote_error(response) or f"Batch request returned HTTP {status_code}."
            body = response.get("body")
            if isinstance(body, dict):
                body_error = body.get("error")
                if body_error:
                    return None, cls._error_text(body_error)
                choices = body.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict) and message.get("content") is not None:
                        return cls._content_text(message.get("content")), None
            if response.get("choices"):
                choices = response.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict) and message.get("content") is not None:
                        return cls._content_text(message.get("content")), None
        if result_error:
            return None, cls._error_text(result_error)
        return None, "Completed OpenRouter batch response did not contain chat completion content."

    @staticmethod
    def _content_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for part in value:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
        return str(value)

    @staticmethod
    def _error_text(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("message") or value.get("code") or "Batch request failed.")
        return str(value)

    @classmethod
    def _remote_error(cls, response: Mapping[str, Any]) -> str | None:
        error = response.get("error")
        if error:
            return cls._error_text(error)
        body = response.get("body")
        if isinstance(body, dict) and body.get("error"):
            return cls._error_text(body["error"])
        return None


class _BatchHTTPError(OpenRouterBatchError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
