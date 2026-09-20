#!/usr/bin/env python3
"""Export, predict, and score anonymized real-group discourse replay cases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qq_social_agent.discourse_state import resolve_group_discourse
from qq_social_agent.jev_client import JevClient
from qq_social_agent.memory import ChatMessage
from qq_social_agent.reference_resolver import ReplyHint


BUCKET_PATTERNS = {
    "repair": re.compile(r"(?:不是.{0,16}是|我说的是|说错了?|改口|准确说|纠正)"),
    "ellipsis": re.compile(r"(?:这个|那个|这些|那些|呢[？?]?|然后呢|也行|第[一二三四五六七八九十\d]+个|去哪|为啥)"),
    "referent": re.compile(r"(?:^|[^\u4e00-\u9fff])(?:他|她|他们|她们|谁)(?:$|[^\u4e00-\u9fff])|[他她]现在|[他她]说|[他她]的"),
}
REPLY_RE = re.compile(r"回复.{0,40}消息")
URL_RE = re.compile(r"https?://\S+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
AT_RE = re.compile(r"\[@(\d+)\]")
REPLY_SUFFIX_RE = re.compile(r"回复.{0,40}?\[#(\d{3,})\]消息")
LABEL_SUFFIX_RE = re.compile(r"\[#\d{3,}\]")


def _read_env_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _segments(raw: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _at_ids(row: sqlite3.Row) -> tuple[int, ...]:
    result = []
    for segment in _segments(str(row["message_segments_json"] or "")):
        if segment.get("type") != "at":
            continue
        data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
        try:
            result.append(int(data.get("qq")))
        except (TypeError, ValueError):
            pass
    result.extend(int(match) for match in AT_RE.findall(str(row["text"] or "")))
    return tuple(dict.fromkeys(result))


def _bucket(row: sqlite3.Row) -> str:
    text = str(row["text"] or "")
    if BUCKET_PATTERNS["repair"].search(text):
        return "repair"
    if _at_ids(row) or REPLY_RE.search(text):
        return "addressee"
    if BUCKET_PATTERNS["ellipsis"].search(text):
        return "ellipsis"
    if BUCKET_PATTERNS["referent"].search(text):
        return "referent"
    return "baseline"


def _reply_target_id(text: str, suffix_index: dict[str, int]) -> int | None:
    match = REPLY_SUFFIX_RE.search(text or "")
    return suffix_index.get(match.group(1)) if match else None


def _alias(index: int) -> str:
    return f"user_{index:02d}"


def _anonymize_text(
    text: str,
    identity: dict[int, tuple[str, str]],
    *,
    sensitive_names: tuple[str, ...] = (),
) -> str:
    clean = URL_RE.sub("<url>", text or "")
    clean = EMAIL_RE.sub("<email>", clean)
    for user_id, (nickname, alias) in sorted(identity.items(), key=lambda item: len(item[1][0]), reverse=True):
        clean = clean.replace(f"[@{user_id}]", f"[@{alias}]")
        clean = clean.replace(str(user_id), alias)
        if nickname:
            clean = clean.replace(f"{nickname}[#", f"{alias}[#")
        if nickname and len(nickname.strip()) >= 2:
            clean = clean.replace(nickname, alias)
    local_names = {nickname for nickname, _alias_name in identity.values() if nickname}
    for nickname in sensitive_names:
        if nickname in local_names:
            continue
        clean = clean.replace(f"{nickname}[#", "<name>[#")
        if len(nickname) >= 2:
            clean = clean.replace(nickname, "<name>")
    clean = LABEL_SUFFIX_RE.sub("[#anon]", clean)
    return LONG_NUMBER_RE.sub("<number>", clean)[:500]


def export_cases(db_path: Path, output: Path, *, group_id: int, count: int, seed: int) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """select id, user_id, nickname, text, is_bot, created_at,
                  source_message_id, message_segments_json
             from messages where group_id=? order by id""",
        (group_id,),
    ).fetchall()
    sensitive_names = tuple(sorted(
        {str(row["nickname"] or "").strip() for row in rows if str(row["nickname"] or "").strip()},
        key=len,
        reverse=True,
    ))
    global_nicknames = {
        int(row["user_id"]): str(row["nickname"] or "").strip()
        for row in rows
        if str(row["nickname"] or "").strip()
    }
    suffixes: dict[str, list[int]] = defaultdict(list)
    for user_id in global_nicknames:
        suffixes[str(user_id)[-5:]].append(user_id)
    suffix_index = {suffix: values[0] for suffix, values in suffixes.items() if len(values) == 1}
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row["is_bot"] or 0) or not str(row["text"] or "").strip():
            continue
        candidates[_bucket(row)].append(index)
    rng = random.Random(seed)
    buckets = ("addressee", "referent", "ellipsis", "repair", "baseline")
    quota = count // len(buckets)
    chosen: list[tuple[str, int]] = []
    for bucket in buckets:
        pool = candidates[bucket][:]
        rng.shuffle(pool)
        chosen.extend((bucket, index) for index in pool[:quota])
    missing = count - len(chosen)
    if missing > 0:
        used = {index for _, index in chosen}
        pool = [index for index, row in enumerate(rows) if index not in used and not int(row["is_bot"] or 0)]
        rng.shuffle(pool)
        chosen.extend((_bucket(rows[index]), index) for index in pool[:missing])
    chosen.sort(key=lambda item: int(rows[item[1]]["id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    bucket_counts: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for bucket, index in chosen:
            current = rows[index]
            context_rows = rows[max(0, index - 10):index]
            reply_user_id = _reply_target_id(str(current["text"] or ""), suffix_index)
            user_ids = list(dict.fromkeys(
                [int(row["user_id"]) for row in (*context_rows, current)]
                + list(_at_ids(current))
                + ([reply_user_id] if reply_user_id is not None else [])
            ))
            identity = {
                user_id: (
                    next(
                        (str(row["nickname"] or "") for row in reversed((*context_rows, current)) if int(row["user_id"]) == user_id),
                        global_nicknames.get(user_id, ""),
                    ),
                    _alias(offset + 1),
                )
                for offset, user_id in enumerate(user_ids)
            }
            surrogate = {user_id: 100_000 + offset for offset, user_id in enumerate(user_ids, 1)}
            current_uid = int(current["user_id"])
            reply_source = next(
                (row for row in reversed(context_rows) if reply_user_id is not None and int(row["user_id"]) == reply_user_id),
                None,
            )
            sample_id = hashlib.sha256(f"{group_id}:{current['id']}:{seed}".encode()).hexdigest()[:16]
            payload = {
                "sample_id": sample_id,
                "bucket": bucket,
                "current": {
                    "user_id": surrogate[current_uid],
                    "nickname": identity[current_uid][1],
                    "text": _anonymize_text(
                        str(current["text"] or ""), identity, sensitive_names=sensitive_names
                    ),
                    "at_user_ids": [surrogate[user_id] for user_id in _at_ids(current) if user_id in surrogate],
                    "reply_user_id": surrogate.get(reply_user_id) if reply_user_id is not None else None,
                    "reply_text": _anonymize_text(
                        str(reply_source["text"] or "") if reply_source is not None else "",
                        identity,
                        sensitive_names=sensitive_names,
                    ),
                },
                "context": [
                    {
                        "index": offset,
                        "user_id": surrogate[int(row["user_id"])],
                        "nickname": identity[int(row["user_id"])][1],
                        "text": _anonymize_text(
                            str(row["text"] or ""), identity, sensitive_names=sensitive_names
                        ),
                        "is_bot": bool(row["is_bot"]),
                        "created_at": float(row["created_at"] or 0),
                    }
                    for offset, row in enumerate(context_rows)
                ],
                "gold": {
                    "annotated": False,
                    "addressee_user_id": None,
                    "referent_user_ids": None,
                    "ellipsis_kind": None,
                    "ellipsis_source_index": None,
                    "repair_kind": None,
                    "final_action": None,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            bucket_counts[bucket] += 1
    conn.close()
    return {"output": str(output), "count": len(chosen), "buckets": dict(bucket_counts)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def predict_cases(input_path: Path, output_path: Path, *, concurrency: int) -> dict[str, Any]:
    rows = _load_jsonl(input_path)
    client = JevClient(timeout=5.0)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    latencies: list[float] = []

    async def predict(row: dict[str, Any]) -> dict[str, Any]:
        current = row["current"]
        context = row.get("context") or []
        messages = [
            ChatMessage(
                group_id=1,
                user_id=int(item["user_id"]),
                nickname=str(item["nickname"]),
                text=str(item["text"]),
                is_bot=bool(item.get("is_bot")),
                created_at=float(item.get("created_at") or 0),
                source_message_id=str(item.get("index", "")),
            )
            for item in context
        ]
        known = {str(item["nickname"]): int(item["user_id"]) for item in context}
        known[str(current["nickname"])] = int(current["user_id"])
        reply_user_id = current.get("reply_user_id")
        reply_label = next(
            (name for name, user_id in known.items() if user_id == reply_user_id),
            str(reply_user_id or ""),
        )

        def named_resolver(text: str) -> tuple[int, ...]:
            return tuple(user_id for name, user_id in known.items() if name in text)

        async with semaphore:
            started = time.perf_counter()
            try:
                state = await resolve_group_discourse(
                    current_text=str(current["text"]),
                    current_user_id=int(current["user_id"]),
                    current_nickname=str(current["nickname"]),
                    self_id=999_999,
                    recent_messages=messages,
                    reply=ReplyHint(
                        exists=reply_user_id is not None,
                        author_id=int(reply_user_id) if reply_user_id is not None else None,
                        author_label=reply_label,
                        text=str(current.get("reply_text") or ""),
                        message_id="replay" if reply_user_id is not None else "",
                    ),
                    at_user_ids=tuple(int(value) for value in current.get("at_user_ids") or []),
                    named_resolver=named_resolver,
                    jev=client,
                )
                prediction = {
                    "addressee_user_id": state.addressee.target_id,
                    "addressee_status": state.addressee.status,
                    "addressee_confidence": state.addressee.confidence,
                    "referent_user_ids": list(state.reference.user_ids),
                    "referent_status": state.reference.status,
                    "referent_confidence": state.reference.confidence,
                    "ellipsis_kind": state.ellipsis.kind,
                    "ellipsis_source_text": state.ellipsis.source_text,
                    "ellipsis_status": state.ellipsis.status,
                    "ellipsis_confidence": state.ellipsis.confidence,
                    "repair_kind": state.repair.kind,
                    "repair_status": state.repair.status,
                    "state_audit": state.state_audit,
                }
            except Exception as exc:
                prediction = {"error": type(exc).__name__}
            latency = round((time.perf_counter() - started) * 1000, 2)
        latencies.append(latency)
        return {**row, "prediction": prediction, "latency_ms": latency}

    predicted = await asyncio.gather(*(predict(row) for row in rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predicted),
        encoding="utf-8",
    )
    await client.aclose()
    ordered = sorted(latencies)
    return {
        "output": str(output_path),
        "count": len(predicted),
        "errors": sum("error" in row.get("prediction", {}) for row in predicted),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0,
            "p50": ordered[len(ordered) // 2] if ordered else 0,
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0,
        },
        "status_counts": dict(Counter(
            row.get("prediction", {}).get("ellipsis_status", "error") for row in predicted
        )),
    }


def score_cases(path: Path) -> dict[str, Any]:
    rows = [row for row in _load_jsonl(path) if row.get("gold", {}).get("annotated")]
    fields = ("addressee_user_id", "referent_user_ids", "ellipsis_kind", "repair_kind", "final_action")
    result: dict[str, Any] = {"annotated": len(rows), "fields": {}}
    for field in fields:
        comparable = [row for row in rows if row["gold"].get(field) is not None and field in row.get("prediction", {})]
        correct = sum(row["gold"].get(field) == row["prediction"].get(field) for row in comparable)
        result["fields"][field] = {
            "count": len(comparable),
            "correct": correct,
            "accuracy": round(correct / len(comparable), 4) if comparable else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--db", type=Path, default=Path("data/bot.sqlite3"))
    export.add_argument("--output", type=Path, default=Path("data/evals/discourse_replay_300.jsonl"))
    export.add_argument("--group-id", type=int, required=True)
    export.add_argument("--count", type=int, default=300)
    export.add_argument("--seed", type=int, default=20260920)
    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--env", type=Path, default=Path(".env"))
    predict.add_argument("--concurrency", type=int, default=8)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_cases(args.db, args.output, group_id=args.group_id, count=args.count, seed=args.seed)
    elif args.command == "predict":
        key = _read_env_key(args.env)
        if key:
            os.environ["TYPESAFE_API_KEY"] = key
        result = asyncio.run(predict_cases(args.input, args.output, concurrency=args.concurrency))
    else:
        result = score_cases(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
