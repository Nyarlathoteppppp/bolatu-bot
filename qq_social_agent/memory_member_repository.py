"""Group directories, member profiles and impressions on the shared connection."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter

from .memory_models import ChatMessage, GroupInfo, GroupMember, MemberProfile, MemberImpression, MemberProfileSummary
from .memory_corpus import _raw_corpus_tags, _is_low_value_raw_corpus_text
from .memory_text import _clean_text_list, _json_text_list, _compact_text
from .memory_repository_utils import _message_from_row, _relevance_terms


class MemberRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_group_info(
        self,
        *,
        group_id: int,
        group_name: str,
        member_count: int,
        max_member_count: int,
        last_synced_at: float | None = None,
    ) -> None:
        synced = last_synced_at or time.time()
        self.conn.execute(
            """
            insert into group_info(group_id, group_name, member_count, max_member_count, last_synced_at)
            values (?, ?, ?, ?, ?)
            on conflict(group_id) do update set
              group_name = excluded.group_name,
              member_count = excluded.member_count,
              max_member_count = excluded.max_member_count,
              last_synced_at = excluded.last_synced_at
            """,
            (group_id, group_name.strip()[:120], max(0, member_count), max(0, max_member_count), synced),
        )
        self.conn.commit()

    def group_info(self, group_id: int) -> GroupInfo | None:
        row = self.conn.execute(
            """
            select group_id, group_name, member_count, max_member_count, last_synced_at
            from group_info
            where group_id = ?
            """,
            (group_id,),
        ).fetchone()
        return _group_info_from_row(row) if row else None

    def replace_group_members(
        self,
        group_id: int,
        members: list[dict[str, object]],
        *,
        synced_at: float | None = None,
    ) -> int:
        synced = synced_at or time.time()
        self.conn.execute(
            "update group_members set active = 0, last_synced_at = ? where group_id = ?",
            (synced, group_id),
        )
        clean_members: list[tuple[object, ...]] = []
        for member in members:
            user_id = int(member.get("user_id") or 0)
            if user_id <= 0:
                continue
            clean_members.append(
                (
                    group_id,
                    user_id,
                    str(member.get("nickname") or user_id).strip()[:120],
                    str(member.get("card") or "").strip()[:120],
                    str(member.get("role") or "").strip()[:32],
                    str(member.get("title") or "").strip()[:120],
                    float(member.get("joined_at") or 0.0),
                    float(member.get("last_sent_at") or 0.0),
                    synced,
                    1,
                )
            )
        self.conn.executemany(
            """
            insert into group_members(
              group_id, user_id, nickname, card, role, title,
              joined_at, last_sent_at, last_synced_at, active
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              nickname = excluded.nickname,
              card = excluded.card,
              role = excluded.role,
              title = excluded.title,
              joined_at = excluded.joined_at,
              last_sent_at = excluded.last_sent_at,
              last_synced_at = excluded.last_synced_at,
              active = excluded.active
            """,
            clean_members,
        )
        self.conn.commit()
        return len(clean_members)

    def group_member(self, group_id: int, user_id: int) -> GroupMember | None:
        row = self.conn.execute(
            """
            select group_id, user_id, nickname, card, role, title,
                   joined_at, last_sent_at, last_synced_at, active
            from group_members
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        return _group_member_from_row(row) if row else None

    def _upsert_member_profile(
        self,
        group_id: int,
        user_id: int,
        display_name: str,
        *,
        last_seen_at: float,
    ) -> None:
        clean_name = display_name.strip() or str(user_id)
        row = self.conn.execute(
            """
            select aliases_json from member_profiles
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        aliases: list[str] = []
        if row:
            try:
                raw_aliases = json.loads(str(row["aliases_json"]))
            except json.JSONDecodeError:
                raw_aliases = []
            if isinstance(raw_aliases, list):
                aliases = [str(alias).strip() for alias in raw_aliases if str(alias).strip()]
        aliases = _dedupe_names([clean_name, *aliases])[:8]
        self.conn.execute(
            """
            insert into member_profiles(group_id, user_id, display_name, aliases_json, last_seen_at)
            values (?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              display_name = excluded.display_name,
              aliases_json = excluded.aliases_json,
              last_seen_at = excluded.last_seen_at
            """,
            (group_id, user_id, clean_name, json.dumps(aliases, ensure_ascii=False), last_seen_at),
        )

    def member_profiles_for_context(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        limit: int,
    ) -> list[MemberProfile]:
        ordered_user_ids = _dedupe_ints(user_ids)[:limit]
        if not ordered_user_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_user_ids)
        rows = self.conn.execute(
            f"""
            select
              p.group_id,
              p.user_id,
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at
            from member_profiles p
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
            where p.group_id = ? and p.user_id in ({placeholders})
            """,
            (group_id, *ordered_user_ids),
        ).fetchall()
        by_user_id = {int(row["user_id"]): _profile_from_row(row) for row in rows}
        return [by_user_id[user_id] for user_id in ordered_user_ids if user_id in by_user_id]

    def member_impressions_for_context(
        self,
        group_id: int,
        user_ids: list[int],
        *,
        limit: int,
    ) -> list[MemberImpression]:
        ordered_user_ids = _dedupe_ints(user_ids)[:limit]
        if not ordered_user_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_user_ids)
        rows = self.conn.execute(
            f"""
            select
              p.group_id,
              p.user_id,
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at,
              coalesce(i.message_count, 0) as message_count,
              coalesce(i.tag_counts_json, '{{}}') as tag_counts_json,
              coalesce(i.keyword_counts_json, '{{}}') as keyword_counts_json,
              coalesce(i.recent_texts_json, '[]') as recent_texts_json,
              coalesce(i.updated_at, p.last_seen_at) as updated_at,
              coalesce(s.profile_summary, '') as ai_summary,
              coalesce(s.interests_json, '[]') as ai_interests_json,
              coalesce(s.speaking_style, '') as ai_speaking_style,
              coalesce(s.representative_texts_json, '[]') as ai_representative_texts_json,
              coalesce(s.created_at, 0) as ai_summary_at
            from member_profiles p
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
            left join member_impressions i
              on i.group_id = p.group_id and i.user_id = p.user_id
            left join member_profile_summaries s
              on s.id = (
                select latest.id
                from member_profile_summaries latest
                where latest.group_id = p.group_id and latest.user_id = p.user_id
                order by latest.created_at desc, latest.id desc
                limit 1
              )
            where p.group_id = ? and p.user_id in ({placeholders})
            """,
            (group_id, *ordered_user_ids),
        ).fetchall()
        by_user_id = {int(row["user_id"]): _member_impression_from_row(row) for row in rows}
        return [by_user_id[user_id] for user_id in ordered_user_ids if user_id in by_user_id]

    def recent_member_impressions(self, group_id: int, limit: int) -> list[MemberImpression]:
        rows = self.conn.execute(
            """
            select
              p.group_id,
              p.user_id,
              coalesce(nullif(g.card, ''), nullif(g.nickname, ''), p.display_name) as display_name,
              p.aliases_json,
              coalesce(nullif(g.last_synced_at, 0), p.last_seen_at) as last_seen_at,
              coalesce(i.message_count, 0) as message_count,
              coalesce(i.tag_counts_json, '{}') as tag_counts_json,
              coalesce(i.keyword_counts_json, '{}') as keyword_counts_json,
              coalesce(i.recent_texts_json, '[]') as recent_texts_json,
              coalesce(i.updated_at, p.last_seen_at) as updated_at,
              coalesce(s.profile_summary, '') as ai_summary,
              coalesce(s.interests_json, '[]') as ai_interests_json,
              coalesce(s.speaking_style, '') as ai_speaking_style,
              coalesce(s.representative_texts_json, '[]') as ai_representative_texts_json,
              coalesce(s.created_at, 0) as ai_summary_at
            from member_profiles p
            left join group_members g
              on g.group_id = p.group_id and g.user_id = p.user_id and g.active = 1
            left join member_impressions i
              on i.group_id = p.group_id and i.user_id = p.user_id
            left join member_profile_summaries s
              on s.id = (
                select latest.id
                from member_profile_summaries latest
                where latest.group_id = p.group_id and latest.user_id = p.user_id
                order by latest.created_at desc, latest.id desc
                limit 1
              )
            where p.group_id = ?
            order by p.last_seen_at desc
            limit ?
            """,
            (group_id, limit),
        ).fetchall()
        return [_member_impression_from_row(row) for row in rows]

    def active_member_ids_since(
        self,
        group_id: int,
        *,
        since_at: float,
        limit: int,
        min_messages: int,
    ) -> list[int]:
        rows = self.conn.execute(
            """
            select user_id, count(*) as message_count, max(created_at) as last_seen_at
            from messages
            where group_id = ? and is_bot = 0 and created_at >= ? and length(trim(text)) >= 2
            group by user_id
            having count(*) >= ?
            order by message_count desc, last_seen_at desc
            limit ?
            """,
            (group_id, since_at, min_messages, limit),
        ).fetchall()
        return [int(row["user_id"]) for row in rows]

    def member_messages_between(
        self,
        group_id: int,
        user_id: int,
        *,
        start_at: float,
        end_at: float,
        limit: int,
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """
            select id, group_id, user_id, nickname, text, is_bot, created_at, source_message_id, session_id, message_segments_json, raw_message_json, sender_json
            from messages
            where group_id = ?
              and user_id = ?
              and is_bot = 0
              and created_at >= ?
              and created_at < ?
              and length(trim(text)) >= 2
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, user_id, start_at, end_at, limit),
        ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def last_member_profile_summary_at(self, group_id: int, user_id: int) -> float:
        row = self.conn.execute(
            """
            select max(created_at) as ts
            from member_profile_summaries
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        return float(row["ts"] or 0.0) if row else 0.0

    def latest_member_profile_summary(
        self,
        group_id: int,
        user_id: int,
    ) -> MemberProfileSummary | None:
        summaries = self.recent_member_profile_summaries(group_id, user_id, limit=1)
        return summaries[0] if summaries else None

    def add_member_profile_summary(
        self,
        *,
        group_id: int,
        user_id: int,
        profile_summary: str,
        interests: list[str],
        speaking_style: str,
        representative_texts: list[str],
        start_at: float,
        end_at: float,
        message_count: int,
        keep_per_member: int = 3,
    ) -> None:
        summary = profile_summary.strip()
        if not summary:
            return
        now = time.time()
        self.conn.execute(
            """
            insert into member_profile_summaries(
              group_id, user_id, profile_summary, interests_json, speaking_style,
              representative_texts_json, start_at, end_at, message_count, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                group_id,
                user_id,
                summary[:420],
                json.dumps(_clean_text_list(interests, limit=8, item_limit=32), ensure_ascii=False),
                speaking_style.strip()[:260],
                json.dumps(_clean_text_list(representative_texts, limit=5, item_limit=140), ensure_ascii=False),
                start_at,
                end_at,
                max(0, int(message_count)),
                now,
            ),
        )
        self.conn.execute(
            """
            delete from member_profile_summaries
            where group_id = ? and user_id = ?
              and id not in (
                select id from member_profile_summaries
                where group_id = ? and user_id = ?
                order by created_at desc, id desc
                limit ?
              )
            """,
            (group_id, user_id, group_id, user_id, keep_per_member),
        )
        self.conn.commit()

    def prune_member_profile_summaries(self, *, keep_per_member: int = 3) -> int:
        keep = max(1, int(keep_per_member))
        pairs = self.conn.execute(
            "select distinct group_id, user_id from member_profile_summaries"
        ).fetchall()
        deleted = 0
        for row in pairs:
            cursor = self.conn.execute(
                """
                delete from member_profile_summaries
                where group_id = ? and user_id = ?
                  and id not in (
                    select id from member_profile_summaries
                    where group_id = ? and user_id = ?
                    order by created_at desc, id desc
                    limit ?
                  )
                """,
                (row["group_id"], row["user_id"], row["group_id"], row["user_id"], keep),
            )
            deleted += int(cursor.rowcount or 0)
        self.conn.commit()
        return deleted

    def recent_member_profile_summaries(
        self,
        group_id: int,
        user_id: int,
        limit: int,
    ) -> list[MemberProfileSummary]:
        rows = self.conn.execute(
            """
            select group_id, user_id, profile_summary, interests_json, speaking_style,
                   representative_texts_json, start_at, end_at, message_count, created_at
            from member_profile_summaries
            where group_id = ? and user_id = ?
            order by created_at desc, id desc
            limit ?
            """,
            (group_id, user_id, limit),
        ).fetchall()
        return [_member_profile_summary_from_row(row) for row in rows]

    def _update_member_impression(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        text: str,
        *,
        created_at: float,
    ) -> None:
        clean_text = text.strip()
        row = self.conn.execute(
            """
            select message_count, tag_counts_json, keyword_counts_json, recent_texts_json
            from member_impressions
            where group_id = ? and user_id = ?
            """,
            (group_id, user_id),
        ).fetchone()
        if row:
            message_count = int(row["message_count"]) + 1
            tag_counts = _counter_from_json(row["tag_counts_json"])
            keyword_counts = _counter_from_json(row["keyword_counts_json"])
            recent_texts = _recent_texts_from_json(row["recent_texts_json"])
        else:
            message_count = 1
            tag_counts = Counter()
            keyword_counts = Counter()
            recent_texts = []

        tags = _raw_corpus_tags(clean_text)
        tag_counts.update(tags)
        keyword_counts.update(_impression_keywords(clean_text))
        tag_counts = _cap_counter(tag_counts, 60)
        keyword_counts = _cap_counter(keyword_counts, 80)
        if clean_text and not _is_low_value_raw_corpus_text(clean_text):
            recent_texts.append(
                {
                    "text": clean_text[:140],
                    "tags": list(tags[:4]),
                    "at": created_at,
                }
            )
            recent_texts = recent_texts[-16:]

        self.conn.execute(
            """
            insert into member_impressions(
              group_id, user_id, message_count, tag_counts_json,
              keyword_counts_json, recent_texts_json, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(group_id, user_id) do update set
              message_count = excluded.message_count,
              tag_counts_json = excluded.tag_counts_json,
              keyword_counts_json = excluded.keyword_counts_json,
              recent_texts_json = excluded.recent_texts_json,
              updated_at = excluded.updated_at
            """,
            (
                group_id,
                user_id,
                message_count,
                json.dumps(dict(tag_counts), ensure_ascii=False, sort_keys=True),
                json.dumps(dict(keyword_counts), ensure_ascii=False, sort_keys=True),
                json.dumps(recent_texts, ensure_ascii=False),
                created_at,
            ),
        )



def _group_info_from_row(row: sqlite3.Row) -> GroupInfo:
    return GroupInfo(
        group_id=int(row["group_id"]),
        group_name=str(row["group_name"]),
        member_count=int(row["member_count"] or 0),
        max_member_count=int(row["max_member_count"] or 0),
        last_synced_at=float(row["last_synced_at"] or 0.0),
    )


def _group_member_from_row(row: sqlite3.Row) -> GroupMember:
    return GroupMember(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        nickname=str(row["nickname"]),
        card=str(row["card"] or ""),
        role=str(row["role"] or ""),
        title=str(row["title"] or ""),
        joined_at=float(row["joined_at"] or 0.0),
        last_sent_at=float(row["last_sent_at"] or 0.0),
        last_synced_at=float(row["last_synced_at"] or 0.0),
        active=bool(row["active"]),
    )


def _impression_keywords(text: str) -> list[str]:
    compact = _compact_text(text)
    if not compact or _is_low_value_raw_corpus_text(compact):
        return []
    stop_terms = {
        "真的",
        "现在",
        "今天",
        "这个",
        "那个",
        "还是",
        "感觉",
        "不是",
        "没有",
        "怎么",
        "什么",
        "因为",
        "所以",
        "但是",
        "然后",
        "自己",
        "他们",
        "我们",
        "你们",
        "一样",
        "直接",
    }
    terms = [
        term
        for term in _relevance_terms(text)
        if 2 <= len(term) <= 12 and term not in stop_terms and not term.isdigit()
    ]
    terms.sort(key=lambda term: (len(term), term), reverse=True)
    return terms[:12]


def _counter_from_json(value: object) -> Counter[str]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = {}
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        text = str(key).strip()
        if not text:
            continue
        try:
            counter[text] = max(0, int(count))
        except (TypeError, ValueError):
            continue
    return counter


def _recent_texts_from_json(value: object) -> list[dict[str, object]]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        raw = []
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        tags_raw = item.get("tags", [])
        tags = (
            [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            if isinstance(tags_raw, list)
            else []
        )
        try:
            created_at = float(item.get("at", 0.0) or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        result.append({"text": text[:140], "tags": tags[:4], "at": created_at})
    return result


def _cap_counter(counter: Counter[str], limit: int) -> Counter[str]:
    return Counter(dict(counter.most_common(limit)))


def _top_counter_items(counter: Counter[str], limit: int) -> tuple[tuple[str, int], ...]:
    return tuple((key, int(count)) for key, count in counter.most_common(limit) if count > 0)


def _profile_from_row(row: sqlite3.Row) -> MemberProfile:
    try:
        raw_aliases = json.loads(str(row["aliases_json"]))
    except json.JSONDecodeError:
        raw_aliases = []
    aliases = ()
    if isinstance(raw_aliases, list):
        aliases = tuple(_dedupe_names(str(alias).strip() for alias in raw_aliases if str(alias).strip()))
    return MemberProfile(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        display_name=str(row["display_name"]),
        aliases=aliases,
        last_seen_at=float(row["last_seen_at"]),
    )


def _member_impression_from_row(row: sqlite3.Row) -> MemberImpression:
    profile = _profile_from_row(row)
    recent_texts = _recent_texts_from_json(row["recent_texts_json"])
    return MemberImpression(
        group_id=profile.group_id,
        user_id=profile.user_id,
        display_name=profile.display_name,
        aliases=profile.aliases,
        message_count=int(row["message_count"] or 0),
        top_tags=_top_counter_items(_counter_from_json(row["tag_counts_json"]), limit=6),
        top_keywords=_top_counter_items(_counter_from_json(row["keyword_counts_json"]), limit=8),
        recent_texts=tuple(
            str(item["text"])
            for item in recent_texts[-5:]
            if str(item.get("text", "")).strip()
        ),
        ai_summary=str(row["ai_summary"] or "").strip(),
        ai_interests=tuple(_json_text_list(row["ai_interests_json"], limit=8)),
        ai_speaking_style=str(row["ai_speaking_style"] or "").strip(),
        ai_representative_texts=tuple(_json_text_list(row["ai_representative_texts_json"], limit=5)),
        ai_summary_at=float(row["ai_summary_at"] or 0.0),
        last_seen_at=profile.last_seen_at,
        updated_at=float(row["updated_at"] or profile.last_seen_at),
    )


def _member_profile_summary_from_row(row: sqlite3.Row) -> MemberProfileSummary:
    return MemberProfileSummary(
        group_id=int(row["group_id"]),
        user_id=int(row["user_id"]),
        profile_summary=str(row["profile_summary"]),
        interests=tuple(_json_text_list(row["interests_json"], limit=8)),
        speaking_style=str(row["speaking_style"]),
        representative_texts=tuple(_json_text_list(row["representative_texts_json"], limit=5)),
        start_at=float(row["start_at"]),
        end_at=float(row["end_at"]),
        message_count=int(row["message_count"]),
        created_at=float(row["created_at"]),
    )


def _dedupe_names(names: object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = str(name).strip()
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _dedupe_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
