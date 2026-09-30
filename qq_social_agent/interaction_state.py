"""Evidence-backed views of active group interactions, without another resolver."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .resolver_result import RESOLVED

if TYPE_CHECKING:
    from .discourse_state import DiscourseState

QUESTION = re.compile(r"[?？]|(?:怎么|如何|为什么|谁|什么|哪[个里种]|能不能|可不可以|有没有)|吗[。！!～~\s]*$")
TONE_FEEDBACK = re.compile(
    r"(?:^|[，,。！!])\s*(?:你)?(?:别|不要|少|停止).{0,8}(?:怼|骂|嘲讽|阴阳|说教|开玩笑|撒娇)"
    r"|(?:^|[，,。！!])\s*(?:你)?(?:说话)?(?:温柔(?:一点|点)|好好说话)"
    r"|(?:^|[，,。！!])\s*(?:你(?:说话|刚才(?:那句)?)|刚才那句)(?:也|有点|实在)?太(?:冲|凶)"
)
CORRECTION = re.compile(r"你(?:刚才|这句|那句|这里|又)?(?:说错|理解错|搞错|答错)|^我说的是")
CLOSE = re.compile(r"^(?:不用(?:了|回答了)|不问了|解决了|算了不问了|换个话题)[。！!，,\s]*$")


@dataclass(frozen=True)
class InteractionEvidence:
    message_id: int
    source_message_id: str
    user_id: int
    nickname: str
    text: str
    kind: str
    parent_message_id: int | None
    addressed_bot: bool
    created_at: float
    text_provenance: str
    edge_source: str
    action: str = ""
    context_at: float | None = None


@dataclass(frozen=True)
class InteractionState:
    group_id: int
    root_message_id: int
    events: tuple[InteractionEvidence, ...]

    def ancestors(self, event: InteractionEvidence) -> set[int]:
        by_id = {item.message_id: item for item in self.events}
        found: set[int] = set()
        parent_id = event.parent_message_id
        while parent_id in by_id and parent_id not in found:
            found.add(parent_id)
            parent_id = by_id[parent_id].parent_message_id
        return found

    @property
    def pending_questions(self) -> tuple[InteractionEvidence, ...]:
        # A send proves delivery to its trigger, not that the matter was solved.
        sent_to = {item.parent_message_id for item in self.events if item.kind == "sent"}
        closures = [item for item in self.events if item.kind == "closed"]
        return tuple(item for item in self.events if item.kind == "question" and item.addressed_bot
                     and item.message_id not in sent_to and not any(
                         close.user_id == item.user_id and item.message_id in self.ancestors(close)
                         for close in closures))

    @property
    def feedback(self) -> tuple[InteractionEvidence, ...]:
        return tuple(item for item in self.events if item.kind in {"tone_feedback", "correction"})


class InteractionStateStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript("""
            create table if not exists interaction_events (
              message_id integer primary key,
              group_id integer not null,
              parent_message_id integer,
              reply_source_id text not null default '',
              addressed_bot integer not null default 0,
              kind text not null,
              edge_source text not null default '',
              action text not null default '',
              context_at real
            );
            create index if not exists idx_interaction_events_group
              on interaction_events(group_id, message_id);
        """)

    def _source_row(self, group_id: int, source_id: str) -> sqlite3.Row | None:
        if not source_id:
            return None
        return self.conn.execute(
            "select * from messages where group_id = ? and source_message_id = ?",
            (group_id, str(source_id)),
        ).fetchone()

    def _event(self, message_id: int) -> sqlite3.Row | None:
        return self.conn.execute("select * from interaction_events where message_id = ?", (message_id,)).fetchone()

    @staticmethod
    def _utterance(row: sqlite3.Row) -> tuple[str, str]:
        if row["is_bot"]:
            return str(row["text"]), "acknowledged_send"
        # Stored text may contain a quote envelope. Only original text segments
        # establish what this speaker actually said.
        segments = json.loads(row["message_segments_json"] or "[]")
        return ("".join(str(item.get("data", {}).get("text", ""))
                        for item in segments if item.get("type") == "text").strip(),
                "message_segments" if segments else "unavailable")

    @staticmethod
    def _kind(text: str, addressed_bot: bool) -> str:
        if addressed_bot:
            if CLOSE.fullmatch(text.strip()):
                return "closed"
            if TONE_FEEDBACK.search(text):
                return "tone_feedback"
            if CORRECTION.search(text):
                return "correction"
        return "question" if QUESTION.search(text) else "message"

    def observe_inbound(self, *, group_id: int, source_message_id: str,
                        reply_source_id: str = "", addressed_bot: bool = False) -> None:
        row = self._source_row(group_id, source_message_id)
        if row is None or row["is_bot"] or self._event(row["id"]) is not None:
            return
        parent = self._source_row(group_id, reply_source_id)
        # A QQ quote is an observed edge even if the parent predates this ledger.
        addressed_bot = addressed_bot or bool(parent is not None and parent["is_bot"])
        self.conn.execute("""
            insert into interaction_events
              (message_id, group_id, parent_message_id, reply_source_id, addressed_bot, kind, edge_source)
            values (?, ?, ?, ?, ?, ?, ?)
        """, (row["id"], group_id, parent["id"] if parent is not None else None,
              reply_source_id, int(addressed_bot), self._kind(self._utterance(row)[0], addressed_bot),
              "qq_reply" if reply_source_id else ""))
        self.conn.commit()

    def bind_discourse(self, *, group_id: int, source_message_id: str,
                       discourse: DiscourseState, self_id: int,
                       context_message_ids: Iterable[int]) -> InteractionState | None:
        """Use resolved continuation evidence; references to an item are not reply edges."""
        row = self._source_row(group_id, source_message_id)
        if row is None or row["is_bot"]:
            return None
        self.observe_inbound(group_id=group_id, source_message_id=source_message_id)
        event = self._event(row["id"])
        addressed = (discourse.addressee.target_id == self_id
                     if discourse.addressee.status == RESOLVED else bool(event["addressed_bot"]))
        parent = None
        ellipsis = discourse.ellipsis
        if (not event["reply_source_id"] and discourse.continuation.status == RESOLVED
                and ellipsis.status == RESOLVED and ellipsis.kind == "CONTINUATION"):
            if ellipsis.source_message_id:
                parent = self._source_row(group_id, ellipsis.source_message_id)
            elif ellipsis.source_key.startswith("m") and ellipsis.source_key[1:].isdigit():
                parent = self.conn.execute("select * from messages where group_id = ? and id = ?",
                                           (group_id, int(ellipsis.source_key[1:]))).fetchone()
        if parent is not None and parent["id"] < row["id"] and event["parent_message_id"] is None:
            self.conn.execute("update interaction_events set parent_message_id = ?, edge_source = 'discourse_continuation' where message_id = ?",
                              (parent["id"], row["id"]))
        self.conn.execute("update interaction_events set addressed_bot = ?, kind = ? where message_id = ?",
                          (int(addressed), self._kind(self._utterance(row)[0], addressed), row["id"]))
        self.conn.commit()
        return self.for_source(group_id, source_message_id, context_message_ids=context_message_ids)

    def observe_sent(self, *, group_id: int, source_message_id: str,
                     trigger_source_id: str, action: str, context_at: float | None) -> None:
        """Call only after an acknowledged send and message persistence."""
        row = self._source_row(group_id, source_message_id)
        trigger = self._source_row(group_id, trigger_source_id)
        if row is None or not row["is_bot"] or trigger is None or trigger["is_bot"]:
            return
        self.observe_inbound(group_id=group_id, source_message_id=trigger_source_id)
        self.conn.execute("""
            insert or ignore into interaction_events
              (message_id, group_id, parent_message_id, kind, edge_source, action, context_at)
            values (?, ?, ?, 'sent', 'sent_trigger', ?, ?)
        """, (row["id"], group_id, trigger["id"], action, context_at))
        self.conn.commit()

    def for_source(self, group_id: int, source_message_id: str, *,
                   context_message_ids: Iterable[int]) -> InteractionState | None:
        row = self._source_row(group_id, source_message_id)
        event = self._event(row["id"]) if row is not None else None
        if event is None:
            return None
        ids = set(context_message_ids) | {row["id"]}
        if event["parent_message_id"] is not None:
            ids.add(event["parent_message_id"])
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(f"""
            select m.*, e.kind, e.parent_message_id, e.addressed_bot, e.edge_source, e.action, e.context_at
            from messages m left join interaction_events e on m.id = e.message_id
            where m.group_id = ? and m.id in ({placeholders}) order by m.created_at, m.id
        """, (group_id, *ids)).fetchall()
        by_id = {item["id"]: item for item in rows}
        # Follow only admitted evidence edges. This reuses the caller's current
        # context window instead of reviving a whole persisted historic tree.
        connected = {row["id"]}
        root = row["id"]
        while by_id[root]["parent_message_id"] in by_id and by_id[root]["parent_message_id"] not in connected:
            root = by_id[root]["parent_message_id"]
            connected.add(root)
        # Include responses to this turn, while sibling conversations of an
        # ancestor stay on their own branch.
        descendants = {row["id"]}
        changed = True
        while changed:
            changed = False
            for item in rows:
                parent = item["parent_message_id"]
                if parent in descendants and item["id"] not in descendants:
                    descendants.add(item["id"])
                    changed = True
        connected.update(descendants)
        evidence = []
        for item in rows:
            if item["id"] not in connected:
                continue
            text, provenance = self._utterance(item)
            evidence.append(InteractionEvidence(
                item["id"], str(item["source_message_id"] or ""), item["user_id"], item["nickname"],
                text, item["kind"] or "message", item["parent_message_id"], bool(item["addressed_bot"]),
                item["created_at"], provenance, item["edge_source"] or "", item["action"] or "", item["context_at"]))
        return InteractionState(group_id, root, tuple(evidence))

    def own_contributions(self, state: InteractionState | None, *, source_message_id: str,
                          context_message_ids: Iterable[int]) -> list[dict[str, object]]:
        """Generation-only view of acknowledged replies on the current ancestor branch."""
        if state is None:
            return []
        by_id = {event.message_id: event for event in state.events}
        current = next((event for event in state.events if event.source_message_id == source_message_id), None)
        if current is None or current.kind == "closed":
            return []
        ancestors: set[int] = set()
        event = current
        while event.message_id not in ancestors:
            if event.kind == "closed":
                break
            ancestors.add(event.message_id)
            if event.parent_message_id not in by_id:
                break
            event = by_id[event.parent_message_id]
        ids = set(context_message_ids) | {
            event.message_id for event in state.events
            if event.kind == "sent" and event.message_id in ancestors
        }
        if not ids:
            return []
        id_slots = ",".join("?" for _ in ids)
        ancestor_slots = ",".join("?" for _ in ancestors)
        rows = self.conn.execute(f"""
            select m.*, e.action, e.parent_message_id,
                   trigger.source_message_id as trigger_source_id,
                   trigger.user_id as trigger_user_id, trigger.nickname as trigger_speaker,
                   trigger.message_segments_json as trigger_segments
            from messages m join interaction_events e on e.message_id = m.id
            join messages trigger on trigger.id = e.parent_message_id and trigger.group_id = m.group_id
            where m.group_id = ? and m.id in ({id_slots}) and m.is_bot = 1
              and e.kind = 'sent' and e.action != 'meme' and m.created_at <= ?
              and (m.id in ({ancestor_slots}) or e.parent_message_id in ({ancestor_slots}))
            order by m.created_at, m.id
        """, (state.group_id, *ids, current.created_at, *ancestors, *ancestors)).fetchall()
        observed = self.conn.execute(f"""
            select m.*, e.kind, e.parent_message_id from messages m
            join interaction_events e on e.message_id = m.id
            where m.group_id = ? and m.id in ({id_slots}) and m.is_bot = 0
              and m.created_at <= ? order by m.created_at, m.id
        """, (state.group_id, *ids, current.created_at)).fetchall()
        parents = {event.message_id: event.parent_message_id for event in state.events}
        parents.update({item["id"]: item["parent_message_id"] for item in observed})

        def follows(item: sqlite3.Row, target: int) -> bool:
            parent = item["parent_message_id"]
            visited: set[int] = set()
            while parent is not None and parent not in visited:
                if parent == target:
                    return True
                visited.add(parent)
                parent = parents.get(parent)
            return False

        result = []
        for row in rows:
            segments = json.loads(row["trigger_segments"] or "[]")
            trigger_text = "".join(str(item.get("data", {}).get("text", ""))
                                   for item in segments if item.get("type") == "text").strip()
            feedback = [item for item in observed
                        if item["kind"] in {"correction", "tone_feedback", "closed"}
                        and item["created_at"] >= row["created_at"]
                        and (follows(item, row["id"]) or follows(item, row["parent_message_id"]))]
            result.append({
                "message_id": row["id"], "qq_message_id": row["source_message_id"],
                "said": row["text"], "expression_action": row["action"],
                "reply_to_message_id": row["parent_message_id"],
                "trigger_qq_message_id": row["trigger_source_id"], "trigger_said": trigger_text,
                "trigger_user_id": row["trigger_user_id"], "trigger_speaker": row["trigger_speaker"],
                "created_at": row["created_at"], "text_provenance": "acknowledged_send",
                "question_like": bool(QUESTION.search(row["text"])),
                "direct_reply_message_ids": [item["id"] for item in observed if item["parent_message_id"] == row["id"]],
                "subsequent_feedback": [{"message_id": item["id"], "user_id": item["user_id"],
                    "speaker": item["nickname"], "kind": item["kind"], "said": self._utterance(item)[0]}
                    for item in feedback],
            })
        return result


def format_interaction_state(state: InteractionState | None) -> str:
    if state is None:
        return ""

    def evidence(event: InteractionEvidence) -> dict[str, object]:
        result = {"message_id": event.message_id, "qq_message_id": event.source_message_id,
                  "user_id": event.user_id, "speaker": event.nickname, "text": event.text,
                  "kind": event.kind, "reply_to_message_id": event.parent_message_id,
                  "addressed_bot": event.addressed_bot, "created_at": event.created_at,
                  "text_provenance": event.text_provenance, "edge_source": event.edge_source}
        if event.kind == "sent":
            result.update(action=event.action, context_at=event.context_at,
                          context_predates_feedback_ids=[item.message_id for item in state.feedback
                              if event.context_at is not None and item.created_at > event.context_at
                              and event.parent_message_id in state.ancestors(item)])
        return result

    sent = [event for event in state.events if event.kind == "sent"]
    payload = {
        "group_id": state.group_id,
        "root_message_id": state.root_message_id,
        "participants": list(dict.fromkeys(event.user_id for event in state.events)),
        "events": [evidence(event) for event in state.events],
        "question_like_messages_without_bot_send": [event.message_id for event in state.pending_questions],
        "explicit_feedback": [event.message_id for event in state.feedback],
        "last_bot_send": evidence(sent[-1]) if sent else None,
        "last_explicit_closure": next((evidence(event) for event in reversed(state.events) if event.kind == "closed"), None),
    }
    return "<interaction_state>\n" + json.dumps(payload, ensure_ascii=False) + "\n</interaction_state>"
