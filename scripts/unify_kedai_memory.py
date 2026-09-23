from __future__ import annotations

import json
from pathlib import Path

from qq_social_agent.memory import MemoryStore

GID = 1026813421
PRIMARY = 3066256514
ALT = 2947279300
ACTOR = 1535071184

EDU = (
    "邪恶代代/科有代/3066256514 与 纯真代代/科无代/2947279300 是同一个人的两个号"
    "（主号 3066256514，小号 2947279300）。学历：南开本科，北大软微硕士。"
    "不要串到歌迷老蛆或张风雪身上，也不要当成两个群友。"
)
RELATION = (
    "邪恶代代/科有代/3066256514 和 纯真代代/科无代/2947279300 是同一个人的两个号。"
    "主号是 3066256514，小号是 2947279300。观点、学历和偏好按同一个人记。"
)
ALT_IDENTITY = "纯真代代/科无代/2947279300 是邪恶代代/科有代/3066256514 的小号，同一人。"
STYLE = (
    "邪恶科代、邪恶代代、可爱代代、纯真代代、科无代、科蛆代都是同一个人"
    "（主号 3066256514，小号 2947279300）；她长期喜欢反串、夸张政治/民族梗和抽象表达，"
    "这是稳定说话风格和长期人设事实，应长期记住。"
)
PRIMARY_ALIASES = [
    "邪恶代代",
    "邪恶科代",
    "科有代",
    "纯真代代",
    "科无代",
    "科蛆代",
    "I Hate the World and Everything In It",
    "科有代（非北本）",
]
ALT_ALIASES = [
    "纯真代代",
    "科无代",
    "科蛆代",
    "科蛆代小号",
    "邪恶代代",
    "科有代",
    "邪恶科代",
    "Corday",
]


def main() -> None:
    store = MemoryStore(Path("data/bot.sqlite3"))
    expired = []
    for atom_id, reason in (
        (99, "同一人双号，不再按 user_id 拆开归属"),
        (1322, "当前身份不是本科生；南开本科已毕业，现北大软微硕士"),
        (1324, "张风雪开学升大三串记到纯真代代名下"),
    ):
        atom = store.memory_atom(atom_id)
        if atom is None:
            expired.append((atom_id, "missing"))
            continue
        if atom.status not in {"active", "disputed"}:
            expired.append((atom_id, f"status={atom.status}"))
            continue
        ok = store.expire_memory_atom(
            atom_id,
            reason=reason,
            source="manual_kedai_same_person",
            actor_user_id=ACTOR,
        )
        expired.append((atom_id, "expired" if ok else "expire_false"))

    corrected = {}
    edu = store.memory_atom(1065)
    if edu and edu.status in {"active", "disputed"}:
        corrected["1065"] = store.correct_memory_atom(
            1065,
            content=EDU,
            source="builtin_xiee_daida_education_identity",
            actor_user_id=ACTOR,
            reason="主号小号同一人，学历共用",
            confidence=1.0,
            importance=0.95,
        )
    style = store.memory_atom(982)
    if style and style.status in {"active", "disputed"}:
        corrected["982"] = store.correct_memory_atom(
            982,
            content=STYLE,
            source="manual_kedai_profile_fix",
            actor_user_id=ACTOR,
            reason="风格人设覆盖两个号",
            confidence=1.0,
            importance=0.95,
        )

    relation_id = store.upsert_memory_atom(
        atom_type="relation",
        group_id=GID,
        subject_user_id=PRIMARY,
        object_user_id=ALT,
        content=RELATION,
        source="builtin_kedai_same_person",
        confidence=1.0,
        importance=1.0,
    )
    alt_id = store.upsert_memory_atom(
        atom_type="identity",
        group_id=GID,
        subject_user_id=ALT,
        object_user_id=PRIMARY,
        content=ALT_IDENTITY,
        source="builtin_kedai_alt_account_identity",
        confidence=1.0,
        importance=0.95,
    )
    edu_id = store.upsert_memory_atom(
        atom_type="identity",
        group_id=GID,
        subject_user_id=PRIMARY,
        object_user_id=None,
        content=EDU,
        source="builtin_xiee_daida_education_identity",
        confidence=1.0,
        importance=0.95,
    )

    for user_id, aliases in ((PRIMARY, PRIMARY_ALIASES), (ALT, ALT_ALIASES)):
        row = store.conn.execute(
            "select display_name from member_profiles where group_id=? and user_id=?",
            (GID, user_id),
        ).fetchone()
        store.conn.execute(
            "update member_profiles set aliases_json=? where group_id=? and user_id=?",
            (json.dumps(aliases, ensure_ascii=False), GID, user_id),
        )
        print("aliases", user_id, aliases, "display", None if row is None else row[0])
    store.conn.commit()
    print("expired", expired)
    print("corrected", corrected)
    print("relation", relation_id, "alt", alt_id, "edu", edu_id)


if __name__ == "__main__":
    main()
