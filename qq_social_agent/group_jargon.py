from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GroupJargonEntry:
    key: str
    terms: tuple[str, ...]
    explanation: str


GROUP_JARGON_ENTRIES = (
    GroupJargonEntry(
        "plato",
        ("柏拉图",),
        "通常指当前 QQ 群，群名叫“柏拉图”；不是默认指哲学家柏拉图。",
    ),
    GroupJargonEntry(
        "china_alias",
        ("咱妈", "越南"),
        "群内常用来指中国，需要结合语境识别。",
    ),
    GroupJargonEntry(
        "wangzi",
        ("王梓",),
        "群友语境里通常指乌木的暧昧小女友。",
    ),
    GroupJargonEntry(
        "member_enze",
        ("恩泽", "灰機haru", "haru"),
        "指群友灰機haru，群里也会叫恩泽。",
    ),
    GroupJargonEntry(
        "member_wumu",
        ("乌木", "🦕"),
        "乌木和🦕通常指同一个群友。",
    ),
    GroupJargonEntry(
        "member_xuehuo",
        ("血火", "奇恨博士"),
        "血火通常指群友奇恨博士。",
    ),
    GroupJargonEntry(
        "member_kedai",
        ("科蛆代", "科无代"),
        "科蛆代和科无代是同一个人的两个号。",
    ),
    GroupJargonEntry(
        "departed_xhn_xiong",
        ("xhn", "熊熊"),
        "指之前在群里的群友，已经离开群；熊熊以前是 1660502091（雷人）的男朋友，后来变成 xhn。",
    ),
    GroupJargonEntry("japan", ("霓虹",), "日本。"),
    GroupJargonEntry(
        "political_discussion",
        ("见证", "鉴证", "鉴证一下"),
        "通常是在说政治讨论、政治立场判断或键政。",
    ),
    GroupJargonEntry(
        "liberal_slang",
        ("目田", "神游", "神友"),
        "通常指自由派、反感中国或不喜欢中国叙事的人。",
    ),
    GroupJargonEntry(
        "pink",
        ("粉红",),
        "通常指喜欢中国、偏爱国立场的人。",
    ),
    GroupJargonEntry("capitalism", ("zbzy", "资本主义"), "zbzy 是资本主义。"),
    GroupJargonEntry(
        "marx_political_economy",
        ("马克思", "政治经济学", "政经", "阶级", "剩余价值", "生产关系"),
        "群里经常聊马克思、政治经济学、资本主义、阶级、剩余价值、生产关系等相关话题。",
    ),
)

GROUP_JARGON_RULES = """
使用规则：
- 这些词只用于理解群友语义，不要主动像背词典一样解释。
- 黑话命中不代表必须回复，仍按当前插嘴有没有意思判断。
- 涉及中国国内敏感政治、直球抹黑共产党、敏感事件时，仍遵守政治兜底，不要展开危险内容。
""".strip()


def group_jargon_catalog(extra_entries: Iterable[GroupJargonEntry] | None = None) -> str:
    entries = _all_entries(extra_entries)
    lines = ["群内黑话词典候选："]
    for entry in entries:
        lines.append(f"- {', '.join(entry.terms)}：{entry.explanation}")
    return "\n".join(lines)


def group_jargon_context(
    selected_terms: Iterable[str] | None = None,
    extra_entries: Iterable[GroupJargonEntry] | None = None,
) -> str:
    entries = _all_entries(extra_entries)
    if selected_terms is not None:
        selected_keys = _selected_keys(selected_terms, entries)
        entries = tuple(entry for entry in entries if entry.key in selected_keys)
        if not entries:
            return ""

    lines = ["本轮命中的群内黑话："]
    for entry in entries:
        lines.append(f"- {', '.join(entry.terms)}：{entry.explanation}")
    lines.append("")
    lines.append(GROUP_JARGON_RULES)
    return "\n".join(lines)


def detect_group_jargon_terms(
    texts: Iterable[str],
    extra_entries: Iterable[GroupJargonEntry] | None = None,
) -> tuple[str, ...]:
    joined = "\n".join(texts)
    joined_lower = joined.lower()
    matched: list[str] = []
    for entry in _all_entries(extra_entries):
        for term in entry.terms:
            haystack = joined_lower if term.isascii() else joined
            needle = term.lower() if term.isascii() else term
            if needle in haystack:
                matched.append(entry.key)
                break
    return tuple(matched)


def group_jargon_known_terms() -> tuple[str, ...]:
    terms: list[str] = []
    for entry in GROUP_JARGON_ENTRIES:
        terms.append(entry.key)
        terms.extend(entry.terms)
    return tuple(terms)


def _all_entries(extra_entries: Iterable[GroupJargonEntry] | None = None) -> tuple[GroupJargonEntry, ...]:
    if extra_entries is None:
        return GROUP_JARGON_ENTRIES
    return (*GROUP_JARGON_ENTRIES, *tuple(extra_entries))


def _selected_keys(selected_terms: Iterable[str], entries: Iterable[GroupJargonEntry]) -> set[str]:
    selected = {term.strip().lower() for term in selected_terms if term.strip()}
    keys: set[str] = set()
    for entry in entries:
        aliases = {entry.key.lower(), *(term.lower() for term in entry.terms)}
        if selected & aliases:
            keys.add(entry.key)
    return keys
