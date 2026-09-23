from __future__ import annotations

import re
from dataclasses import dataclass


SAFE_POLITICAL_REDIRECT = ""

_TARGET_PARTY_RE = re.compile(r"(中国共产党|共产党|中共|ccp|党国)", re.IGNORECASE)
_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|邪恶|纳粹|屠杀|血债|"
    r"卖国|汉奸|垃圾|傻逼|傻卵|畜生|狗|烂透|腐败透顶)",
    re.IGNORECASE,
)

# 只保留直球敏感专名。过短、过日常的词（64/包子/连任/新疆/白纸/喝茶）不打码，避免误伤闲聊。
# 黑话（腊肉、腊爹、咱妈、火宅）不在此列。
_SENSITIVE_TERMS = (
    "文化大革命",
    "cultural revolution",
    "wenge",
    "文革",
    "反右",
    "黑五类",
    "三反五反",
    "大跃进",
    "三年困难时期",
    "三年大饥荒",
    "大饥荒",
    "夹边沟",
    "红卫兵",
    "批斗",
    "上山下乡",
    "四人帮",
    "林彪事件",
    "反革命",
    "天安门广场事件",
    "天安门事件",
    "八九六四",
    "64事件",
    "8964",
    "六四",
    "坦克人",
    "tankman",
    "白纸运动",
    "四通桥",
    "乌鲁木齐火灾",
    "佳士工人",
    "铜锣湾书店",
    "雨伞运动",
    "占中",
    "721元朗",
    "831太子站",
    "胡锦涛离场",
    "铁链女",
    "习近平",
    "xi jinping",
    "xjp",
    "习主席",
    "习大大",
    "毛泽东",
    "mzd",
    "毛主席",
    "教员",
    "邓小平",
    "江泽民",
    "胡锦涛",
    "总书记",
    "维尼",
    "习禁评",
    "习总",
    "邓公",
    "小平同志",
    "梁家河",
    "赵紫阳",
    "刘晓波",
    "胡耀邦",
    "薄熙来",
    "周永康",
    "秦刚",
    "彭帅",
    "任志强",
    "许志永",
    "高智晟",
    "陈光诚",
    "蔡霞",
    "郭文贵",
    "王丹",
    "吾尔开希",
    "江青",
    "张春桥",
    "姚文元",
    "王洪文",
    "大纪元",
    "明慧网",
    "九评共产党",
    "法轮功",
    "falun gong",
    "轮媒",
    "退党",
    "活摘",
    "中国共产党",
    "共产党",
    "gcd",
    "中共",
    "CCP",
    "ccp",
    "党国",
    "新疆种族灭绝",
    "维吾尔种族灭绝",
    "维吾尔集中营",
    "新疆集中营",
    "东突",
    "世维会",
    "西藏独立",
    "藏独",
    "达赖",
    "香港国安法",
    "反送中",
    "港独",
    "台湾独立",
    "台独",
    "民主运动",
    "颜色革命",
    "零八宪章",
    "辱包",
    "习包子",
    "包帝",
    "庆丰帝",
    "刁大犬",
    "总加速师",
    "墙国",
    "赵家人",
)

_EXTRA_SENSITIVE_PATTERNS = (
    r"(?<![A-Za-z0-9])cultural\s*revolution(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])falun\s*gong(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])xi\s*jinping(?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])x\s*j\s*p(?![A-Za-z0-9])",
)

_SHORT_ASCII_TERMS = frozenset({"ccp", "xjp", "mzd", "gcd", "wenge", "tankman"})


def _term_pattern(term: str) -> str:
    escaped = re.escape(term)
    # Match simple obfuscation without normalizing or rewriting the source text.
    separator = r"[\s·._\u200b-\u200d\ufeff]{0,3}"
    if term.isascii() and term.lower() in _SHORT_ASCII_TERMS:
        spaced = separator.join(re.escape(char) for char in term)
        return rf"(?<![A-Za-z0-9]){spaced}(?![A-Za-z0-9])"
    if len(term) >= 2 and all("\u4e00" <= char <= "\u9fff" for char in term):
        return separator.join(re.escape(char) for char in term)
    return escaped


# These existing entries have common non-political senses. Keep them out of the
# hard mask and ask Jev about each occurrence, not the entire message/topic.
_CONTEXTUAL_TERMS = frozenset({"教员", "维尼", "gcd", "wenge"})
_CONTEXTUAL_RE = re.compile(
    "|".join(_term_pattern(term) for term in sorted(_CONTEXTUAL_TERMS)), re.IGNORECASE,
)


@dataclass(frozen=True)
class PoliticalCandidate:
    key: str
    start: int
    end: int
    text: str


def political_candidates(text: str) -> tuple[PoliticalCandidate, ...]:
    return tuple(
        PoliticalCandidate(f"span_{index}", match.start(), match.end(), match.group())
        for index, match in enumerate(_CONTEXTUAL_RE.finditer(text or ""))
    )


_SENSITIVE_DOMESTIC_RE = re.compile(
    "|".join(_term_pattern(term) for term in sorted(set(_SENSITIVE_TERMS) - _CONTEXTUAL_TERMS, key=len, reverse=True))
    + "|"
    + "|".join(_EXTRA_SENSITIVE_PATTERNS),
    re.IGNORECASE,
)
_MASK_RE = _SENSITIVE_DOMESTIC_RE
_MASK_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|纳粹|屠杀|血债|腐败透顶)",
    re.IGNORECASE,
)


def has_political_redline(_text: str) -> bool:
    """Input intercept disabled: keyword hits must not block or canned-redirect chat."""
    return False


def political_safe_reply() -> str:
    return SAFE_POLITICAL_REDIRECT


@dataclass(frozen=True)
class PoliticalSanitizeResult:
    public_text: str
    original_text: str
    hits: tuple[str, ...]
    guarded: bool


def sanitize_political_output(reply: str) -> tuple[str, bool]:
    result = sanitize_political_output_detail(reply)
    return result.public_text, result.guarded


def sanitize_political_output_detail(
    reply: str, *, contextual_keys: tuple[str, ...] | None = None,
) -> PoliticalSanitizeResult:
    if not reply:
        return PoliticalSanitizeResult(reply, reply, (), False)
    hits: list[str] = []

    def collect(match: re.Match[str]) -> str:
        hits.append(match.group(0))
        return _mask_match(match)

    masked, count = _MASK_RE.subn(collect, reply)
    if _TARGET_PARTY_RE.search(reply) and _ATTACK_RE.search(reply):
        masked, attack_count = _MASK_ATTACK_RE.subn(collect, masked)
        count += attack_count
    # Keys resolve only against code-extracted spans in this exact text. Jev
    # cannot supply replacement text or arbitrary offsets. Masks preserve length.
    candidates = political_candidates(reply)
    # Political output is the explicitly fail-closed exception: unavailable
    # contextual judgement preserves the original conservative mask policy.
    selected = {candidate.key for candidate in candidates} if contextual_keys is None else set(contextual_keys)
    for candidate in candidates:
        if candidate.key not in selected:
            continue
        masked = masked[:candidate.start] + "*" * (candidate.end - candidate.start) + masked[candidate.end:]
        hits.append(candidate.text)
        count += 1
    unique_hits = tuple(dict.fromkeys(hits))
    return PoliticalSanitizeResult(
        public_text=masked,
        original_text=reply,
        hits=unique_hits,
        guarded=count > 0,
    )


def format_gag_memory(public_text: str, hits: tuple[str, ...] | list[str]) -> str:
    if not public_text or not hits:
        return public_text
    joined = "、".join(dict.fromkeys(str(item) for item in hits if str(item).strip()))
    if not joined:
        return public_text
    return f"{public_text}（内容{joined}已被风雪你的口球屏蔽成***）"


def prepare_group_political_output(reply: str) -> PoliticalSanitizeResult:
    result = sanitize_political_output_detail(reply)
    if not result.guarded:
        return result
    return PoliticalSanitizeResult(
        public_text=result.public_text,
        original_text=result.original_text,
        hits=result.hits,
        guarded=True,
    )


def _mask_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    return "".join("*" if not ch.isspace() else ch for ch in raw)
