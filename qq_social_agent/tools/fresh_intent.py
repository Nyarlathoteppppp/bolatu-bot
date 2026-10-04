"""Search intent detection and query normalization."""
from __future__ import annotations

import re

from .fresh_text import (
    _clean_text,
)
from .fresh_types import (
    FreshIntent,
)


def fresh_kind_from_text(text: str) -> str | None:
    intent = detect_fresh_intent(text)
    return intent.kind if intent else None


def detect_fresh_intent(text: str) -> FreshIntent | None:
    raw_text = str(text or "")
    full_text = re.sub(r"\s+", " ", raw_text).strip()
    normalized = _normalize_query(full_text)
    compact = re.sub(r"\s+", "", full_text.casefold())
    if not compact or _is_low_value_fresh_query(compact):
        return None

    explicit_query = None
    explicit_source = full_text
    for candidate in _explicit_search_candidate_texts(raw_text):
        explicit_query = _explicit_search_query(candidate)
        if explicit_query is None:
            explicit_query = _embedded_explicit_search_query(candidate)
        if explicit_query is not None:
            explicit_source = candidate
            break
    explicit = explicit_query is not None
    kind = _classify_fresh_kind(explicit_source if explicit else full_text, explicit=explicit)
    if kind is None:
        return None
    if explicit_query is not None and _is_weak_search_object(explicit_query):
        explicit_query = None
        explicit = False
        kind = _classify_fresh_kind(full_text, explicit=False)
        if kind is None:
            return None
    query = (
        _clean_explicit_search_query(explicit_query or "")
        if explicit_query is not None
        else _fresh_query_from_text(_current_reply_text(full_text) or normalized)
    )
    query = _compact_search_query(query)
    if _is_low_value_fresh_query(query) or _is_weak_search_object(query):
        return None
    return FreshIntent(
        query=query,
        kind=kind,
        explicit=explicit,
        required=explicit or _requires_fresh_verification(full_text),
    )


def should_use_fresh_context(query: str, fallback_text: str = "") -> bool:
    query = _normalize_query(query)
    if not query or _is_low_value_fresh_query(query):
        return False
    return detect_fresh_intent(f"{query} {fallback_text}") is not None


def _normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    return query[:120]


def _explicit_search_candidate_texts(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    current = _current_reply_text(re.sub(r"\s+", " ", str(text or "")).strip())
    if current:
        candidates.append(current)
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[.、)）]\s*", "", line)
        for separator in ("说：", "说:", "：", ":"):
            if separator in line:
                tail = line.split(separator, 1)[-1].strip()
                if tail:
                    candidates.append(tail)
                    break
        candidates.append(line)
    candidates.append(str(text or ""))
    output: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        clean = re.sub(r"\s+", " ", item).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
    return tuple(output)


def _clean_explicit_search_query(query: str) -> str:
    clean = _normalize_query(query)
    clean = re.sub(
        r"^(?:讲讲|讲一下|说说|说一下|聊聊|介绍一下|看看|看一下|分析一下|讲一讲|说一说|关于)\s*",
        "",
        clean,
    ).strip(" ，,：:")
    clean = re.sub(r"^(?:一下|下|这个|这件事|这东西)\s*", "", clean).strip(" ，,：:")
    return _compact_search_query(clean or query)


def _fresh_query_from_text(text: str) -> str:
    query = _normalize_query(text)
    explicit_query = _explicit_search_query(query)
    if explicit_query is not None:
        return _normalize_query(explicit_query)
    query = re.sub(r"(现在|今天)?(怎么样了|怎么了|是什么情况|咋了|如何了)$", "", query).strip()
    query = re.sub(r"(最新消息|最新新闻|新闻|赛果|比分|结果)$", "", query).strip()
    return _compact_search_query(query or text)


_QUERY_STOPWORDS = {
    "让",
    "使",
    "把",
    "将",
    "被",
    "给",
    "对",
    "向",
    "与",
    "和",
    "及",
    "以及",
    "的",
    "了",
    "着",
    "过",
    "是",
    "在",
    "有",
    "如何",
    "怎样",
    "怎么",
    "什么",
    "哪些",
    "哪个",
    "这个",
    "那个",
    "成为",
    "一下",
    "讲讲",
    "看看",
    "说说",
}
_QUERY_TITLE_HINT_RE = re.compile(r"[：:]|如何成为|怎样成为|关键能力|一文看懂|深度解析")


def _compact_search_query(query: str) -> str:
    clean = _normalize_query(query)
    if not clean:
        return ""
    compact = re.sub(r"[\s，。！？,.!?]+", "", clean)
    looks_like_title = (
        len(compact) > 40
        or bool(_QUERY_TITLE_HINT_RE.search(clean))
        or ("\"" in clean or "“" in clean or "”" in clean)
    )
    if not looks_like_title:
        return clean
    split_source = clean
    for stopword in sorted(_QUERY_STOPWORDS, key=len, reverse=True):
        split_source = split_source.replace(stopword, " ")
    raw_tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9._+-]*|[\u4e00-\u9fff]{2,}",
        split_source,
    )
    tokens: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        normalized = token.strip()
        if not normalized or normalized in _QUERY_STOPWORDS:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(normalized)
        if len(tokens) >= 6:
            break
    compacted = " ".join(tokens[:6]).strip()
    compacted_len = len(re.sub(r"\s+", "", compacted))
    if compacted_len <= 2:
        return clean
    return compacted[:120]


def _current_reply_text(text: str) -> str:
    """Extract the current speaker's part from the enriched QQ reply wrapper."""

    if "回复" not in text or "消息【" not in text or not text.endswith("】"):
        return ""
    for separator in ("：", ":"):
        if separator not in text:
            continue
        current = text.rsplit(separator, 1)[-1].removesuffix("】").strip()
        if current:
            return current
    return ""


def _is_low_value_fresh_query(text: str) -> bool:
    compact = re.sub(r"[\s，。！？,.!?]+", "", text.lower())
    if not compact:
        return True
    low_value_tokens = (
        "你好",
        "美好",
        "测试",
        "周几",
        "星期几",
        "几点",
        "日期",
        "乱码",
        "随便搜搜",
        "你能搜什么",
        "搜索功能",
        "联网功能",
    )
    if any(token in compact for token in low_value_tokens):
        return True
    return len(compact) <= 2


def _is_weak_search_object(text: str) -> bool:
    compact = re.sub(r"[\s，。！？,.!?]+", "", str(text or "").casefold())
    if not compact or len(compact) <= 2:
        return True
    weak_exact = {
        "然后",
        "开始",
        "思路",
        "看看",
        "说说",
        "这个",
        "那个",
        "一下",
        "东西",
        "然后开始",
        "想个思路",
        "然后开始想个思路",
        "随便",
        "那个东西",
        "这件事",
    }
    if compact in weak_exact:
        return True
    if compact.startswith("然后开始") and len(compact) <= 12:
        return True
    return False


_EXPLICIT_SEARCH_RE = re.compile(
    r"^\s*"
    r"(?:(?:张风雪|风雪)[，,：:\s]*)?"
    r"(?:(?:请|麻烦|你能不能|你可以|能不能|可以)\s*)?"
    r"(?:"
    r"帮我\s*找(?:一下)?|"
    r"(?:帮我|你)?\s*(?:去|来)?\s*(?:"
    r"联网(?:搜索|搜|查|看)(?:一下)?|"
    r"网上(?:搜索|搜|查|找|看)(?:一下)?|"
    r"上网(?:搜索|搜|查|找|看)(?:一下)?|"
    r"搜索(?:一下)?|搜一下|搜搜|搜|查一下|查查|查"
    r")"
    r")"
    r"[，,：:\s]*(?P<query>.+?)\s*$",
    flags=re.IGNORECASE,
)


def _explicit_search_query(text: str) -> str | None:
    match = _EXPLICIT_SEARCH_RE.match(text)
    if match is None:
        return None
    query = _normalize_query(match.group("query"))
    return query or None


def _embedded_explicit_search_query(text: str) -> str | None:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value or any(token in value for token in ("搜索功能", "联网功能", "搜搜功能")):
        return None
    pattern = (
        r"(?:称|说|让|叫|要|想|在)?\s*(?:你)?\s*"
        r"(?:联网(?:搜索|搜|查|看)(?:一下)?|网上(?:搜索|搜|查|找|看)(?:一下)?|"
        r"上网(?:搜索|搜|查|找|看)(?:一下)?|搜索(?:一下)?(?!功能)|搜一下|搜搜|查一下|查查)"
        r"[，,：:\s]*(?P<query>[^。！？!；;\n]{2,80})"
    )
    match = re.search(pattern, value, flags=re.IGNORECASE)
    if match is None:
        return None
    query = _normalize_query(match.group("query"))
    return query or None


def _classify_fresh_kind(text: str, *, explicit: bool) -> str | None:
    lowered = text.casefold()
    sports_terms = (
        "赛果",
        "比分",
        "赛程",
        "世界杯",
        "msi",
        "nba",
        "欧冠",
        "英超",
        "比赛",
        "赛事",
        "战绩",
    )
    news_terms = (
        "新闻",
        "消息",
        "热点",
        "局势",
        "冲突",
        "战争",
        "政策",
        "发布会",
        "通报",
        "事故",
        "地震",
        "台风",
        "选举",
        "进展",
    )
    news_subject_terms = news_terms + (
        "美国",
        "伊朗",
        "以色列",
        "乌克兰",
        "俄罗斯",
        "政府",
        "公司",
        "游戏",
    )
    fresh_terms = (
        "最新",
        "刚刚",
        "刚才",
        "今天",
        "今年",
        "本届",
        "现在",
        "目前",
        "发生什么",
        "怎么了",
        "怎么样了",
        "结果",
    )
    has_sports = any(term in lowered for term in sports_terms)
    has_news = any(term in lowered for term in news_terms)
    has_freshness = any(term in lowered for term in fresh_terms)

    if has_sports and (explicit or has_freshness):
        return "sports"
    if explicit:
        return "news" if has_news else "web"
    if _requires_fresh_verification(text):
        academic_terms = ("菲奖", "菲尔兹", "学术", "论文", "猜想", "定理", "期刊", "大学")
        return "web" if any(term in lowered for term in academic_terms) else "news"
    if has_freshness and any(term in lowered for term in news_subject_terms):
        return "news"
    if has_freshness and any(term in lowered for term in ("版本", "文档", "官网", "更新", "发布")):
        return "web"
    return None


def _requires_fresh_verification(text: str) -> bool:
    """Identify concrete current outcomes that should never rely on stale model memory."""

    lowered = text.casefold()
    time_terms = (
        "今天",
        "今年",
        "本届",
        "刚刚",
        "刚才",
        "最新",
        "目前",
        "现在已经",
        "已经确定",
        "已经公布",
    )
    outcome_terms = (
        "得主",
        "获奖",
        "拿到",
        "名单",
        "颁奖",
        "当选",
        "夺冠",
        "冠军",
        "排名",
        "入选",
        "官宣",
        "公布",
        "发布",
        "实锤",
        "确定",
        "解决了",
        "证明了",
    )
    return any(term in lowered for term in time_terms) and any(
        term in lowered for term in outcome_terms
    )


def _safe_external_query(query: str, *, max_chars: int = 120) -> str:
    clean = _clean_text(str(query or ""))
    clean = re.sub(
        r"(?i)\b(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]{8,}",
        "[已隐藏令牌]",
        clean,
    )
    clean = re.sub(
        r"(?i)\b(?:api[_\s-]?key|access[_\s-]?token|refresh[_\s-]?token|token|secret|password)"
        r"\s*[:=：]\s*[^\s，,;；]{6,}",
        "[已隐藏密钥]",
        clean,
    )
    clean = re.sub(r"(?i)\b(?:sk|pk)[-_][A-Za-z0-9_-]{12,}\b", "[已隐藏密钥]", clean)
    clean = re.sub(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+",
        r"\1[已隐藏]",
        clean,
    )
    clean = re.sub(r"(?<!\d)\d{7,12}(?!\d)", "[QQ号]", clean)
    clean = re.sub(r"[\x00-\x1f\x7f]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if _looks_like_bad_external_query(clean):
        return ""
    return clean[: max(1, int(max_chars))].rstrip()


def _looks_like_bad_external_query(query: str) -> bool:
    compact = re.sub(r"[\s，。！？,.!?~～]+", "", query.casefold())
    if not compact:
        return True
    if _has_external_lookup_signal(compact):
        return False
    abstract_tokens = ("平行宇宙", "美少女权", "被踢", "踢了", "踢出去", "给踢", "被骂", "骂了")
    if any(token in compact for token in abstract_tokens):
        return True
    if len(compact) <= 28 and re.search(r"^[你我他她它].{0,16}(?:被|给|把).{0,16}(?:踢|骂|打|杀|大肆|达斯|火宅)", compact):
        return True
    return False


def _has_external_lookup_signal(compact: str) -> bool:
    signal_terms = ("搜", "查", "最新", "现在", "今天", "今年", "新闻", "发生什么", "怎么了", "官网", "文档", "发布", "官宣", "赛程", "比分", "价格", "行情", "股票", "美股", "币价", "比特币", "以太坊")
    return any(term in compact for term in signal_terms)
