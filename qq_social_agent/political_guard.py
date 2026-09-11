from __future__ import annotations

import re


SAFE_POLITICAL_REDIRECT = ""

_TARGET_PARTY_RE = re.compile(r"(中国共产党|共产党|中共|ccp|党国)", re.IGNORECASE)
_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|邪恶|纳粹|屠杀|血债|"
    r"卖国|汉奸|垃圾|傻逼|傻卵|畜生|狗|烂透|腐败透顶)",
    re.IGNORECASE,
)

# 只保留直球敏感专名。过短、过日常的词（64/包子/连任/新疆/白纸/喝茶）不打码，避免误伤闲聊。
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

# 敏感词输出用整词 “*拼音首字母*”：习近平 -> *xjp*，共产党 -> *gcd*。
_PINYIN_CHARS = "三上下世东中丰丹主乌九习乡书事五产人件任会伞佳修元光克党全八公六共兵再刁刘刚制功加动劳包化南占卫县反取台右吾周命哥喝四困国场坎坦城基墙士大天太失夹女姚媒子学安宪家封尔尼局山工帅师希帝席帮平年广庆店康开张强彪彭律志态总慧批摘政教文斗新族时明春晓晟晶智朗期木李来林桥毛民永江沟河治法波泽洪活派海消涛清港湾潮火灭灾焚熙犬独王班瓶生疆登白禅离种秦突立站章类紫红纪纸终绝维网耀育胡自色茶荒营蔡薄藏蟹被西记许评诚贵赖赵跃身轮辱边达运近进连迫退送通速邦郭铁铜链锣锦门阳陈难集雨零霞青革颜饥香高鲁黑齐"
_PINYIN_INITIALS = "ssxsdzfdzwjxxsswcrjrhsjxygkdqbglgbzdlgzgjdlbhnzwxfqtywzmghskgcktcjqsdttsjnymzxaxjfenjsgssxdxbpngqdkkzqbplztzhpzzjwdxzsmcxcjzlqmlllqmmyjghzfbzhhphxtqgwchmzfxqdwbpsjdbclzqtlzzlzhjzzjwwyyhzschycbcxbxjxpcglzyslrbdyjjlptstsbgttlljmycnjylxqgyjxglhq"
_PINYIN_INITIAL_MAP = dict(zip(_PINYIN_CHARS, _PINYIN_INITIALS))
_PINYIN_INITIAL_MAP.update(
    {
        "暴": "b",
        "打": "d",
        "倒": "d",
        "推": "t",
        "翻": "f",
        "下": "x",
        "灭": "m",
        "亡": "w",
        "垮": "k",
        "裁": "c",
        "纳": "n",
        "粹": "c",
        "屠": "t",
        "杀": "s",
        "血": "x",
        "债": "z",
        "腐": "f",
        "败": "b",
        "透": "t",
        "顶": "d",
    }
)


def _term_pattern(term: str) -> str:
    escaped = re.escape(term)
    if term.isascii() and term.lower() in _SHORT_ASCII_TERMS:
        return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return escaped


_SENSITIVE_DOMESTIC_RE = re.compile(
    "|".join(_term_pattern(term) for term in sorted(_SENSITIVE_TERMS, key=len, reverse=True))
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


def sanitize_political_output(reply: str) -> tuple[str, bool]:
    if not reply:
        return reply, False
    masked, count = _MASK_RE.subn(_mask_match, reply)
    if _TARGET_PARTY_RE.search(reply) and _ATTACK_RE.search(reply):
        masked, attack_count = _MASK_ATTACK_RE.subn(_mask_match, masked)
        count += attack_count
    return masked, count > 0


def _mask_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    initials: list[str] = []
    for ch in raw:
        if ch.isspace():
            continue
        initials.append(_initial_for_mask(ch))
    token = "".join(initials) or "*"
    return f"*{token}*"


def _initial_for_mask(ch: str) -> str:
    if "\u4e00" <= ch <= "\u9fff":
        return _PINYIN_INITIAL_MAP.get(ch, "*")
    if ch.isalpha():
        return ch.lower()
    if ch.isdigit():
        return ch
    return "*"
