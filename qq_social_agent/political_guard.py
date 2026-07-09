from __future__ import annotations

import re


SAFE_POLITICAL_REDIRECT = "这话题别在群里直球冲塔，没必要把聊天往敏感政治上带。换个能聊的。"

_TARGET_PARTY_RE = re.compile(r"(中国共产党|共产党|中共|ccp|党国)")
_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|邪恶|纳粹|屠杀|血债|"
    r"卖国|汉奸|垃圾|傻逼|傻卵|畜生|狗|烂透|腐败透顶)"
)
_SENSITIVE_DOMESTIC_RE = re.compile(
    r"("
    r"六四|八九六四|8964|64事件|天安门事件|天安门广场事件|坦克人|tankman|"
    r"白纸运动|四通桥|彭载舟|李文亮|709律师|乌坎|佳士工人|"
    r"法轮功|退党|活摘|"
    r"新疆集中营|新疆种族灭绝|维吾尔种族灭绝|维吾尔集中营|强迫劳动|"
    r"西藏独立|藏独|达赖|"
    r"港独|反送中|香港国安法|"
    r"台独|台湾独立"
    r")",
    re.IGNORECASE,
)


def has_political_redline(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if _SENSITIVE_DOMESTIC_RE.search(normalized):
        return True
    return bool(_TARGET_PARTY_RE.search(normalized) and _ATTACK_RE.search(normalized))


def political_safe_reply() -> str:
    return SAFE_POLITICAL_REDIRECT


def sanitize_political_output(reply: str) -> tuple[str, bool]:
    if has_political_redline(reply):
        return political_safe_reply(), True
    return reply, False


def _normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
