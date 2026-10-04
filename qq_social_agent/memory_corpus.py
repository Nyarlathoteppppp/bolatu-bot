"""Shared existing raw-corpus classification helpers."""
from __future__ import annotations

from .memory_text import _compact_text


def _raw_corpus_tags(text: str) -> tuple[str, ...]:
    compact = _compact_text(text)
    tag_patterns = (
        ("玩梗", ("草", "哈哈", "笑死", "绷", "典", "麻了", "乐", "抽象", "梗", "开宰")),
        ("互损", ("傻逼", "弱智", "废物", "滚", "爹", "别学", "不如", "唐")),
        ("安慰", ("难受", "不开心", "顶不住", "撑不住", "压力", "破防", "烦死", "累")),
        ("反串", ("建议", "支持", "感觉不如", "这下", "赢", "什么成分")),
        ("政治", ("政治", "资本", "无产", "阶级", "粉红", "神友", "咱妈", "霓虹", "美国", "日本")),
        ("代码", ("代码", "bug", "报错", "炸了", "python", "java", "ai", "模型", "api")),
        ("倒霉", ("亏", "完蛋", "坏了", "炸了", "寄", "崩", "没人理")),
        ("恋爱", ("老婆", "喜欢", "暧昧", "女友", "男朋友", "宝宝")),
        ("行情", ("股票", "美股", "比特币", "btc", "eth", "亏钱", "涨", "跌")),
    )
    tags: list[str] = []
    for tag, patterns in tag_patterns:
        if any(pattern in compact for pattern in patterns):
            tags.append(tag)
    return tuple(tags)


def _is_low_value_raw_corpus_text(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return True
    if len(compact) <= 1:
        return True
    return compact in {
        "6",
        "66",
        "666",
        "草",
        "哈哈",
        "哈哈哈",
        "嗯",
        "哦",
        "好",
        "好的",
        "可以",
        "绷",
    }
