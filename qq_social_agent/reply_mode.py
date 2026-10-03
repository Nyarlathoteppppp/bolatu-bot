"""How Fengxue picks up a line: follow it, branch from it, or her own call.

A reply that only ever mirrors the last message reads like an echo; people
also hear one thing and bring up what it reminds them of. The three ways are
drawn evenly for every chat reply. A direct question is answered first and
the drawn way shapes the rest; tone in serious moments is left to the persona.
"""
from __future__ import annotations

import random
from typing import Callable

FOLLOW = "follow"
ASSOCIATE = "associate"
FREE = "free"

MODE_GUIDES = {
    FOLLOW: "本轮接法：顺着对方这句聊，围绕对方说的那一点回应。",
    ASSOCIATE: (
        "本轮接法：联想展开。这一轮不要停在对方这句本身：从里面抓一个点，联想到一个相关但更大、或更贴近你自己兴趣的话题"
        "（行业、技术、社会现象、你自己的经历或打算），说出你在那个话题上的看法。可以先用半句接住对方，再把话头带过去；"
        "要让人看得出联想的来路，别替对方下结论。"
        "示意（别照搬）：有人吐槽某个国产 AI 难用，你聊到中美 AI 在算力和数据上的差距。"
    ),
    FREE: "本轮接法：随你发挥。顺着说、联想开去、吐槽、分享自己的事都行，选此刻最像你会说的那种。",
}
ANSWER_FIRST = "对方问了你具体的问题，先把它答清楚；"


def pick_reply_mode(rng: Callable[[], float] = random.random) -> str:
    roll = rng()
    if roll < 1 / 3:
        return FOLLOW
    if roll < 2 / 3:
        return ASSOCIATE
    return FREE


def reply_mode_guide(mode: str, *, addressed_question: bool) -> str:
    guide = MODE_GUIDES[mode]
    if not addressed_question:
        return guide
    return "本轮接法：" + ANSWER_FIRST + "之后" + guide.removeprefix("本轮接法：")
