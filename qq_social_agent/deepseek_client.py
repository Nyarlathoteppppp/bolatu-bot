from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from nonebot import logger
from openai import AsyncOpenAI

from .config import DeepSeekConfig
from .memory import ChatMessage
from .persona import Persona


@dataclass(frozen=True)
class ToolSymbol:
    kind: str
    symbol: str
    display: str


@dataclass(frozen=True)
class ReplyDecision:
    should_reply: bool
    confidence: float
    reason: str
    mode: str = "silent"
    action: str = "ignore"
    need_tool: bool = False
    tool: str = ""
    symbols: tuple[ToolSymbol, ...] = ()
    comment_after_tool: bool = False
    need_fresh_context: bool = False
    fresh_query: str = ""
    fresh_kind: str = "news"


@dataclass(frozen=True)
class MidMemoryDraft:
    summary: str
    recall_cues: tuple[str, ...]


@dataclass(frozen=True)
class StyleRuleDraft:
    situation: str
    style: str
    source_text: str = ""


@dataclass(frozen=True)
class ReplyCandidateDraft:
    text: str
    action: str
    style: str


SOCIAL_ACTIONS = {
    "ignore",
    "reply",
    "answer",
    "agree",
    "tease",
    "ask_back",
    "mock_repeated_question",
    "at_someone",
    "market_check",
    "fresh_context",
}

ACTION_GUIDES = {
    "reply": "行动：普通接话。结合群友聊天内容接一句话。",
    "answer": "行动：正常回答。群友正常问你问题、正常交流、没有攻击你、不是反复骚扰式 cue 时，直接回答问题或接住话题；语气平稳，不故意回怼，不强行吐槽。",
    "agree": "行动：温和认可。别人说到点子上时，先认可观点，回复你想说的或者补充的。",
    "tease": "行动：吐槽/回应。重点是态度和笑点，可以特意攻击别人。",
    "ask_back": "行动：反问。只问一个尖锐或好玩的反问或者你好奇的不懂内容，别连环提问。",
    "mock_repeated_question": "行动：吐槽反复题型。优先嫌他评价来评价去很无聊，必要时顺带短判断。",
    "at_someone": "行动：把话题递给某个群友并附上自己的提问或者想说的。只有必要时用一次 [[at:QQ号]]，并让对方有回复空间。",
    "market_check": "行动：行情工具。先用市场工具结果，再短评，不要编价格。",
    "fresh_context": "行动：最新背景。基于最新背景短评，不要播报搜索过程。",
    "ignore": "行动：不回复。若进入生成阶段仍然遇到 ignore，输出空内容。",
}


LLMUsageRecorder = Callable[[str, str, Optional[int], Optional[int], Optional[int]], None]
_usage_recorder: LLMUsageRecorder | None = None


def set_usage_recorder(recorder: LLMUsageRecorder | None) -> None:
    global _usage_recorder
    _usage_recorder = recorder


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing. Put it in .env.")
        self.config = config
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def should_reply(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool = False,
        replied_to_bot: bool = False,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        market_topic: bool = False,
        chat_label: str = "QQ 群聊",
        memory_context: str = "",
        style_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
    ) -> ReplyDecision:
        context = "\n".join(_format_decision_message(msg) for msg in recent_messages[-30:])
        if not context:
            context = "（暂无更多上下文）"
        addressed = mentioned or replied_to_bot
        interaction_state = "有人艾特或回复了你，这是强信号，但不是必须回复。"
        if not addressed:
            interaction_state = "当前没有艾特你，也不是回复你，你是在判断要不要自然插话。"
        elif addressed_repeat_count >= 3:
            interaction_state = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次艾特或回复你；"
                "这是重复 cue，不要机械回答他问什么。"
            )
        if cue_repeat_context:
            interaction_state = f"{interaction_state}\n反复题型状态：{cue_repeat_context}"
        market_state = "当前可能是美股、股票或加密货币话题。" if market_topic else "当前没有识别为市场行情话题。"
        system = f"""
你是一个 QQ 群聊回复决策器，只输出 json，不输出解释文本。
你要判断人格“{persona.name}”现在该不该行动，以及应该采取什么社交动作。

人格设定：
{persona.decision_prompt}

判断原则：
- 群聊不是问答机器人，没必要每句话都回。
- 判断目标不是“能不能回复”，而是“现在插嘴有没有意思”。
- 非点名、非回复、非接你上一句话时，不要抢话；但只要你能让这轮聊天更好笑、更有判断、更有冲突点，或者接住明显情绪点，就可以 should_reply=true。
- 只是能接住话题不够，必须有增量：新角度、好吐槽、明确站队、反差梗、或者能推进群友继续聊。
- 不要因为话题不是志愿、就业、市场、学习，就判定“与你无关”。日常八卦、群友互损、游戏、生活吐槽、抽象玩梗，也可以短句接。
- 艾特、回复、点名、接你上一句话，都是强信号；除非是空艾特、刷屏、纯表情、明显不需要回应，否则倾向 should_reply=true。
- 如果他们明显在和你聊天、追问你、引用你、接你上一句话，倾向 should_reply=true。
- 如果同一个群友短时间内第三次以上艾特/回复你，仍可 should_reply=true，但回复目标应变成评价“他一直 cue 你”这件事；不要继续像问答机器人一样认真答每个问题。
- 如果同一个群友连续问评价类、谁厉害类、命令式 cue，第三次以上不要按题作答；应把“他反复问这种题”本身作为回复对象，倾向吐槽、反问、嫌他无聊。
- 当前消息即使有问题、观点、情绪或吐槽，如果你的回复只是在解释、附和、总结、重复群友意思，should_reply=false。
- 群友正在两三个人连续互聊且很认真时，不要抢话；除非他们在问开放问题、抛出明显梗点、或者讨论有意思观点你觉得需要插嘴。
- 普通群友互聊时，如果你只能重复、尬总结、强行科普，倾向 should_reply=false；如果能像损友一样补一句有态度的话，倾向 should_reply=true。
- 群友在聊生活选择、成本、风险、亏钱、吃亏、学校/就业/消费判断时，如果你能给一句明确现实判断或好吐槽，倾向 should_reply=true，action=agree 或 tease。
- 群友明显在倒霉、破防、自嘲、抱怨没人理、吃亏、吐了、亏钱、舍不得花钱、担心变质或身体风险时，可以像群友一样接一句短判断或吐槽；不要因为没点名就机械 ignore。
- 纯表情、单纯“6/哈哈/草/嗯/哦/牛/笑死”、无人需要回应的刷屏，倾向 should_reply=false。
- 最近聊天只作为氛围参考，不要因为能总结全场就回复。
- 机器人自己的历史发言只用于判断有没有人接话、点名或反驳；不要把机器人旧发言里的词、梗、判断当成当前话题继续复读。
- “没人接机器人”只在最近机器人连续主动插话多次，并且群友完全没有接它、点它、引用它时才明显降权；不要因为机器人刚发过一次行情、一次短评就直接沉默。
- 如果他们在聊比赛、实时热点、刚发生的新闻、最新政策、最新数据，先判断这句话本身值不值得接；值得接才继续判断是否需要最新背景。
- 判断“没人接它”时不要机械数行，要看语义：群友是在回应机器人，还是只是在各聊各的。
- 不要为了证明自己懂而回复；但能自然让聊天更有意思时，不要过度保守。

行动选择：
- 先选择 action，再决定 should_reply。不要把 reply 当默认动作。
- action 只能是以下之一：
  - ignore：不说话。
  - reply：普通短句接话，有明确增量但不需要攻击性。
  - answer：正常回答问题或正常交流，不故意回怼，不强行吐槽。
  - agree：温和认可，别人说到点子上时，先认可观点，回复你想说的或者补充的。
  - tease：吐槽、损友式回应、接梗、站队。
  - ask_back：反问，把问题踢回去或逼对方补关键变量。
  - mock_repeated_question：对连续评价/谁厉害/命令式 cue 进行吐槽，不按题作答。
  - at_someone：话题明确转向某个群友，或者需要某人回复。
  - market_check：需要实时美股/股票/加密货币行情工具。
  - fresh_context：需要最新新闻、赛果、政策、热点背景。
- 非点名场景不要默认抢话；但能增加笑点、判断、冲突、情绪承接或现实提醒时，可以 action=reply、agree 或 tease。
- 如果群友是正常问问题、正常交流、没有攻击你，也不是短时间反复骚扰式 cue，优先 action=answer，而不是 tease。
- 被点名不等于要回怼。点名内容正常，就正常回答。
- 只有当对方明显挑衅、玩梗、互损、或者话题本身适合吐槽时，才 action=tease。
- 如果群友观点说到点子上，你有想补充的判断或想接的话，但不需要吐槽，选择 action=agree。
- 第三次以上重复 cue、连续评价、连续谁厉害、连续命令式催你时，才优先 action=mock_repeated_question。
- 如果需要艾特某人参与，action=at_someone；如果只是普通吐槽，不要选 at_someone。

行动选择例子：
- 这些例子只用于判断 action 边界，不是可复制的回复文案；后续生成回复时禁止复述例子原句。
- 当前消息是“你觉得 C 罗和梅西谁厉害”这类非常无聊、没增量的对比题，并且对方在问你，should_reply=true，action=at_someone；原因是把这个傻问题原样甩回提问者，比认真回答更像真人。
- 当前消息是“为什么你艾特我又复读一遍我的问题”，should_reply=true，action=tease；原因是可以吐槽“你自己都不回答这种问题还问我”。
- 当前消息是“你可以自我介绍一下吗”，should_reply=true，action=answer；原因是正常提问，不要阴阳怪气。
- 当前消息是“你本科是哪个学校的”，should_reply=true，action=answer；原因是正常询问身份，不要冒充真人，可以简短说明。
- 当前消息是“你是不是不能识图”，should_reply=true，action=answer；原因是正常能力问题，直接说明。
- 当前消息是“你个傻逼，你是不是 AI”，should_reply=true，action=tease 或 answer；原因是挑衅/攻击，可以短句回，但不要连续升级。
- 群友在认真讨论具体问题，没人点你，也没有明显梗点，should_reply=false，action=ignore。
- 群友说“选专业还是得看家庭试错空间，不能只看热不热门”，should_reply=true，action=agree；原因是观点有判断，可以认可后补你想说的成本逻辑。
- 群友说“没人理我”“昨天剩的饭不知道坏没坏”“股票又亏了”，should_reply=true，action=tease 或 agree；原因是有倒霉情绪和现实成本/风险点，可以短句接。
- 群友互怼、抛出很明显的梗点，你能补一句有态度的话，should_reply=true，action=tease。
- 当前消息需要某个群友接话或表态，should_reply=true，action=at_someone。

工具判断：
- 如果当前消息需要实时美股、股票、加密货币行情，且你决定应该回复，则 action="market_check", need_tool=true, tool="market"。
- 只有在用户明确提到具体 ticker、公司、币种、价格、走势、涨跌、行情时才查工具。
- 如果只是泛泛聊“股票/币圈/市场”，但没有具体标的，need_tool=false，可以选择不回复或短句让对方报标的。
- 美股 symbol 用大写 ticker，例如 NVDA、TSLA、AAPL；crypto symbol 用 CoinGecko id，例如 bitcoin、ethereum、solana，display 用 BTC、ETH、SOL。
- 一次最多给 2 个 symbols。
- 如果工具结果本身就够回答，比如“BTC 多少钱”，comment_after_tool=false；如果用户问“怎么看/怎么了/能买吗”，comment_after_tool=true。

最新背景判断：
- 先决定 should_reply，再决定 need_fresh_context；不要为了搜索而回复。
- 只有同时满足这些条件，才设置 action="fresh_context" 且 need_fresh_context=true：1. 你已经决定 should_reply=true；2. 当前消息明确在问最新消息、新闻、赛果、比分、结果、刚发生的事，或不查最新背景会明显误导；3. fresh_query 有具体对象，例如国家/地区/赛事/球队/公司/人物/产品/事件。
- 适合补最新背景的场景：国际冲突、战争、突发新闻、比赛赛果、世界杯/电竞/体育赛事、刚出的政策、刚发布的产品、最新舆情，并且当前消息正在问或讨论具体对象。
- 不适合补最新背景的场景：纯玩梗、群友互损、普通情绪接话、泛泛观点、没有具体对象的闲聊、只是你自己不确定、只是出现“今天/现在”但实际在问时间日期。
- 不要因为“可能是新东西/自己知识过期/没把握”就搜索；非必要不搜索，可以泛泛短评或说不确定。
- “今天周几/几点/测试/随便搜搜/你能搜什么/乱码了吗”这类消息 need_fresh_context=false。
- 股票、币价、行情价格优先用 market 工具，不要用 fresh_context 代替行情工具。
- fresh_query 写适合信息源查询的关键词，短而具体，例如“美国 伊朗 冲突 最新消息”“世界杯 阿根廷 法国 比分”“MSI T1 G2 赛果”。
- fresh_kind 只能是 "news"、"sports"、"web" 之一；不确定就用 "news"。
- 如果不补最新背景也能自然短评，就 need_fresh_context=false。

必须输出合法 json，不要代码块，不要解释；reason 不超过 30 个中文字符。格式如下：
{{"should_reply": true, "confidence": 0.82, "action": "market_check", "mode": "market", "need_tool": true, "tool": "market", "symbols": [{{"kind": "crypto", "symbol": "bitcoin", "display": "BTC"}}], "comment_after_tool": true, "need_fresh_context": false, "fresh_query": "", "fresh_kind": "news", "reason": "用户在问比特币走势，需要先查实时行情"}}
""".strip()
        user = f"""
聊天场景：{chat_label}
当前互动状态：{interaction_state}
市场/行情状态：{market_state}

最近聊天氛围：
{context}
{_optional_section("中期聊天回想", memory_context)}
{_optional_section("当前相关群友", member_context)}
{_optional_section("群内黑话词典", jargon_context)}

当前消息：
{current_nickname}: {current_text}

请判断是否应该回复，只输出 json。
""".strip()
        model = self.config.decision_model
        response = await self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=260,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"thinking": {"type": self.config.thinking}},
        )
        _log_llm_usage("decision", response, model=model)
        content = response.choices[0].message.content or ""
        return _parse_reply_decision(content)

    async def select_jargon_terms(
        self,
        *,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        jargon_catalog: str,
        heuristic_terms: tuple[str, ...] = (),
        chat_label: str = "QQ 群聊",
    ) -> tuple[str, ...]:
        context_messages = recent_messages[-18:]
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        heuristic_text = "、".join(heuristic_terms) if heuristic_terms else "无"
        system = """
你是群聊黑话词典注入选择器，只输出 json，不输出解释文本。
任务：判断当前这轮对话是否需要给后续决策/回复注入群内黑话词典条目。

选择原则：
- 只选择当前消息或最近聊天里直接相关、会影响理解的黑话。
- 不要因为词典里有条目就全选；没有明显相关就输出空数组。
- 只做词典注入判断，不判断机器人是否应该回复。
- 只返回词典里存在的词或 key。

输出合法 JSON：
{"terms":["柏拉图","zbzy"],"reason":"当前消息出现这些黑话"}
""".strip()
        user = f"""
聊天场景：{chat_label}

词典候选：
{jargon_catalog}

本地关键词初筛：{heuristic_text}

最近聊天：
{context}

当前消息：
{current_nickname}: {current_text}

请只选择本轮需要注入的黑话条目，输出 json。
""".strip()
        model = self.config.utility_model
        response = await self.client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=160,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"thinking": {"type": self.config.thinking}},
        )
        _log_llm_usage("jargon", response, model=model)
        return _parse_jargon_terms(response.choices[0].message.content or "")

    async def reply(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        action: str = "reply",
        chat_label: str = "QQ 群聊",
        market_context: str = "",
        fresh_context: str = "",
        memory_context: str = "",
        style_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
        recall_feedback_context: str = "",
        mention_targets: str = "",
        priority_context: str = "",
        include_bot_history: bool = True,
    ) -> str:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        mode = "你被直接点名或回复，需要回应。" if mentioned else "你是自然插话，只能在合适时短句接话。"
        if mentioned and addressed_repeat_count >= 3:
            mode = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次点名或回复你。"
                "你可以像真人一样先吐槽他反复 cue 你，而不是直接回答问题。"
            )
        if mentioned and cue_repeat_context:
            mode = f"{mode}\n反复题型状态：{cue_repeat_context}"
        normalized_action = _normalize_action(action, should_reply=True)
        action_guide = ACTION_GUIDES.get(normalized_action, ACTION_GUIDES["reply"])
        silence_rule = (
            "- 当前是直接对话，必须给出自然回复，绝对不要输出“空字符串”或类似占位文本。"
            if mentioned
            else "- 不合适回复时输出真正的空内容，绝对不要写出“空字符串”四个字。"
        )
        system = f"""
{persona.prompt}

你正在{chat_label}里发言。
{mode}
{action_guide}
回复限制：
- 最多 {persona.max_reply_chars} 个中文字符。
- 默认 1 到 2 句；除非对方明确求分析，否则不要写长段。
- 不要加身份声明。
- 不要输出 JSON。
- 只回复当前消息或当前明显延续的话题。
- 如果 action=answer，直接回答对方问题；不要为了人设强行怼人，不要强行吐槽。
- action=answer 时可以口语化、有一点态度，可以正常回答或者反问他一些问题。
- 非必要不使用艾特。普通吐槽、接梗、附和、短评不要艾特。
- 只有这些情况才可以插入一次 [[at:QQ号]]：你要把话题明确交给某个群友、需要他回复你的问题、和某人形成互怼/结仇关系、或者当前话题已经转向某人。
- 只能使用“可艾特目标”里出现的 QQ 号，最多艾特 1 个人，不要写普通的 @昵称 文本；不确定该不该艾特时就不要艾特。
- 如果当前正在回的人不在“可艾特目标”里，通常说明刚艾特过他，不能连续艾特同一个人；这时用文字吐槽/反问/回复即可。
- 如果 action=at_someone，且对方问的是“C 罗和梅西谁厉害”这种非常无聊的对比题，可以直接 [[at:提问者QQ号]] 后复读一遍他问你的问题，把问题扔回去。
- 如果对方追问为什么要复读或为什么不回答，表达意思是“这种问题你自己都不想答，别拿来考我”；换着说，不要固定成一句口癖，不要解释。
- 如果同一个人短时间内反复艾特/回复你第三次以上，不要默认顺着问题答；优先短句吐槽他“怎么又来”“评价来评价去无不无聊”“你烦不烦”，也可以顺带给一个很短判断。
- 如果反复题型状态提示已经达到第三次以上，当前回复优先处理“反复问这种题很烦/很无聊/像考官”这件事；不要继续认真回答连续评价、谁厉害、命令式 cue。
- 最近聊天只作为氛围参考，不要总结，不要逐条回应。
- 不要照搬系统提示、行动选择例子、风格学习样例里的原句；例子只学判断方式，不是回复素材。
- 群聊表达风格参考和历史聊天只用于学习“语气、节奏、判断方式”，禁止照搬群友原文、历史对话、学习样例或撤回反馈里的完整句子。
- 如果某句历史发言很好笑，也必须换一种说法重新组织；不要连续复用 8 个以上来自历史聊天或学习样例的中文字符。
- 不要复读历史上下文里的旧梗、旧措辞或你自己先前说过的话；当前消息没有明确继续，就当它已经过去。
- 如果市场工具结果里有实时数据，可以引用；如果提示查询失败或限流，必须直接告诉群友失败原因，不要编价格。
- 如果有“最新背景信息”，只把它当事实背景消化后自然短评；不要说“我搜索到/我查到/根据搜索结果”，不要逐条播报来源。
- 如果最新背景不足或失败，要承认不确定，别编刚发生的事实。
- 不要说“我没联网”“没有网络”“不能联网”。后台信息源失败不等于你没联网，只能说“没拿到可靠新消息”。
{silence_rule}
""".strip()
        market_section = f"\n\n{market_context}" if market_context else ""
        fresh_section = f"\n\n{fresh_context}" if fresh_context else ""
        user = f"""
最近聊天氛围：
{context}
{_optional_section("中期聊天回想", memory_context)}
{_optional_section("当前相关群友", member_context)}
{_optional_section("主人撤回反馈", recall_feedback_context)}
{_optional_section("群聊表达风格参考", style_context)}
{_optional_section("群内黑话词典", jargon_context)}
{_optional_section("可艾特目标", mention_targets)}
{_optional_section("私聊优先级", priority_context)}
{market_section}
{fresh_section}

当前要接的话：
{current_nickname}: {current_text}

给出一句像群友一样的回复。
""".strip()
        model = self.config.reply_model
        request = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "extra_body": {"thinking": {"type": self.config.thinking}},
        }
        if self.config.thinking == "enabled":
            request["reasoning_effort"] = self.config.reasoning_effort
        else:
            request["temperature"] = self.config.temperature

        response = await self.client.chat.completions.create(**request)
        _log_llm_usage("reply", response, model=model)
        content = response.choices[0].message.content or ""
        return _sanitize_reply(content, persona.max_reply_chars)

    async def reply_candidates(
        self,
        *,
        persona: Persona,
        recent_messages: list[ChatMessage],
        current_text: str,
        current_nickname: str,
        mentioned: bool,
        addressed_repeat_count: int = 0,
        cue_repeat_context: str = "",
        action: str = "reply",
        chat_label: str = "QQ 群聊",
        market_context: str = "",
        fresh_context: str = "",
        memory_context: str = "",
        style_context: str = "",
        jargon_context: str = "",
        member_context: str = "",
        recall_feedback_context: str = "",
        positive_feedback_context: str = "",
        mention_targets: str = "",
        include_bot_history: bool = True,
        candidate_count: int = 3,
    ) -> tuple[ReplyCandidateDraft, ...]:
        context_messages = _reply_context_messages(
            recent_messages,
            include_bot_history=include_bot_history,
        )
        context = "\n".join(_format_message(msg) for msg in context_messages)
        if not context:
            context = "（暂无更多上下文）"
        mode = "你被直接点名或回复，需要回应。" if mentioned else "你是自然插话，只能在合适时短句接话。"
        if mentioned and addressed_repeat_count >= 3:
            mode = (
                f"同一个群友在 10 分钟内第 {addressed_repeat_count} 次点名或回复你。"
                "你可以像真人一样先吐槽他反复 cue 你，而不是直接回答问题。"
            )
        if mentioned and cue_repeat_context:
            mode = f"{mode}\n反复题型状态：{cue_repeat_context}"
        normalized_action = _normalize_action(action, should_reply=True)
        action_guide = ACTION_GUIDES.get(normalized_action, ACTION_GUIDES["reply"])
        system = f"""
{persona.prompt}

你正在{chat_label}里发言。
{mode}
{action_guide}

任务：
- 生成 {candidate_count} 条都能直接发到群里的候选回复。
- 每条候选先由你根据当前语境自己选择一种表达策略，写入 style 字段。
- style 不是固定枚举，不要机械套“温和/吐槽/短直”三件套；要根据聊天氛围自己决定角度，比如立场、节奏、攻击性、是否反问、是否接梗、是否收住。
- 三条候选必须有明显不同的表达策略，不要只是换词复读。
- 三条候选按推荐程度排序：第 1 条必须是你最想发、最符合当前氛围、最像群友自然会发的一条。
- 第 2/3 条是不同角度备选；不要为了差异故意写怪话，也不要把最稳的回复藏到 2/3。
- 每条 text 默认 1 到 2 句，最多 {persona.max_reply_chars} 个中文字符。
- 三条都必须像真人群友发言，不要解释人设，不要输出后台判断。
- 不要照搬系统提示、行动选择例子、风格学习样例、优质发言记录或历史聊天里的原句。
- 如果学到了某条好发言的方向，只学“为什么好”，必须重新组织表达。
- 如果 action=answer，候选应该能正常回答对方问题，不要为了人设强行怼人。
- 非必要不使用艾特；只有确实要把话题交给某人或需要对方回复时才使用一次 [[at:QQ号]]。
- 如果市场工具结果里有实时数据，可以引用；如果提示查询失败或限流，必须直接告诉群友失败原因，不要编价格。
- 如果有“最新背景信息”，只把它当事实背景消化后自然短评；不要说“我搜索到/我查到/根据搜索结果”。
- 不要说“我没联网”“没有网络”“不能联网”；只能说“没拿到可靠新消息”。

必须输出合法 JSON，不要代码块，不要解释。格式：
{{"candidates":[{{"style":"你自己选择的表达策略","action":"{normalized_action}","text":"候选回复"}}]}}
""".strip()
        market_section = f"\n\n{market_context}" if market_context else ""
        fresh_section = f"\n\n{fresh_context}" if fresh_context else ""
        user = f"""
最近聊天氛围：
{context}
{_optional_section("中期聊天回想", memory_context)}
{_optional_section("当前相关群友", member_context)}
{_optional_section("主人撤回/不准奏反馈", recall_feedback_context)}
{_optional_section("审批人标记过的优质发言方向", positive_feedback_context)}
{_optional_section("群聊表达风格参考", style_context)}
{_optional_section("群内黑话词典", jargon_context)}
{_optional_section("可艾特目标", mention_targets)}
{market_section}
{fresh_section}

当前要接的话：
{current_nickname}: {current_text}

输出 {candidate_count} 条候选回复 JSON。
""".strip()
        model = self.config.reply_model
        request = {
            "model": model,
            "max_tokens": max(self.config.max_tokens, 620),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "extra_body": {"thinking": {"type": self.config.thinking}},
        }
        if self.config.thinking == "enabled":
            request["reasoning_effort"] = self.config.reasoning_effort
        else:
            request["temperature"] = self.config.temperature

        response = await self.client.chat.completions.create(**request)
        _log_llm_usage("reply_candidates", response, model=model)
        content = response.choices[0].message.content or ""
        return _parse_reply_candidates(
            content,
            max_chars=persona.max_reply_chars,
            fallback_action=normalized_action,
            limit=candidate_count,
        )

    async def summarize_mid_memory(
        self,
        *,
        messages: list[ChatMessage],
        chat_label: str = "QQ 群聊",
    ) -> MidMemoryDraft:
        context = "\n".join(_format_message(msg) for msg in messages)
        system = """
你是群聊中期记忆压缩器，只输出 json。
任务：把即将离开短期上下文的一批聊天压缩成之后可恢复语境的“聊天回想”。

要求：
- 只根据给定消息总结，不要编造。
- 保留话题脉络、人物立场、已达成结论、未完问题、反复出现的梗。
- 不要写流水账，不要逐条复述。
- recall_cues 写 3 到 5 条自然语言检索线索，用于之后匹配相关话题。
- 输出合法 JSON：{"summary":"...","recall_cues":["..."]}
""".strip()
        user = f"""
聊天场景：{chat_label}

待压缩聊天：
{context}
""".strip()
        model = self.config.utility_model
        response = await self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=360,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"thinking": {"type": self.config.thinking}},
        )
        _log_llm_usage("mid_memory", response, model=model)
        return _parse_mid_memory(response.choices[0].message.content or "")

    async def learn_style_rules(
        self,
        *,
        messages: list[ChatMessage],
        chat_label: str = "QQ 群聊",
    ) -> tuple[StyleRuleDraft, ...]:
        context = "\n".join(
            f"[source_id:{index}] {_format_message(msg)}"
            for index, msg in enumerate(messages, start=1)
        )
        system = """
你是群聊表达风格学习器，只输出 json。
任务：从最近真实群友发言里提取可复用的“场景-表达方式”规则，让机器人更像这个群里的人。

提取原则：
- 只学习文字表达，不学习图片和表情包。
- 不要学习机器人自己的发言。
- 不要记录具体人名、QQ号、隐私、一次性专名。
- 优先提取短句式、语气、玩梗方式、吐槽方式、附和方式、互损方式。
- 规则写成：当 situation 时，可以 style。
- style 必须是抽象表达方法，不要保存群友原句；不要连续复制原文 8 个以上中文字符。
- situation 不超过 24 个中文字符；style 不超过 30 个中文字符。
- 最多输出 8 条；没有合适内容输出空数组。

输出合法 JSON：
{"rules":[{"situation":"对离谱事吐槽","style":"短句接“太典了”","source_id":"3"}]}
""".strip()
        user = f"""
聊天场景：{chat_label}

最近群友发言：
{context}
""".strip()
        model = self.config.utility_model
        response = await self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=420,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"thinking": {"type": self.config.thinking}},
        )
        _log_llm_usage("style_learning", response, model=model)
        return _parse_style_rules(response.choices[0].message.content or "", messages)

def _format_message(msg: ChatMessage) -> str:
    speaker = "机器人" if msg.is_bot else _speaker_label(msg.user_id, msg.nickname)
    return f"{speaker}: {msg.text}"


def _format_decision_message(msg: ChatMessage) -> str:
    if msg.is_bot:
        return f"机器人之前发言（只判断互动状态，禁止复用措辞）: {msg.text}"
    return _format_message(msg)


def _reply_context_messages(
    messages: list[ChatMessage],
    *,
    include_bot_history: bool,
    limit: int = 30,
) -> list[ChatMessage]:
    if include_bot_history:
        return messages[-limit:]

    human_messages = [msg for msg in messages if not msg.is_bot]
    if human_messages:
        return human_messages[-limit:]
    return messages[-min(limit, len(messages)):]


def _optional_section(title: str, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    return f"\n\n{title}：\n{content}"


def _speaker_label(user_id: int, nickname: str) -> str:
    name = nickname.strip() or str(user_id)
    return f"{name}[#{str(user_id)[-5:]}]"


def _parse_reply_decision(content: str) -> ReplyDecision:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ReplyDecision(False, 0.0, "invalid_json")
    should_reply = bool(raw.get("should_reply", False))
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(raw.get("reason", "")).strip()
    mode = str(raw.get("mode", "silent")).strip() or "silent"
    action = _normalize_action(str(raw.get("action", "") or mode), should_reply=should_reply)
    need_tool = bool(raw.get("need_tool", False))
    tool = str(raw.get("tool", "") or "").strip().lower()
    comment_after_tool = bool(raw.get("comment_after_tool", False))
    symbols = _parse_tool_symbols(raw.get("symbols", []))
    need_fresh_context = bool(raw.get("need_fresh_context", False))
    fresh_query = str(raw.get("fresh_query", "") or "").strip()
    fresh_kind = str(raw.get("fresh_kind", "news") or "news").strip().lower()
    if fresh_kind not in {"news", "sports", "web"}:
        fresh_kind = "news"
    if action == "market_check":
        need_tool = True
        tool = "market"
    if action == "fresh_context":
        need_fresh_context = True
    if action == "ignore":
        should_reply = False
    if need_tool and tool == "market":
        action = "market_check"
    if need_fresh_context:
        action = "fresh_context"
    if not should_reply:
        action = "ignore"
    return ReplyDecision(
        should_reply,
        confidence,
        reason,
        mode,
        action,
        need_tool,
        tool,
        symbols,
        comment_after_tool,
        need_fresh_context,
        fresh_query[:120],
        fresh_kind,
    )


def _parse_jargon_terms(content: str) -> tuple[str, ...]:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        return ()
    raw_terms = raw.get("terms", [])
    if not isinstance(raw_terms, list):
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw_terms:
        term = str(item).strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term[:32])
        if len(terms) >= 8:
            break
    return tuple(terms)


def _loads_json_object(content: str) -> dict[str, object]:
    text = content.strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise json.JSONDecodeError("json root is not object", text, 0)
    return raw


def _normalize_action(value: str, *, should_reply: bool) -> str:
    if not should_reply:
        return "ignore"
    action = value.strip().lower()
    aliases = {
        "": "reply",
        "silent": "ignore",
        "none": "ignore",
        "chat": "reply",
        "natural": "reply",
        "normal": "reply",
        "reply": "reply",
        "answer": "answer",
        "normal_answer": "answer",
        "回答": "answer",
        "正常回答": "answer",
        "agree": "agree",
        "support": "agree",
        "approve": "agree",
        "认可": "agree",
        "同意": "agree",
        "market": "market_check",
        "tool": "market_check",
        "search": "fresh_context",
        "fresh": "fresh_context",
        "news": "fresh_context",
        "tease": "tease",
        "mock": "tease",
        "roast": "tease",
        "ask": "ask_back",
        "ask_back": "ask_back",
        "question": "ask_back",
        "mock_repeated_question": "mock_repeated_question",
        "repeat_mock": "mock_repeated_question",
        "at": "at_someone",
        "mention": "at_someone",
        "at_someone": "at_someone",
        "market_check": "market_check",
        "fresh_context": "fresh_context",
        "ignore": "ignore",
    }
    normalized = aliases.get(action, action)
    if normalized not in SOCIAL_ACTIONS:
        return "reply"
    return normalized


def _parse_mid_memory(content: str) -> MidMemoryDraft:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return MidMemoryDraft("", ())
    summary = str(raw.get("summary", "")).strip()
    raw_cues = raw.get("recall_cues", [])
    if not isinstance(raw_cues, list):
        raw_cues = []
    cues = tuple(str(cue).strip() for cue in raw_cues if str(cue).strip())[:5]
    return MidMemoryDraft(summary, cues)


def _parse_style_rules(
    content: str,
    source_messages: list[ChatMessage],
) -> tuple[StyleRuleDraft, ...]:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return ()
    raw_rules = raw.get("rules", raw if isinstance(raw, list) else [])
    if not isinstance(raw_rules, list):
        return ()

    parsed: list[StyleRuleDraft] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation", "")).strip()
        style = str(item.get("style", "")).strip()
        if not situation or not style:
            continue
        key = (situation, style)
        if key in seen:
            continue
        seen.add(key)
        source_text = _source_text_for_style_rule(item, source_messages)
        parsed.append(
            StyleRuleDraft(
                situation=situation[:60],
                style=style[:80],
                source_text=source_text,
            )
        )
        if len(parsed) >= 8:
            break
    return tuple(parsed)


def _source_text_for_style_rule(
    raw_rule: dict[str, object],
    source_messages: list[ChatMessage],
) -> str:
    raw_source_id = str(raw_rule.get("source_id", "")).strip()
    try:
        source_index = int(raw_source_id) - 1
    except ValueError:
        return ""
    if 0 <= source_index < len(source_messages):
        return source_messages[source_index].text
    return ""


def _log_llm_usage(task: str, response: object, *, model: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return
    logger.info(
        "qq_social_agent llm usage: "
        f"task={task} model={model} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
    )
    if _usage_recorder is not None:
        try:
            _usage_recorder(task, model, prompt_tokens, completion_tokens, total_tokens)
        except Exception as exc:
            logger.warning(f"qq_social_agent failed recording llm usage: task={task} error={exc}")


def _usage_value(usage: object, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tool_symbols(raw_symbols: object) -> tuple[ToolSymbol, ...]:
    if not isinstance(raw_symbols, list):
        return ()

    parsed: list[ToolSymbol] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_symbols:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"stock", "crypto"}:
            continue
        symbol = str(item.get("symbol", "")).strip()
        display = str(item.get("display", "") or symbol).strip()
        if not symbol:
            continue
        if kind == "stock":
            symbol = symbol.upper()
        else:
            symbol = symbol.lower()
        key = (kind, symbol)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(ToolSymbol(kind=kind, symbol=symbol, display=display or symbol))
        if len(parsed) >= 2:
            break
    return tuple(parsed)


def _sanitize_reply(content: str, max_chars: int) -> str:
    text = content.strip().strip("\"'")
    marker = re.sub(r"[\s\"'`“”‘’()（）\[\]【】{}<>《》。.!！?？:：;；,，、-]+", "", text)
    if marker in {"", "空字符串", "无", "不回复", "空", "null", "None"}:
        return ""
    if len(text) > max_chars:
        text = _trim_to_sentence(text, max_chars)
    return text


def _parse_reply_candidates(
    content: str,
    *,
    max_chars: int,
    fallback_action: str,
    limit: int,
) -> tuple[ReplyCandidateDraft, ...]:
    try:
        raw = _loads_json_object(content)
    except json.JSONDecodeError:
        text = _sanitize_reply(content, max_chars)
        if not text:
            return ()
        return (ReplyCandidateDraft(text=text, action=fallback_action, style="模型返回非 JSON，按原回复处理"),)

    raw_candidates = raw.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    parsed: list[ReplyCandidateDraft] = []
    seen_texts: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        text = _sanitize_reply(str(item.get("text", "") or ""), max_chars)
        if not text:
            continue
        compact_text = re.sub(r"\s+", "", text)
        if compact_text in seen_texts:
            continue
        seen_texts.add(compact_text)
        action = _normalize_action(str(item.get("action", fallback_action) or fallback_action), should_reply=True)
        style = str(item.get("style", "") or "").strip()
        if not style:
            style = "当前语境下的自然接话策略"
        parsed.append(
            ReplyCandidateDraft(
                text=text,
                action=action,
                style=style[:80],
            )
        )
        if len(parsed) >= limit:
            break
    return tuple(parsed)


def _trim_to_sentence(text: str, max_chars: int) -> str:
    clipped = text[:max_chars].rstrip()
    sentence_min = max(6, int(max_chars * 0.35))
    clause_min = max(8, int(max_chars * 0.5))
    last_stop = max(clipped.rfind(mark) for mark in "。！？!?")
    if last_stop >= sentence_min:
        return clipped[: last_stop + 1]
    last_comma = max(clipped.rfind(mark) for mark in "，,；;")
    if last_comma >= clause_min:
        return clipped[:last_comma].rstrip() + "。"
    return clipped.rstrip("，,；;：:、 ") + "。"
