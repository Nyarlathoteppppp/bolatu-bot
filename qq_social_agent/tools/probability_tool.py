from __future__ import annotations

import json
from typing import Any
from nonebot import logger

from ..deepseek_client import DeepSeekClient
from ..jev_client import JevClient
from ..pipeline_types import ToolKind, ToolRequest, ToolResult


class JevProbabilityTool:
    """Use Jev for an explicitly requested subjective estimate, not actuarial odds."""

    def __init__(
        self,
        jev_client: JevClient,
        deepseek_client: DeepSeekClient,
    ) -> None:
        self.jev_client = jev_client
        self.deepseek_client = deepseek_client

    async def execute(self, request: ToolRequest) -> ToolResult:
        query = request.query.strip()
        if not query:
            return ToolResult(ToolKind.PROBABILITY, "empty_query", error="query_required")

        recent_context = str(request.arguments.get("context", "") or query)

        # 1. Use DeepSeek to refine query into typed Jev state and instructions
        try:
            prompt = (
                "群友向张风雪询问某件事情发生的可能性、几率或概率。\n"
                "请结合上下文，将该问题提炼为给 Jev (TypeSafe System One) 评估的标准输入。\n"
                "背景只能使用用户或上下文已提供的证据，不得补造录取率、统计样本或个人经历。\n"
                "必须输出合法 JSON，包含：\n"
                "- \"state\": 包含背景和具体情况的客观描述（去除口语命令词如'风雪你觉得'、'算算概率'等）\n"
                "- \"instructions\": 供 noul (yes/no 概率) 评估的问题陈述（明确陈述该事件为真或将要发生）\n\n"
                f"当前上下文：\n{recent_context}\n\n"
                "只输出合法 JSON：\n"
                "{\"state\": \"...\", \"instructions\": \"...\"}"
            )
            # Use utility route to parse
            resp = await self.deepseek_client._chat_completion(
                task="utility",
                route_name="utility",
                request={
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            content = getattr(resp.choices[0].message, "content", "") or ""
            extracted = json.loads(content)
            state = str(extracted.get("state", query)).strip()
            instructions = str(extracted.get("instructions", query)).strip()
        except Exception as exc:
            logger.warning(f"qq_social_agent jev probability prompt parsing failed: {exc}")
            state = recent_context
            instructions = query

        # 2. Call Jev to evaluate probability
        try:
            prob = await self.jev_client.evaluate_probability(
                state=state,
                instructions=instructions,
            )
        except Exception as exc:
            logger.warning(f"qq_social_agent jev probability evaluation failed: {exc}")
            return ToolResult(
                ToolKind.PROBABILITY,
                "error",
                error=str(exc)[:200],
            )

        pct = prob * 100.0
        context = (
            f"【概率评估】命题：{instructions}\n"
            f"模型主观估计：约 {pct:.0f}%（noul={prob:.4f}）。\n"
            "这是基于给定信息的粗略判断，不是该现实事件经过统计校准的发生率。"
            "用户明确要求数值时才给粗略数字并说明关键依据；不要称为测算结果，不编造样本或计算过程。"
        )
        return ToolResult(
            ToolKind.PROBABILITY,
            "ok",
            context=context,
            evidence=f"jev_prob={prob:.4f}",
            metadata={"probability": prob, "instructions": instructions, "state": state},
        )
