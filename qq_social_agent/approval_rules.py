from __future__ import annotations

import re


APPROVAL_REJECT_REASON_RE = re.compile(r"^不准奏(?P<index>[1-3ABCabc])?原因\s*[:：]\s*(?P<reason>.+)$", re.DOTALL)
APPROVAL_CHOICE_RE = re.compile(r"^([1-3ABCabc])([!！])?$")
JARGON_ADD_RE = re.compile(
    r"^/黑话\s*[:：]?\s*(?P<term>.+?)\s*(?:指代|意思是|=|->|:|：)\s*[:：]?\s*(?P<meaning>.+)$"
)
JARGON_DELETE_RE = re.compile(r"^/删黑话\s*[:：]?\s*(?P<term>.+)$")
JARGON_LIST_RE = re.compile(r"^/黑话列表\s*$")
APPROVAL_HELP_COMMANDS = {"审批规则", "规则", "帮助", "S", "s"}
APPROVAL_DETAIL_COMMANDS = {"审批规则详情", "详细规则", "规则详情", "详细解释", "R", "r"}
TOKEN_REPORT_COMMAND_ALIASES = {"token用量", "tokens", "token", "usage", "用量", "消耗", "费用"}

APPROVAL_RULES_MESSAGE = """张风雪审批单：
A/1 发候选1；B/2 发候选2；C/3 发候选3。
D/X/取消 不发；T 打开工具单；R 展开详细目录。
主人：A!/1! 标优；不准奏原因：xxx；不准奏B原因：xxx。
开关：开启/关闭 群聊决策；开启审查/关闭审查 人工审查。
基础审批人只能用 A/B/C/D/X/1/2/3/取消。"""

BOT_TOOL_INDEX_MESSAGE = """张风雪 bot工具目录：
回复字母展开栏目；回复字母+数字直接执行常用查询。

A 查看/调试：A1拦截 A2模型 A3审批人 A4私聊白名单 A5 token。
B 学习/画像：B1记忆 B2风格 B3群友画像。
C 模型：C1状态；切回复模型/切画像模型 后面自己填模型名。
D 开关：D1审查状态 D2开启审查 D3关闭审查 D4开启群聊 D5关闭群聊。
E 审批说明；F 黑话；G 私聊；H 审批人；P prompt；Z 全部。
原中文命令仍可用：bot工具 查看、bot工具 学习、bot工具 模型、bot工具 私聊、bot工具 prompt、群友画像 20、拦截 20。"""

BOT_TOOL_SHORTCUT_COMMANDS = {
    "a1": "拦截 20",
    "a2": "模型状态",
    "a3": "审批人列表",
    "a4": "私聊白名单",
    "a5": "token用量",
    "b1": "记忆 8",
    "b2": "风格 8",
    "b3": "群友画像 20",
    "c1": "模型状态",
    "d1": "审查状态",
    "d2": "开启审查",
    "d3": "关闭审查",
    "d4": "开启",
    "d5": "关闭",
    "f1": "/黑话列表",
    "g1": "私聊白名单",
    "h1": "审批人列表",
}

BOT_TOOL_SECTION_MESSAGES = {
    "approval": """bot工具 审批：
1. 回 A/B/C 或 1/2/3：发送对应候选。
2. 回 D/X/取消：不发。
3. 主人可回 A!/B!/C! 或 1!/2!/3!：发送并标记优质。
4. 主人可回 不准奏原因：xxx；不准奏B原因：xxx；不准奏2原因：xxx：取消并记录指定候选的问题。
5. 不准奏原因只保存主人给的原始评价，不再用 LLM 总结改写。
6. 两个审批人谁先回听谁的；另一个人的旧数字会被短时间忽略，避免串到下一条。
7. 基础审批人只有 A/B/C/D/X/1/2/3/取消 权限，不能改模型、黑话、白名单、开关。""",
    "view": """bot工具 查看：
快捷：A1拦截20；A2模型状态；A3审批人列表；A4私聊白名单；A5 token。
1. 拦截 20 或 /bot blocked 20：查看最近后端拦截、频率门、LLM 不发原因。
2. 模型状态：查看当前各 LLM 流程模型、fallback、API key 来源、可切换模型清单。
3. 审批人列表：查看当前主人和基础审批人。
4. 私聊白名单：查看当前私人聊天白名单。
5. 记忆 10：查看最近 10 条中期聊天回想。
6. 风格 10：查看最近 10 条风格学习规则。
7. 群友画像 20：查看最近活跃群友的长期印象和兴趣摘要。
8. token用量：目前已关闭统计；查询会提示关闭原因。""",
    "jargon": """bot工具 黑话：
快捷：F1 查看黑话列表。
1. /黑话：咱妈 指代：中国。
2. /黑话：达斯 指代：打死。
3. /黑话列表：查看自定义黑话。
4. /删黑话：咱妈：删除指定黑话。
5. 只有主人/工具管理员能改黑话。""",
    "switch": """bot工具 开关：
快捷：D1审查状态；D2开启审查；D3关闭审查；D4开启群聊；D5关闭群聊。
1. 开启：恢复群聊进入决策。
2. 关闭：暂停群聊进入决策，并清空待审候选。
3. 开启审查：机器人想发群时先发审批单。
4. 关闭审查：机器人想发群时直接发送第 1 候选，不再等 1/2/3。
5. 审查状态：查看当前是否需要人工审批。
6. 每次开启/关闭会重新发送简版审批规则。""",
    "approver": """bot工具 审批人：
快捷：H1 审批人列表。
1. 审批人列表：查看当前主人和基础审批人。
2. 加审批 123456：添加基础审批人。
3. 删审批 123456：删除基础审批人。
4. 只有主人能增删审批人；基础审批人不能用工具命令。""",
    "private": """bot工具 私聊：
快捷：G1 私聊白名单。
1. 私聊白名单：查看当前允许普通私聊聊天的 QQ。
2. 加私聊 123456：添加运行时私聊白名单。
3. 删私聊 123456：删除运行时私聊白名单。
4. config.yaml 里的 access_control.allowed_private_users 是固定白名单。
5. 1535071184 是命令专用号，只处理审批/工具命令，不走普通私聊生成。
6. 工具管理员/调试号可隐式普通私聊，不需要额外加入。
7. 白名单只影响普通私聊聊天，不授予工具权限。
8. 测试号 2776760548 可用 强服从 / 关闭强服从 / 强服从状态。
9. 测试号 2776760548 可用 强服从：具体内容，单次注入最高优先级调试提示。""",
    "model": """bot工具 模型：
快捷：C1 模型状态。切模型命令仍需自己填模型名。
1. 模型状态：展开可切换部分、当前模型、fallback、API key 来源、可切换模型清单。
2. 可切换部分：决策模型、回复模型、黑话模型、记忆模型、风格模型、画像模型。
3. 决策模型：群聊是否插嘴、action、是否需要联网搜索。
4. 回复模型：私聊回复、群聊审批三候选。
5. 黑话模型：黑话词典注入选择。
6. 记忆模型：中期聊天回想压缩。
7. 风格模型：群聊表达风格学习。
8. 画像模型：群友长期画像摘要。
9. 切回复模型 siliconflow/MiniMaxAI/MiniMax-M2.5。
10. 切风格模型 siliconflow/MiniMaxAI/MiniMax-M2.5。
11. 切画像模型 siliconflow/MiniMaxAI/MiniMax-M2.5。
12. 切决策模型 siliconflow/Qwen/Qwen3.5-35B-A3B。
13. 切工具模型 deepseek/deepseek-v4-flash：批量切黑话/记忆/风格/画像。
14. 清模型覆盖：恢复 config.yaml 默认模型。""",
    "learning": """bot工具 学习：
快捷：B1记忆8；B2风格8；B3群友画像20。
1. 记忆 8：查看最近 8 条中期聊天回想。
2. 近期记忆 20：查看最近 20 条中期聊天回想，最多 30 条。
3. 风格 8：查看最近 8 条风格学习规则。
4. 风格学习 20：查看最近 20 条风格学习规则，最多 30 条。
5. 群友画像 20：查看最近活跃群友的单独印象、兴趣、关键词和代表性原话。
6. 记忆来源：短期上下文外的旧聊天，由 memory 模型压缩成回想。
7. 风格来源：最近群友原文，由 style 模型抽取“场景 -> 表达策略”。
8. 群友画像来源：后端实时统计每个人原文；每天最多一次用 memory 模型给活跃群友补摘要。
9. 生成回复时只注入相关片段，不会把所有记忆、风格和画像全塞进 prompt。
10. 学习规则只当策略参考，prompt 已要求禁止照搬原文。""",
    "prompt": """bot工具 prompt：
1. 集中 prompt 文件：/Users/ywbw/qq-social-agent/prompts/zhangfengxue.yaml。
2. persona：人格、自我认知、说话方式。
3. flows.decision：群聊是否插嘴、action、是否需要最新背景。
4. flows.jargon_select：黑话词典注入选择。
5. flows.reply：私聊/兜底单条回复。
6. flows.reply_candidates：群聊审批三候选生成。
7. flows.mid_memory：中期记忆压缩。
8. flows.style_learning：群聊表达风格学习。
9. action_guides：action 对应的生成方向。""",
}

BOT_TOOL_FULL_MESSAGE = "\n\n".join([BOT_TOOL_INDEX_MESSAGE, *BOT_TOOL_SECTION_MESSAGES.values()])
APPROVAL_RULES_DETAIL_MESSAGE = BOT_TOOL_INDEX_MESSAGE
