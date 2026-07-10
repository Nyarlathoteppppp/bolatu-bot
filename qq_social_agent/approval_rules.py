from __future__ import annotations

import re


APPROVAL_REJECT_REASON_RE = re.compile(r"^不准奏(?P<index>[1-3])?原因\s*[:：]\s*(?P<reason>.+)$", re.DOTALL)
APPROVAL_CHOICE_RE = re.compile(r"^([1-3])([!！])?$")
JARGON_ADD_RE = re.compile(
    r"^/黑话\s*[:：]?\s*(?P<term>.+?)\s*(?:指代|意思是|=|->|:|：)\s*[:：]?\s*(?P<meaning>.+)$"
)
JARGON_DELETE_RE = re.compile(r"^/删黑话\s*[:：]?\s*(?P<term>.+)$")
JARGON_LIST_RE = re.compile(r"^/黑话列表\s*$")
APPROVAL_HELP_COMMANDS = {"审批规则", "规则", "帮助"}
APPROVAL_DETAIL_COMMANDS = {"审批规则详情", "详细规则", "规则详情", "详细解释"}
TOKEN_REPORT_COMMAND_ALIASES = {"token用量", "tokens", "token", "usage", "用量", "消耗", "费用"}

APPROVAL_RULES_MESSAGE = """张风雪审批单：
1. 机器人想发群时，只会先发候选给审批人，不会直接发群。
2. AI 默认把最想发的候选放在 1。
3. 回 1/2/3：发送对应候选；回 取消：不发。
4. 两个审批人谁先回复听谁的；另一人的旧数字会被忽略，避免串到下一条。
5. 基础审批人只有 1/2/3/取消 权限。
6. 主人可用 1!/2!/3! 标优，或 不准奏原因：xxx 记录负反馈。
7. 回 bot工具 或 审批规则详情：查看 token、拦截记录、黑话、开关、审批人管理。"""

APPROVAL_RULES_DETAIL_MESSAGE = """张风雪 bot 工具单：
一、审批
1. 回 1/2/3：发送对应候选。
2. 回 取消：不发。
3. 主人可回 1!/2!/3!：发送并标记优质。
4. 主人可回 不准奏原因：xxx；不准奏2原因：xxx：取消并记录指定候选的问题。

二、查看
1. token用量：近 24 小时 token 明细。
2. token用量 1h/7d/all：按时间窗口查询；token用量 2026-07-10：查指定日期。
3. 拦截 20 或 /bot blocked 20：查看最近后端拦截、频率门、LLM 不发原因。
4. 审批人列表：查看当前主人和基础审批人。

三、黑话
1. /黑话：咱妈 指代：中国。
2. /黑话：达斯 指代：打死。
3. /黑话列表：查看自定义黑话。
4. /删黑话：咱妈：删除指定黑话。

四、开关
1. 开启：恢复群聊进入决策。
2. 关闭：暂停群聊进入决策，并清空待审候选。
3. 每次开启/关闭会重新发送简版审批规则。

五、审批人管理
1. 加审批 123456：添加基础审批人。
2. 删审批 123456：删除基础审批人。
3. 只有主人能增删审批人；基础审批人不能用工具命令。"""
