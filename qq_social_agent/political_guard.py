from __future__ import annotations

import re


SAFE_POLITICAL_REDIRECT = "这话题别在群里直球冲塔，没必要把聊天往敏感政治上带。换个能聊的。"

_TARGET_PARTY_RE = re.compile(r"(中国共产党|共产党|中共|ccp|党国)", re.IGNORECASE)
_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|邪恶|纳粹|屠杀|血债|"
    r"卖国|汉奸|垃圾|傻逼|傻卵|畜生|狗|烂透|腐败透顶)",
    re.IGNORECASE,
)

_SENSITIVE_TERMS = (
    # 历史事件与运动
    "文化大革命",
    "cultural revolution",
    "wenge",
    "wg",
    "文革",
    "反右",
    "右派",
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
    # 近现代事件与公共事件
    "天安门广场事件",
    "天安门事件",
    "天安门",
    "八九六四",
    "64事件",
    "8964",
    "六四",
    "八九",
    "64",
    "学生运动",
    "学潮",
    "学运",
    "坦克人",
    "tankman",
    "白纸运动",
    "白纸",
    "四通桥",
    "乌鲁木齐火灾",
    "佳士工人",
    "佳士",
    "乌坎",
    "铜锣湾书店",
    "雨伞运动",
    "占中",
    "721元朗",
    "831太子站",
    "胡锦涛离场",
    "上海封城",
    "动态清零",
    "铁链女",
    "丰县",
    # 人物、组织、媒体
    "习近平",
    "xi jinping",
    "xjp",
    "习主席",
    "习大大",
    "国家主席",
    "总书记",
    "政治局",
    "中南海",
    "毛泽东",
    "mzd",
    "毛主席",
    "赵紫阳",
    "刘晓波",
    "胡耀邦",
    "胡锦涛",
    "李克强",
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
    "轮子",
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
    # 疆藏港台与民族地区议题
    "新疆种族灭绝",
    "维吾尔种族灭绝",
    "维吾尔集中营",
    "新疆集中营",
    "新疆",
    "维吾尔",
    "强迫劳动",
    "再教育营",
    "东突",
    "世维会",
    "陈全国",
    "西藏独立",
    "西藏",
    "藏独",
    "达赖",
    "班禅",
    "自焚",
    "香港国安法",
    "反送中",
    "港独",
    "台湾独立",
    "台独",
    # 政治表达、黑话与隐喻
    "民主运动",
    "颜色革命",
    "公民运动",
    "零八宪章",
    "修宪",
    "人大修宪",
    "取消任期",
    "终身制",
    "连任",
    "登基",
    "维尼",
    "辱包",
    "包子",
    "习包子",
    "包帝",
    "庆丰帝",
    "刁大犬",
    "瓶子",
    "总加速师",
    "加速师",
    "墙国",
    "赵家人",
    "河蟹",
    "被消失",
    "喝茶",
    "晶哥",
)

_EXTRA_SENSITIVE_PATTERNS = (
    r"cultural\s*revolution",
    r"falun\s*gong",
    r"xi\s*jinping",
    r"x\s*j\s*p",
)

# 敏感词输出用“* + 拼音首字母”交替脱敏：习近平 -> *j*，共产党 -> *c*。
_PINYIN_CHARS = "三上下世东中丰丹主乌九习乡书事五产人件任会伞佳修元光克党全八公六共兵再刁刘刚制功加动劳包化南占卫县反取台右吾周命哥喝四困国场坎坦城基墙士大天太失夹女姚媒子学安宪家封尔尼局山工帅师希帝席帮平年广庆店康开张强彪彭律志态总慧批摘政教文斗新族时明春晓晟晶智朗期木李来林桥毛民永江沟河治法波泽洪活派海消涛清港湾潮火灭灾焚熙犬独王班瓶生疆登白禅离种秦突立站章类紫红纪纸终绝维网耀育胡自色茶荒营蔡薄藏蟹被西记许评诚贵赖赵跃身轮辱边达运近进连迫退送通速邦郭铁铜链锣锦门阳陈难集雨零霞青革颜饥香高鲁黑齐"
_PINYIN_INITIALS = "ssxsdzfdzwjxxsswcrjrhsjxygkdqbglgbzdlgzgjdlbhnzwxfqtywzmghskgcktcjqsdttsjnymzxaxjfenjsgssxdxbpngqdkkzqbplztzhpzzjwdxzsmcxcjzlqmlllqmmyjghzfbzhhphxtqgwchmzfxqdwbpsjdbclzqtlzzlzhjzzjwwyyhzschycbcxbxjxpcglzyslrbdyjjlptstsbgttlljmycnjylxqgyjxglhq"
_PINYIN_INITIAL_MAP = dict(zip(_PINYIN_CHARS, _PINYIN_INITIALS))

_SENSITIVE_DOMESTIC_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(_SENSITIVE_TERMS, key=len, reverse=True))
    + "|"
    + "|".join(_EXTRA_SENSITIVE_PATTERNS),
    re.IGNORECASE,
)
_MASK_RE = _SENSITIVE_DOMESTIC_RE
_MASK_ATTACK_RE = re.compile(
    r"(打倒|推翻|下台|灭亡|垮台|独裁|暴政|纳粹|屠杀|血债|腐败透顶)",
    re.IGNORECASE,
)


def has_political_redline(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if _SENSITIVE_DOMESTIC_RE.search(text) or _SENSITIVE_DOMESTIC_RE.search(normalized):
        return True
    return bool(_TARGET_PARTY_RE.search(normalized) and _ATTACK_RE.search(normalized))


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
    masked: list[str] = []
    visible_index = 0
    for ch in raw:
        if ch.isspace():
            masked.append(ch)
            continue
        if visible_index % 2 == 0:
            masked.append("*")
        else:
            masked.append(_initial_for_mask(ch))
        visible_index += 1
    return "".join(masked)


def _initial_for_mask(ch: str) -> str:
    if "\u4e00" <= ch <= "\u9fff":
        return _PINYIN_INITIAL_MAP.get(ch, "*")
    if ch.isalpha():
        return ch.lower()
    return "*"


def _normalize(text: str) -> str:
    text = text.lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
