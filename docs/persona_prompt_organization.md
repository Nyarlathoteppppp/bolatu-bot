# 风雪角色卡与运行规则

`prompts/zhangfengxue.yaml` 的 `persona.prompt` 保留身份、PList、性格和真实群友关系。北大计算机大二、CMU、彩羽式反应等措辞均未改写。

本轮只将已有的独立隐私与指令边界段落逐字移入 `persona.runtime_rules`。事实、能力、输出要求与性格混写的段落完整保留在正文；这部分尚未独立拆分。

`PersonaRegistry` 读取两个字段，按核心正文、运行规则的顺序合成 `Persona.prompt`。缺少独立 `decision_prompt` 时也使用合成结果。所有生成调用继续读取完整文本。直接构造 `Persona` 的参数不变。

整理前后，风雪的有效 `Persona.prompt` 逐字一致，`flows` 内容完全一致。当前中文卡是参考一色彩羽角色页后的适配文本，没有声称它是原卡原文。

管理台仍编辑原始 YAML；热加载会重新构造注册表，无需改变保存流程。
