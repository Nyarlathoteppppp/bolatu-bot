import sqlite3

from qq_social_agent.memory import MemoryStore


def test_mid_summary_batch_excludes_recent_messages(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(8):
        memory.add_message(1, index, f"u{index}", f"m{index}", created_at=100 + index)

    batch = memory.messages_for_mid_summary(1, keep_recent=3, batch_size=10)

    assert [message.text for message in batch] == ["m0", "m1", "m2", "m3", "m4"]


def test_add_memory_summary_advances_state(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(8):
        memory.add_message(1, index, f"u{index}", f"m{index}", created_at=100 + index)
    batch = memory.messages_for_mid_summary(1, keep_recent=3, batch_size=10)

    memory.add_memory_summary(1, batch, summary="聊了退群和找人", recall_cues=["退群", "找人"])

    summaries = memory.recent_memory_summaries(1, 3)
    assert len(summaries) == 1
    assert summaries[0].summary == "聊了退群和找人"
    assert summaries[0].recall_cues == ("退群", "找人")
    assert memory.messages_for_mid_summary(1, keep_recent=3, batch_size=10) == []


def test_relevant_memory_summaries_match_query_cues(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(8):
        memory.add_message(1, index, f"u{index}", f"m{index}", created_at=100 + index)

    first_batch = memory.messages_for_mid_summary(1, keep_recent=0, batch_size=4)
    memory.add_memory_summary(
        1,
        first_batch,
        summary="群里聊过 Claude 封中国号，态度很烦。",
        recall_cues=["Claude 封号", "讨厌中国用户"],
    )
    second_batch = memory.messages_for_mid_summary(1, keep_recent=0, batch_size=4)
    memory.add_memory_summary(
        1,
        second_batch,
        summary="群里聊过剩饭和外卖。",
        recall_cues=["剩饭", "外卖"],
    )

    summaries = memory.relevant_memory_summaries(1, "claude 为什么封号", limit=1)

    assert len(summaries) == 1
    assert "Claude" in summaries[0].summary


def test_style_rules_are_kept_recent(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    memory.add_style_rules(
        1,
        [(f"场景{i}", f"表达{i}", f"来源{i}") for i in range(5)],
        keep=3,
    )

    rules = memory.recent_style_rules(1, 10)
    assert len(rules) == 3
    assert [rule.situation for rule in rules] == ["场景2", "场景3", "场景4"]


def test_relevant_style_rules_match_current_text(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_style_rules(
        1,
        [
            ("聊亏钱", "用一句现实成本短评", "股票又亏了"),
            ("聊吃饭", "用短句接日常", "中午吃什么"),
        ],
        keep=10,
    )

    rules = memory.relevant_style_rules(1, "股票亏麻了", limit=1)

    assert len(rules) == 1
    assert rules[0].situation == "聊亏钱"


def test_personal_style_only_applies_to_source_speaker(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_style_rules(
        1,
        [("被说中时", "用目移装傻", "（目移）", (184589072,), (11,))],
    )

    assert memory.relevant_style_rules(1, "被说中了", limit=3, speaker_user_id=100) == []
    rules = memory.relevant_style_rules(1, "被说中了", limit=3, speaker_user_id=184589072)
    assert len(rules) == 1
    assert rules[0].scope == "personal"


def test_focused_style_migration_preserves_muyi_as_group(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 184589072, "小鸟", "（目移）", created_at=1)
    memory.conn.execute(
        "insert into style_rules(group_id,situation,style,source_text,created_at,scope) values(1,?,?,?,?, 'legacy')",
        ("被说中时", "用目移装傻", "（目移）", 2.0),
    )
    memory.conn.commit()

    stats = memory.migrate_focused_style_rules(1, 184589072)
    rules = memory.relevant_style_rules(1, "被说中", limit=3, speaker_user_id=100)

    assert stats["kept_group"] == 1
    assert len(rules) == 1
    assert rules[0].scope == "group"


def test_relevant_raw_corpus_examples_include_original_and_neighbors(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 100, "A", "今天午饭吃什么", created_at=100)
    memory.add_message(1, 101, "B", "股票又亏麻了，真的顶不住", created_at=101)
    memory.add_message(1, 102, "C", "这就是资本市场教育费", created_at=102)
    memory.add_message(1, 999, "bot", "机器人旧回复", is_bot=True, created_at=103)

    examples = memory.relevant_raw_corpus_examples(1, "股票亏了怎么接", limit=2, context_radius=1)

    assert len(examples) == 1
    assert examples[0].message.text == "股票又亏麻了，真的顶不住"
    assert "倒霉" in examples[0].tags
    assert [message.text for message in examples[0].before] == ["今天午饭吃什么"]
    assert [message.text for message in examples[0].after] == ["这就是资本市场教育费"]


def test_relevant_raw_corpus_examples_excludes_current_message(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 100, "A", "没人理我", created_at=100)
    memory.add_message(1, 101, "B", "没人理我", created_at=101)

    examples = memory.relevant_raw_corpus_examples(
        1,
        "没人理我",
        limit=3,
        exclude_user_id=101,
        exclude_text="没人理我",
    )

    assert [example.message.user_id for example in examples] == [100]


def test_relevant_raw_corpus_examples_prefers_selected_user_with_limit(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 100, "A", "股票亏了怎么接", created_at=100)
    memory.add_message(1, 101, "B", "股票亏了真难受", created_at=101)
    memory.add_message(1, 184589072, "小鸟", "股票亏了先别装高手", created_at=90)
    memory.add_message(1, 184589072, "小鸟", "股票亏了就是交学费", created_at=91)
    memory.add_message(1, 184589072, "小鸟", "股票亏了但这句不用学", created_at=92)

    examples = memory.relevant_raw_corpus_examples(
        1,
        "股票亏了",
        limit=4,
        preferred_user_id=184589072,
        preferred_limit=2,
        preferred_score_multiplier=1.25,
        preferred_score_bonus=2.0,
    )

    assert [example.message.user_id for example in examples[:2]] == [184589072, 184589072]
    assert sum(1 for example in examples if example.message.user_id == 184589072) == 2


def test_member_profiles_track_aliases_by_user_id(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    memory.add_message(1, 3370998238, "乌木", "第一句", created_at=100)
    memory.add_message(1, 3370998238, "🦕", "第二句", created_at=200)
    memory.add_message(1, 12345, "别人", "第三句", created_at=300)

    profiles = memory.member_profiles_for_context(1, [3370998238, 12345], limit=8)

    assert len(profiles) == 2
    assert profiles[0].user_id == 3370998238
    assert profiles[0].display_name == "🦕"
    assert profiles[0].aliases == ("🦕", "乌木")
    assert profiles[1].display_name == "别人"


def test_member_profiles_backfill_from_existing_messages(tmp_path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        create table messages (
          id integer primary key autoincrement,
          group_id integer not null,
          user_id integer not null,
          nickname text not null,
          text text not null,
          is_bot integer not null default 0,
          created_at real not null
        );
        insert into messages(group_id, user_id, nickname, text, is_bot, created_at)
        values
          (1, 3370998238, '乌木', '旧消息', 0, 100),
          (1, 3370998238, '🦕', '新消息', 0, 200),
          (1, 999, 'bot', '机器人消息', 1, 300);
        """
    )
    conn.commit()
    conn.close()

    memory = MemoryStore(db_path)
    profiles = memory.member_profiles_for_context(1, [3370998238, 999], limit=8)

    assert len(profiles) == 1
    assert profiles[0].display_name == "🦕"
    assert profiles[0].aliases == ("🦕", "乌木")


def test_member_impressions_track_tags_keywords_and_ai_summary(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    memory.add_message(1, 3370998238, "乌木", "股票又亏麻了，代码也炸了", created_at=100)
    memory.add_message(1, 3370998238, "乌木", "比特币跌得有点顶不住", created_at=200)
    memory.add_member_profile_summary(
        group_id=1,
        user_id=3370998238,
        profile_summary="经常聊行情和代码，遇到亏钱会直接破防。",
        interests=["股票", "比特币", "代码"],
        speaking_style="短句吐槽，情绪比较直接。",
        representative_texts=["股票又亏麻了，代码也炸了"],
        start_at=100,
        end_at=200,
        message_count=2,
    )

    impressions = memory.member_impressions_for_context(1, [3370998238], limit=8)

    assert len(impressions) == 1
    impression = impressions[0]
    assert impression.message_count == 2
    assert impression.ai_summary == "经常聊行情和代码，遇到亏钱会直接破防。"
    assert impression.ai_interests == ("股票", "比特币", "代码")
    assert "行情" in [tag for tag, _ in impression.top_tags]
    assert impression.ai_representative_texts == ("股票又亏麻了，代码也炸了",)


def test_active_member_ids_since_respects_min_messages(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    memory.add_message(1, 100, "A", "第一句有内容", created_at=100)
    memory.add_message(1, 100, "A", "第二句有内容", created_at=110)
    memory.add_message(1, 200, "B", "只有一句", created_at=120)

    assert memory.active_member_ids_since(1, since_at=90, limit=10, min_messages=2) == [100]


def test_bot_sent_message_and_recalled_feedback_round_trip(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    memory.add_bot_sent_message(
        group_id=1,
        message_id=42,
        bot_reply="这句很僵硬",
        trigger_user_id=100,
        trigger_nickname="群友",
        trigger_text="你怎么看",
        action="answer",
        created_at=1000.0,
    )
    sent = memory.bot_sent_message(1, 42)

    assert sent is not None
    assert sent.bot_reply == "这句很僵硬"
    assert sent.trigger_text == "你怎么看"

    memory.add_recalled_reply_feedback(
        group_id=1,
        message_id=42,
        bot_reply="这句很僵硬",
        trigger_user_id=100,
        trigger_nickname="群友",
        trigger_text="你怎么看",
        action="answer",
        owner_reason="回答僵硬",
        scene_summary="群友问看法",
        bad_reply_problem="回答像模板",
        avoid_rule="不要模板化回答",
        better_direction="先给短判断",
        tags=["僵硬", "模板"],
        operator_id=1535071184,
        reason_user_id=1535071184,
        recalled_at=1100.0,
        reason_at=1110.0,
    )

    feedback = memory.recent_recalled_reply_feedback(1, 3)
    assert len(feedback) == 1
    assert feedback[0].owner_reason == "回答僵硬"
    assert feedback[0].avoid_rule == "不要模板化回答"
    assert feedback[0].tags == ("僵硬", "模板")

    memory.add_approved_reply_feedback(
        group_id=1,
        candidate_text="这条挺像群友",
        trigger_user_id=100,
        trigger_nickname="群友",
        trigger_text="没人理我",
        action="tease",
        style="接住倒霉情绪，短句吐槽",
        operator_id=3370998238,
        created_at=1120.0,
    )

    approved = memory.recent_approved_reply_feedback(1, 3)
    assert len(approved) == 1
    assert approved[0].candidate_text == "这条挺像群友"
    assert approved[0].style == "接住倒霉情绪，短句吐槽"

    memory.upsert_custom_jargon(
        group_id=1,
        term="达斯",
        explanation="指代：打死",
        created_by=3370998238,
    )
    jargon = memory.custom_jargon_entries(1)
    assert len(jargon) == 1
    assert jargon[0].term == "达斯"
    assert jargon[0].explanation == "指代：打死"
    assert memory.delete_custom_jargon(1, "达斯")
    assert memory.custom_jargon_entries(1) == []


def test_messages_before_returns_prior_context(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    for index in range(8):
        memory.add_message(1, index, f"u{index}", f"m{index}", created_at=100 + index)

    messages = memory.messages_before(1, before_at=106, limit=3)

    assert [message.text for message in messages] == ["m3", "m4", "m5"]


def test_llm_usage_summary_and_recent_events(tmp_path, monkeypatch) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    monkeypatch.setattr("time.time", lambda: 2000.0)
    memory.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        created_at=1000.0,
    )
    memory.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=200,
        completion_tokens=30,
        total_tokens=230,
        created_at=1990.0,
    )
    memory.add_llm_usage(
        task="reply_candidates",
        model="deepseek-v4-pro",
        prompt_tokens=300,
        completion_tokens=40,
        total_tokens=None,
        created_at=1995.0,
    )

    summaries = memory.llm_usage_summary(since_seconds=60)
    recent = memory.recent_llm_usage_events(since_seconds=60, limit=2)

    assert [(item.task, item.model, item.call_count, item.total_tokens) for item in summaries] == [
        ("reply_candidates", "deepseek-v4-pro", 1, 340),
        ("decision", "deepseek-v4-flash", 1, 230),
    ]
    assert [event.task for event in recent] == ["reply_candidates", "decision"]


def test_llm_usage_summary_accepts_absolute_time_range(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_llm_usage(
        task="before",
        model="deepseek-v4-flash",
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        created_at=99.0,
    )
    memory.add_llm_usage(
        task="inside",
        model="deepseek-v4-flash",
        prompt_tokens=200,
        completion_tokens=20,
        total_tokens=220,
        created_at=100.0,
    )
    memory.add_llm_usage(
        task="after",
        model="deepseek-v4-flash",
        prompt_tokens=300,
        completion_tokens=30,
        total_tokens=330,
        created_at=200.0,
    )

    summaries = memory.llm_usage_summary(start_at=100.0, end_at=200.0)
    recent = memory.recent_llm_usage_events(start_at=100.0, end_at=200.0, limit=5)

    assert [(item.task, item.total_tokens) for item in summaries] == [("inside", 220)]
    assert [event.task for event in recent] == ["inside"]


def test_metric_events_and_memory_atoms_round_trip(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_metric_event(
        event_type="decision_result",
        group_id=1026813421,
        user_id=100,
        stage="llm",
        action="echo_mood",
        metadata={"reason": "接情绪"},
        created_at=1000.0,
    )
    memory.add_metric_event(
        event_type="approval_accepted",
        group_id=1026813421,
        user_id=100,
        stage="approval",
        action="echo_mood",
        metadata={},
        created_at=1010.0,
    )

    summaries = memory.metric_summary(start_at=900.0, end_at=1100.0, group_id=1026813421)
    recent = memory.recent_metric_events(start_at=900.0, end_at=1100.0, group_id=1026813421)

    assert [(item.event_type, item.stage, item.action, item.count) for item in summaries] == [
        ("approval_accepted", "approval", "echo_mood", 1),
        ("decision_result", "llm", "echo_mood", 1),
    ]
    assert recent[0].event_type == "approval_accepted"
    assert recent[1].metadata == {"reason": "接情绪"}

    atom_id = memory.upsert_memory_atom(
        atom_type="relation",
        group_id=1026813421,
        subject_user_id=1535071184,
        content="歌迷老蛆是张风雪的主人。",
        source="test",
        confidence=1.0,
        importance=0.9,
    )
    assert atom_id > 0

    atoms = memory.relevant_memory_atoms(
        1026813421,
        "主人",
        subject_user_ids=[1535071184],
        limit=3,
    )
    assert len(atoms) == 1
    assert atoms[0].content == "歌迷老蛆是张风雪的主人。"
    assert memory.delete_memory_atom(atom_id)


def test_llm_usage_source_key_deduplicates(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    assert memory.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        created_at=1000.0,
        source_key="log:1",
    )
    assert not memory.add_llm_usage(
        task="decision",
        model="deepseek-v4-flash",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        created_at=1000.0,
        source_key="log:1",
    )

    summaries = memory.llm_usage_summary()
    assert len(summaries) == 1
    assert summaries[0].call_count == 1


def test_app_kv_round_trip(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")

    assert memory.app_kv_get("notice:1") is None
    memory.app_kv_set("notice:1", "sent")

    assert memory.app_kv_get("notice:1") == "sent"
