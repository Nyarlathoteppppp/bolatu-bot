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
