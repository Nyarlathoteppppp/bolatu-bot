import time

from qq_social_agent.config import RateConfig
from qq_social_agent.memory import MemoryStore
from qq_social_agent.rate_limiter import RateLimiter


def _config() -> RateConfig:
    return RateConfig(
        min_interval_seconds=60,
        hard_mention_interval_seconds=5,
        max_replies_per_10min=2,
        max_replies_per_hour=5,
        max_consecutive_replies=1,
        quiet_hours_enabled=False,
        quiet_hours_start="01:00",
        quiet_hours_end="08:00",
    )


def test_rate_limiter_allows_empty_history(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    limiter = RateLimiter(memory, _config())
    assert limiter.allow(1, mentioned=False).allowed


def test_rate_limiter_blocks_cooldown(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 999, "bot", "hello", is_bot=True, created_at=time.time())
    limiter = RateLimiter(memory, _config())
    decision = limiter.allow(1, mentioned=False)
    assert not decision.allowed
    assert decision.reason == "cooldown"


def test_rate_limiter_allows_mention_with_shorter_cooldown(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    memory.add_message(1, 999, "bot", "hello", is_bot=True, created_at=time.time() - 10)
    limiter = RateLimiter(memory, _config())
    assert limiter.allow(1, mentioned=True).allowed


def test_queued_mention_ignores_reply_sent_after_event_arrived(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "bot.sqlite3")
    arrived_at = time.time()
    memory.add_message(1, 999, "bot", "later reply", is_bot=True, created_at=arrived_at + 5)
    limiter = RateLimiter(memory, _config())

    decision = limiter.allow(1, mentioned=True, now=arrived_at + 6, event_at=arrived_at)

    assert decision.allowed
