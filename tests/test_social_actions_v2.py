import asyncio

from qq_social_agent.social_actions import PokeContext, SocialActionService


class FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_api(self, api: str, **data: object) -> dict[str, object]:
        self.calls.append((api, data))
        return {"status": "ok"}


def _service(**overrides: object) -> SocialActionService:
    values: dict[str, object] = {
        "poke_enabled": True,
        "poke_familiar_user_ids": {12345},
        "poke_global_cooldown_seconds": 0,
        "poke_per_group_cooldown_seconds": 0,
        "poke_per_user_cooldown_seconds": 0,
        "poke_max_per_group_day": 4,
    }
    values.update(overrides)
    return SocialActionService(**values)  # type: ignore[arg-type]


def test_poke_is_disabled_by_default() -> None:
    service = SocialActionService()
    bot = FakeBot()

    result = asyncio.run(
        service.poke_user(
            bot,
            group_id=100,
            user_id=12345,
            context=PokeContext(was_poked=True),
            now=1000.0,
        )
    )

    assert not result.sent
    assert result.reason == "deny_disabled"
    assert bot.calls == []


def test_reciprocal_poke_allows_unfamiliar_user() -> None:
    service = _service()
    bot = FakeBot()

    result = asyncio.run(
        service.poke_user(
            bot,
            group_id=100,
            user_id=99999,
            context=PokeContext(was_poked=True),
            now=1000.0,
        )
    )

    assert result.sent
    assert result.policy_reason == "reciprocal_poke"
    assert bot.calls == [("send_poke", {"user_id": 99999, "group_id": 100})]


def test_familiar_direct_cue_can_poke() -> None:
    service = _service()
    bot = FakeBot()

    result = asyncio.run(
        service.poke_user(
            bot,
            group_id=100,
            user_id=12345,
            context=PokeContext(directly_cued=True),
            now=1000.0,
        )
    )

    assert result.sent
    assert result.policy_reason == "familiar_direct_cue"


def test_unfamiliar_user_without_reciprocal_signal_is_denied() -> None:
    service = _service()
    bot = FakeBot()

    result = asyncio.run(
        service.poke_user(
            bot,
            group_id=100,
            user_id=99999,
            context=PokeContext(directly_cued=True, playful_banter=True),
            now=1000.0,
        )
    )

    assert not result.sent
    assert result.reason == "deny_unfamiliar_user"
    assert bot.calls == []


def test_poke_has_hard_global_and_daily_limits() -> None:
    service = _service(poke_global_cooldown_seconds=300, poke_max_per_group_day=1)
    bot = FakeBot()
    context = PokeContext(was_poked=True)

    first = asyncio.run(service.poke_user(bot, group_id=100, user_id=1, context=context, now=1000.0))
    global_limited = asyncio.run(service.poke_user(bot, group_id=200, user_id=2, context=context, now=1100.0))
    daily_limited = asyncio.run(service.poke_user(bot, group_id=100, user_id=3, context=context, now=1400.0))

    assert first.sent
    assert global_limited.reason == "poke_global_cooldown"
    assert daily_limited.reason == "poke_group_daily_limit"
