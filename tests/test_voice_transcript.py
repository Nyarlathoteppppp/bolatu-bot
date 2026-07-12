import asyncio

import httpx
import pytest

from qq_social_agent.tools.voice_transcript import (
    VoiceTranscriptConfig,
    VoiceTranscriptContext,
    VoiceTranscriptRequest,
    VoiceTranscriptService,
    SiliconFlowTranscriptProvider,
    evaluate_voice_transcript_policy,
)


class FakeProvider:
    def __init__(self, text: str = "这是语音内容"):
        self.text = text
        self.calls = []

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
        language: str,
    ) -> str:
        self.calls.append((audio, filename, content_type, language))
        return self.text


def _request(context: VoiceTranscriptContext, *, audio: bytes = b"voice") -> VoiceTranscriptRequest:
    return VoiceTranscriptRequest(
        audio=audio,
        filename="group.amr",
        content_type="audio/amr",
        language="zh",
        context=context,
    )


def test_voice_transcript_is_disabled_by_default_and_never_calls_provider() -> None:
    provider = FakeProvider()
    service = VoiceTranscriptService(provider=provider)

    result = asyncio.run(service.transcribe(_request(VoiceTranscriptContext(mentioned=True))))

    assert result.status == "disabled"
    assert result.reason == "disabled"
    assert provider.calls == []
    assert service.status_snapshot()["counters"]["provider_calls"] == 0


def test_enabled_voice_transcript_skips_ordinary_unaddressed_audio() -> None:
    provider = FakeProvider()
    service = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="fake"),
        provider=provider,
    )

    result = asyncio.run(service.transcribe(_request(VoiceTranscriptContext())))

    assert result.status == "skipped"
    assert result.reason == "context_not_allowed"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (VoiceTranscriptContext(mentioned=True), "mentioned"),
        (VoiceTranscriptContext(replied_to_bot=True), "replied_to_bot"),
        (VoiceTranscriptContext(approval_requested=True), "approval_requested"),
    ],
)
def test_only_addressed_or_approval_contexts_can_call_provider(
    context: VoiceTranscriptContext,
    reason: str,
) -> None:
    provider = FakeProvider()
    config = VoiceTranscriptConfig(enabled=True, provider_name="fake")
    service = VoiceTranscriptService(config, provider=provider)

    policy = evaluate_voice_transcript_policy(config, context)
    result = asyncio.run(service.transcribe(_request(context)))

    assert policy.allowed and policy.reason == reason
    assert result.ok and result.reason == reason
    assert result.transcript == "这是语音内容"
    assert len(provider.calls) == 1
    assert "可能有识别错误" in result.to_context()


def test_enabled_service_without_provider_degrades_without_external_call() -> None:
    service = VoiceTranscriptService(VoiceTranscriptConfig(enabled=True, provider_name="not-configured"))

    result = asyncio.run(service.transcribe(_request(VoiceTranscriptContext(mentioned=True))))

    assert result.status == "provider_unavailable"
    assert result.reason == "no_transcript_provider_configured"
    assert service.status_snapshot()["provider_configured"] is False


def test_service_checks_size_and_content_type_before_provider() -> None:
    provider = FakeProvider()
    service = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="fake", max_audio_bytes=1_024),
        provider=provider,
    )

    too_large = asyncio.run(
        service.transcribe(_request(VoiceTranscriptContext(mentioned=True), audio=b"x" * 1_025))
    )
    wrong_type = asyncio.run(
        service.transcribe(
            VoiceTranscriptRequest(
                audio=b"voice",
                content_type="video/mp4",
                context=VoiceTranscriptContext(replied_to_bot=True),
            )
        )
    )

    assert too_large.status == "too_large"
    assert wrong_type.status == "unsupported_type"
    assert provider.calls == []


def test_transcript_is_truncated_and_never_stored_in_status() -> None:
    provider = FakeProvider("secret transcript " * 20)
    service = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="fake", max_transcript_chars=40),
        provider=provider,
    )

    result = asyncio.run(service.transcribe(_request(VoiceTranscriptContext(approval_requested=True))))
    status = service.status_snapshot()

    assert result.ok and result.truncated
    assert len(result.transcript) <= 40
    assert "secret transcript" not in str(status)
    assert status["counters"]["provider_calls"] == 1


def test_provider_timeout_and_error_are_friendly() -> None:
    class SlowProvider:
        async def transcribe(self, audio: bytes, **kwargs: object) -> str:
            await asyncio.sleep(0.1)
            return "late"

    timeout_service = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="slow", timeout_seconds=0.01),
        provider=SlowProvider(),
    )
    timeout = asyncio.run(timeout_service.transcribe(_request(VoiceTranscriptContext(mentioned=True))))

    class BrokenProvider:
        async def transcribe(self, audio: bytes, **kwargs: object) -> str:
            raise RuntimeError("provider secret details")

    broken_service = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="broken"),
        provider=BrokenProvider(),
    )
    failure = asyncio.run(broken_service.transcribe(_request(VoiceTranscriptContext(mentioned=True))))

    assert timeout.status == "timeout"
    assert failure.status == "provider_error"
    assert failure.reason == "RuntimeError"
    assert "provider secret details" not in str(broken_service.status_snapshot())


def test_voice_transcript_config_is_bounded_and_default_off() -> None:
    default = VoiceTranscriptConfig.from_config(None)
    configured = VoiceTranscriptConfig.from_config(
        {
            "enabled": True,
            "provider": "local-whisper",
            "timeout_seconds": 999,
            "max_audio_bytes": 1,
            "max_transcript_chars": 999_999,
            "allowed_content_types": ["AUDIO/AMR; codecs=amr", "audio/ogg"],
        }
    )

    assert not default.enabled
    assert configured.enabled
    assert configured.provider_name == "local-whisper"
    assert configured.timeout_seconds == 60.0
    assert configured.max_audio_bytes == 1_024
    assert configured.max_transcript_chars == 20_000
    assert configured.allowed_content_types == ("audio/amr", "audio/ogg")


def test_siliconflow_provider_uses_official_transcription_endpoint_without_leaking_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-secret"
        body = request.content
        assert b"FunAudioLLM/SenseVoiceSmall" in body
        assert b"voice.mp3" in body
        return httpx.Response(200, json={"text": "识别成功"})

    async def scenario() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = SiliconFlowTranscriptProvider(
                api_key="test-secret",
                base_url="https://api.siliconflow.cn/v1",
                client=client,
            )
            return await provider.transcribe(
                b"audio",
                filename="voice.mp3",
                content_type="audio/mpeg",
                language="zh",
            )

    assert asyncio.run(scenario()) == "识别成功"
    assert len(requests) == 1


def test_siliconflow_provider_from_config_requires_key(monkeypatch) -> None:
    config = VoiceTranscriptConfig(enabled=True, provider_name="siliconflow")
    monkeypatch.delenv(config.api_key_env, raising=False)

    assert SiliconFlowTranscriptProvider.from_config(config) is None

    monkeypatch.setenv(config.api_key_env, "configured")
    provider = SiliconFlowTranscriptProvider.from_config(config)
    assert provider is not None
    asyncio.run(provider.aclose())
