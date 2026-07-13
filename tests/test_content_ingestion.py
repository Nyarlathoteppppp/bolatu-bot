import asyncio
import base64
from types import SimpleNamespace

from qq_social_agent.content_ingestion import ContentIngestionService
from qq_social_agent.tools.file_content_reader import FileContentReader
from qq_social_agent.tools.voice_transcript import (
    VoiceTranscriptConfig,
    VoiceTranscriptContext,
    VoiceTranscriptService,
)


class FakeBot:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_api(self, api: str, **data: object) -> object:
        self.calls.append((api, data))
        return self.responses.get(api, {})


class FakeVoiceProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio: bytes, **kwargs: object) -> str:
        self.calls += 1
        assert audio == b"mp3-bytes"
        return "群友说今天天气不错"


def _event(*segments: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(message=list(segments))


def test_addressed_file_reads_base64_content() -> None:
    bot = FakeBot({"get_file": {"data": {"base64": base64.b64encode("第一行".encode()).decode()}}})
    service = ContentIngestionService(file_reader=FileContentReader())
    event = _event({"type": "file", "data": {"file_id": "f1", "name": "说明.txt"}})

    result = asyncio.run(
        service.context_for_event(
            bot,
            event,
            allow_file_content=True,
            voice_context=VoiceTranscriptContext(),
        )
    )

    assert result.file_status == "ok"
    assert "第一行" in result.file_context
    assert result.file_name == "说明.txt"
    assert result.file_source_id == "f1"
    assert result.file_text == "第一行"
    assert bot.calls[0][0] == "get_file"


def test_ordinary_file_does_not_fetch_body() -> None:
    bot = FakeBot({})
    service = ContentIngestionService()
    event = _event({"type": "file", "data": {"file_id": "f1", "name": "说明.txt"}})

    result = asyncio.run(
        service.context_for_event(
            bot,
            event,
            allow_file_content=False,
            voice_context=VoiceTranscriptContext(),
        )
    )

    assert result.file_status == "context_not_allowed"
    assert bot.calls == []


def test_addressed_voice_converts_then_transcribes() -> None:
    provider = FakeVoiceProvider()
    voice = VoiceTranscriptService(
        VoiceTranscriptConfig(enabled=True, provider_name="fake"),
        provider=provider,
    )
    bot = FakeBot(
        {
            "get_record": {"data": {"file": "/tmp/voice.mp3"}},
            "get_file": {"data": {"base64": base64.b64encode(b"mp3-bytes").decode()}},
        }
    )
    service = ContentIngestionService(voice_service=voice)
    event = _event({"type": "record", "data": {"file_id": "r1", "file": "voice.silk"}})

    result = asyncio.run(
        service.context_for_event(
            bot,
            event,
            allow_file_content=False,
            voice_context=VoiceTranscriptContext(mentioned=True),
        )
    )

    assert result.voice_status == "ok"
    assert "今天天气不错" in result.voice_context
    assert provider.calls == 1
    assert [call[0] for call in bot.calls] == ["get_record", "get_file"]
