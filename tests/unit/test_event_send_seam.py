"""AstrMessageEvent owns the outbound contract: send is abstract, streaming falls back once."""

from unittest.mock import AsyncMock, patch

import pytest

from astrbot.core.message.components import Image, Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata


class _RecordingEvent(AstrMessageEvent):
    """Concrete event that records every chain handed to send."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: list[MessageChain] = []

    async def send(self, message: MessageChain) -> None:
        self.sent.append(message)
        await super().send(message)


def _event(cls=_RecordingEvent):
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE
    message.self_id = "bot"
    message.sender = MessageMember(user_id="member")
    message.message = [Plain("hi")]
    return cls("hi", message, PlatformMetadata("x", "x", "x"), "member")


async def _chunks(*chains: MessageChain):
    for chain in chains:
        yield chain


@pytest.fixture(autouse=True)
def _no_metrics():
    with patch(
        "astrbot.core.platform.astr_message_event.Metric.upload", new=AsyncMock()
    ):
        yield


def test_event_without_send_cannot_be_instantiated():
    class Incomplete(AstrMessageEvent):
        pass

    with pytest.raises(TypeError, match="send"):
        _event(Incomplete)


@pytest.mark.asyncio
async def test_send_records_the_send_operation():
    event = _event()
    assert event._has_send_oper is False

    await event.send(MessageChain([Plain("hi")]))

    assert event._has_send_oper is True


@pytest.mark.asyncio
async def test_default_streaming_buffers_into_one_message():
    event = _event()

    await event.send_streaming(
        _chunks(
            MessageChain([Plain("Hel")]),
            MessageChain([Plain("lo "), Plain("world")]),
            MessageChain([Image.fromURL("http://x/a.png")]),
        )
    )

    assert len(event.sent) == 1
    assert isinstance(event.sent[0].chain[0], Plain)
    assert event.sent[0].chain[0].text == "Hello world"
    assert isinstance(event.sent[0].chain[1], Image)
    assert event._has_send_oper is True


@pytest.mark.asyncio
async def test_default_streaming_with_empty_stream_sends_nothing():
    event = _event()

    await event.send_streaming(_chunks())

    assert event.sent == []
    assert event._has_send_oper is False


@pytest.mark.asyncio
async def test_realtime_segmenting_sends_finished_sentences_as_they_arrive():
    event = _event()

    with patch("astrbot.core.platform.astr_message_event.asyncio.sleep", AsyncMock()):
        await event.send_streaming(
            _chunks(
                MessageChain([Plain("第一句。第二")]),
                MessageChain([Plain("句！")]),
                MessageChain([Image.fromURL("http://x/a.png")]),
                MessageChain([Plain("尾巴没有标点")]),
            ),
            use_fallback=True,
        )

    texts = [
        c.chain[0].text if isinstance(c.chain[0], Plain) else type(c.chain[0]).__name__
        for c in event.sent
    ]
    assert texts == ["第一句。", "第二句！", "Image", "尾巴没有标点"]
    assert event._has_send_oper is True


@pytest.mark.asyncio
async def test_platform_with_native_streaming_marks_sent_without_default_delivery():
    class Native(_RecordingEvent):
        async def send_streaming(self, generator, use_fallback=False):
            self.streamed = [chain async for chain in generator]
            self.mark_sent()

    event = _event(Native)

    await event.send_streaming(_chunks(MessageChain([Plain("a")])))

    assert event.sent == []
    assert len(event.streamed) == 1
    assert event._has_send_oper is True
