from collections.abc import AsyncGenerator
from typing import Any

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, MessageChain


class DingtalkMessageEvent(AstrMessageEvent):
    def __init__(
        self,
        message_str,
        message_obj,
        platform_meta,
        session_id,
        client: Any = None,
        adapter: "Any" = None,
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.client = client
        self.adapter = adapter

    def build_unique_session_id(self) -> str | None:
        return self.get_sender_id()

    async def send(self, message: MessageChain) -> None:
        if not self.adapter:
            logger.error("钉钉消息发送失败: 缺少 adapter")
            return
        await self.adapter.send_message_chain_with_incoming(
            incoming_message=self.message_obj.raw_message,
            message_chain=message,
        )
        await super().send(message)

    async def send_streaming(
        self,
        generator: AsyncGenerator[MessageChain, None],
        use_fallback: bool = False,
    ) -> None:
        """Deliver a streaming reply as one buffered message.

        DingTalk replies go through one HTTP message call, so the stream is
        always delivered as a single message. ``use_fallback`` is ignored.

        Args:
            generator: Reply chunks as the model produces them.
            use_fallback: Ignored; this platform always buffers.
        """
        await super().send_streaming(generator, use_fallback=False)
