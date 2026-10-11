import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from astrbot.core.pipeline.respond.stage import RespondStage  # noqa: F401
from astrbot.core.platform.sources.aiocqhttp import (
    aiocqhttp_platform_adapter as adapter_module,
)
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)

BOT = 10000
NOW = int(time.time())
SINCE = NOW - 3600


class FakePreferences:
    def __init__(self) -> None:
        self.data: dict = {}

    async def global_get(self, key, default=None):
        return self.data.get(key, default)

    async def global_put(self, key, value) -> None:
        self.data[key] = value


def _msg(message_id, user_id, sent_at, segments, group_id=None):
    payload = {
        "self_id": BOT,
        "user_id": user_id,
        "time": sent_at,
        "message_id": message_id,
        "message_type": "group" if group_id else "private",
        "sub_type": "normal" if group_id else "friend",
        "sender": {"user_id": user_id, "nickname": f"u{user_id}"},
        "message": segments,
        "raw_message": "",
        "font": 0,
        "post_type": "message_sent" if user_id == BOT else "message",
    }
    if group_id:
        payload["group_id"] = group_id
    return payload


def _text(text):
    return {"type": "text", "data": {"text": text}}


def _at(qq):
    return {"type": "at", "data": {"qq": str(qq)}}


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(adapter_module, "sp", FakePreferences())
    monkeypatch.setattr(adapter_module.asyncio, "sleep", AsyncMock())
    return AiocqhttpAdapter(
        {"id": "qq", "ws_reverse_host": "127.0.0.1", "ws_reverse_port": 6199},
        {},
        asyncio.Queue(),
    )


def _wire(adapter, contacts, histories):
    async def call_action(action, **kwargs):
        if action == "get_recent_contact":
            return contacts
        if action == "get_friend_msg_history":
            return {"messages": histories[("friend", kwargs["user_id"])]}
        if action == "get_group_msg_history":
            return {"messages": histories[("group", kwargs["group_id"])]}
        if action == "get_group_member_info":
            return {"card": "nineteen"}
        raise AssertionError(action)

    adapter.bot.call_action = AsyncMock(side_effect=call_action)


def _drain(adapter):
    events = []
    while not adapter._event_queue.empty():
        events.append(adapter._event_queue.get_nowait())
    return events


@pytest.mark.asyncio
async def test_private_backlog_replays_latest_with_earlier_context(adapter):
    _wire(
        adapter,
        [{"peerUin": "20001", "chatType": 1, "msgTime": str(NOW - 60)}],
        {
            ("friend", 20001): [
                _msg(1, 20001, SINCE - 10, [_text("before offline")]),
                _msg(2, 20001, NOW - 600, [_text("are you there")]),
                _msg(3, 20001, NOW - 60, [_text("hello?")]),
            ]
        },
    )

    await adapter._reply_offline_backlog(BOT, SINCE)

    events = _drain(adapter)
    assert len(events) == 1
    assert events[0].message_str == "hello?"
    assert events[0].message_obj.timestamp == NOW - 60
    backlog = events[0].get_extra("offline_backlog")
    assert backlog["sent_at"] == NOW - 60
    assert len(backlog["earlier"]) == 1
    assert backlog["earlier"][0].endswith("are you there")


@pytest.mark.asyncio
async def test_group_backlog_keeps_only_unanswered_mentions(adapter):
    _wire(
        adapter,
        [
            {"peerUin": "30001", "chatType": 2, "msgTime": str(NOW - 60)},
            {"peerUin": "30002", "chatType": 2, "msgTime": str(NOW - 60)},
        ],
        {
            ("group", 30001): [
                _msg(11, 20001, NOW - 300, [_at(BOT), _text("ping")], 30001),
                _msg(12, 20002, NOW - 200, [_text("just chatting")], 30001),
            ],
            ("group", 30002): [
                _msg(21, 20003, NOW - 300, [_at(BOT), _text("hi")], 30002),
                _msg(22, BOT, NOW - 100, [_text("hi there")], 30002),
            ],
        },
    )

    await adapter._reply_offline_backlog(BOT, SINCE)

    events = _drain(adapter)
    assert [e.message_obj.message_id for e in events] == ["11"]
    assert events[0].get_extra("offline_backlog")["earlier"] == []


@pytest.mark.asyncio
async def test_backlog_skips_messages_already_handled_live(adapter):
    _wire(
        adapter,
        [{"peerUin": "20001", "chatType": 1, "msgTime": str(NOW - 10)}],
        {
            ("friend", 20001): [
                _msg(2, 20001, NOW - 600, [_text("missed")]),
                _msg(3, 20001, NOW - 10, [_text("live")]),
            ]
        },
    )
    adapter._seen_message_ids.append("3")

    await adapter._reply_offline_backlog(BOT, SINCE)

    assert _drain(adapter) == []


@pytest.mark.asyncio
async def test_backlog_ignores_contacts_quiet_since_last_message(adapter):
    _wire(
        adapter,
        [{"peerUin": "20001", "chatType": 1, "msgTime": str(SINCE - 5)}],
        {},
    )

    await adapter._reply_offline_backlog(BOT, SINCE)

    assert _drain(adapter) == []
    actions = [c.kwargs["action"] for c in adapter.bot.call_action.await_args_list]
    assert actions == ["get_recent_contact"]


@pytest.mark.asyncio
async def test_handle_msg_persists_newest_message_time(adapter):
    _wire(adapter, [], {})
    payload = _msg(5, 20001, NOW - 30, [_text("hey")])
    abm = await adapter.convert_message(adapter_module.Event.from_payload(payload))

    await adapter.handle_msg(abm)

    assert adapter_module.sp.data == {f"aiocqhttp_last_message_time_qq_{BOT}": NOW - 30}
    assert "5" in adapter._seen_message_ids
    assert len(_drain(adapter)) == 1
