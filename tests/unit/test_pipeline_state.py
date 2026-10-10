"""Typed PipelineState replaces the string-keyed extras between core stages."""

from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.pipeline.process_stage.stage import ProcessStage
from astrbot.core.pipeline.state import PipelineState
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import ProviderRequest


def _event() -> AstrMessageEvent:
    message = AstrBotMessage()
    message.type = MessageType.FRIEND_MESSAGE
    message.self_id = "bot"
    message.sender = MessageMember(user_id="member")
    message.message = [Plain("hi")]
    return AstrMessageEvent("hi", message, PlatformMetadata("x", "x", "x"), "member")


def test_set_extra_with_legacy_key_lands_on_typed_state():
    event = _event()
    req = ProviderRequest(prompt="p")

    event.set_extra("provider_request", req)
    event.set_extra("_session_isolated", True)

    assert event.pipeline.provider_request is req
    assert event.pipeline.session_isolated is True
    assert event.get_extra()["provider_request"] is req


def test_get_extra_with_legacy_key_reads_typed_state():
    event = _event()
    event.pipeline.reasoning_content = "because"
    event.pipeline.activated_handlers = ["h"]

    assert event.get_extra("_llm_reasoning_content") == "because"
    assert event.get_extra("activated_handlers") == ["h"]
    assert event.get_extra("provider_request", "dflt") == "dflt"
    assert event.get_extra("_streaming_finished") is None


def test_legacy_dictionary_and_typed_state_share_updates():
    event = _event()
    extras = event.get_extra()
    extras["agent_stop_requested"] = True
    assert event.pipeline.agent_stop_requested is True
    event.pipeline.reasoning_content = "because"
    assert extras["_llm_reasoning_content"] == "because"
    assert dict(extras)["agent_stop_requested"] is True
    del extras["agent_stop_requested"]
    assert event.pipeline.agent_stop_requested is False


def test_legacy_defaults_and_explicit_none_are_preserved():
    event = _event()
    assert event.get_extra("agent_stop_requested", "missing") == "missing"
    event.set_extra("provider_request", None)
    assert event.get_extra("provider_request", "missing") is None


def test_clear_extra_resets_pipeline_state_in_place():
    event = _event()
    state = event.pipeline
    state.provider_request = ProviderRequest(prompt="old")
    state.agent_stop_requested = True
    state.activated_handlers.append("handler")
    event.set_extra("plugin", "value")

    event.clear_extra()

    assert event.pipeline is state
    assert event.get_extra() == {}
    assert state.provider_request is None
    assert state.agent_stop_requested is False
    assert state.activated_handlers == []


def test_pipeline_mutable_defaults_are_event_local():
    first, second = _event(), _event()
    first.pipeline.activated_handlers.append("handler")
    first.pipeline.handlers_parsed_params["handler"] = {"arg": 1}
    assert second.pipeline.activated_handlers == []
    assert second.pipeline.handlers_parsed_params == {}
    assert PipelineState().activated_handlers == []


def test_copied_event_rebinds_pipeline_to_copied_extras():
    original = _event()
    original.pipeline.reasoning_content = "original"
    cloned = copy(original)
    cloned._extras = dict(original.get_extra())
    cloned.pipeline.reasoning_content = "copy"
    cloned.pipeline.agent_stop_requested = True

    assert original.pipeline.reasoning_content == "original"
    assert original.pipeline.agent_stop_requested is False
    assert cloned.get_extra("_llm_reasoning_content") == "copy"


def test_plugin_keys_stay_in_extras():
    event = _event()
    event.set_extra("my_plugin_key", 1)

    assert event.get_extra("my_plugin_key") == 1
    assert event.get_extra() == {"my_plugin_key": 1}


@pytest.mark.parametrize(
    ("sent", "addressed", "call_llm", "result", "stopped", "expected"),
    [
        (False, True, False, None, False, True),
        (False, True, False, "reply", False, True),
        (False, True, False, "reply", True, False),
        (False, True, False, None, True, False),
        (True, True, False, None, False, False),
        (False, False, False, None, False, False),
        (False, True, True, None, False, False),
    ],
)
def test_should_run_default_agent(sent, addressed, call_llm, result, stopped, expected):
    event = _event()
    event._has_send_oper = sent
    event.is_at_or_wake_command = addressed
    event.call_llm = call_llm
    if result is not None:
        event.set_result(MessageEventResult().message(result))
    if stopped:
        event.stop_event()

    assert event.should_run_default_agent() is expected


@pytest.mark.asyncio
async def test_process_stage_routes_plugin_llm_request_through_typed_state():
    stage = ProcessStage.__new__(ProcessStage)
    stage.ctx = SimpleNamespace(astrbot_config={"provider_settings": {"enable": True}})
    req = ProviderRequest(prompt="from plugin")

    async def star_process(_event):
        yield req

    async def agent_process(event):
        assert event.pipeline.provider_request is req
        yield

    stage.star_request_sub_stage = SimpleNamespace(process=star_process)
    stage.agent_sub_stage = SimpleNamespace(process=agent_process)
    event = _event()
    event.pipeline.activated_handlers = [MagicMock()]
    event._has_send_oper = True  # no default agent after the plugin path

    async for _ in stage.process(event):
        pass

    assert event.pipeline.provider_request is req


@pytest.mark.asyncio
async def test_process_stage_runs_default_agent_from_predicate():
    stage = ProcessStage.__new__(ProcessStage)
    stage.ctx = SimpleNamespace(astrbot_config={"provider_settings": {"enable": True}})
    agent = AsyncMock()

    async def agent_process(event):
        await agent(event)
        yield

    stage.star_request_sub_stage = SimpleNamespace(process=None)
    stage.agent_sub_stage = SimpleNamespace(process=agent_process)
    addressed = _event()
    addressed.is_at_or_wake_command = True
    silent = _event()

    async for _ in stage.process(addressed):
        pass
    async for _ in stage.process(silent):
        pass

    agent.assert_awaited_once_with(addressed)
