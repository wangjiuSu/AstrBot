"""Pipeline stages read platform policy from PlatformMetadata and event hooks.

Adapter names must not leak into ``astrbot/core/pipeline``; every per-platform
behaviour is declared by the adapter (metadata flag or event-class override).
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot.core.message.components import Json, Plain
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage
from astrbot.core.pipeline.respond.stage import RespondStage
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.pipeline.whitelist_check.stage import WhitelistCheckStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.manager import BUILTIN_ADAPTER_MODULES, PlatformManager
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.platform.register import platform_cls_map
from astrbot.core.platform.sources.lark.lark_event import LarkMessageEvent
from astrbot.core.platform.sources.webchat.webchat_event import WebChatMessageEvent
from astrbot.core.platform.sources.wecom_ai_bot.wecomai_event import (
    WecomAIBotMessageEvent,
)
from astrbot.core.star.star_handler import StarHandlerRegistry

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "astrbot" / "core" / "pipeline"
ADAPTER_NAMES = sorted(BUILTIN_ADAPTER_MODULES) + ["webchat"]


def _event(meta: PlatformMetadata, *, group: bool = True) -> AstrMessageEvent:
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE if group else MessageType.FRIEND_MESSAGE
    message.group_id = "group" if group else ""
    message.self_id = "bot"
    message.sender = MessageMember(user_id="member")
    message.message = [Plain("hi")]
    event = AstrMessageEvent("hi", message, meta, "group" if group else "member")
    event.send = AsyncMock()
    return event


def _no_plugin_handlers():
    """Run the wake stage against an empty handler registry."""
    return patch.multiple(
        "astrbot.core.pipeline.waking_check.stage",
        star_handlers_registry=StarHandlerRegistry(),
        SessionPluginManager=SimpleNamespace(
            filter_handlers_by_session=AsyncMock(
                side_effect=lambda _e, handlers: handlers
            )
        ),
    )


def test_pipeline_has_no_adapter_name_literals():
    """Platform names belong to adapters; stages must consult metadata instead."""
    names = "|".join(re.escape(n) for n in ADAPTER_NAMES)
    pattern = re.compile(rf"""["'](?:{names})["']""")
    offenders = [
        f"{path.relative_to(PIPELINE_DIR.parent.parent.parent)}:{lineno}"
        for path in PIPELINE_DIR.rglob("*.py")
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == []


def test_pipeline_does_not_import_adapter_sources():
    offenders = [
        str(path.relative_to(PIPELINE_DIR))
        for path in PIPELINE_DIR.rglob("*.py")
        if "platform.sources" in path.read_text()
    ]
    assert offenders == []


@pytest.mark.asyncio
async def test_whitelist_skips_trusted_source_only():
    stage = WhitelistCheckStage()
    await stage.initialize(
        SimpleNamespace(
            astrbot_config={
                "platform_settings": {
                    "enable_id_white_list": True,
                    "id_whitelist": ["someone-else"],
                    "wl_ignore_admin_on_group": False,
                    "wl_ignore_admin_on_friend": False,
                    "id_whitelist_log": False,
                }
            }
        )
    )
    trusted = _event(PlatformMetadata("x", "x", "x", trusted_source=True))
    untrusted = _event(PlatformMetadata("x", "x", "x"))

    await stage.process(trusted)
    await stage.process(untrusted)

    assert trusted.is_stopped() is False
    assert untrusted.is_stopped() is True


@pytest.mark.asyncio
async def test_waking_private_chat_honours_trusted_source():
    stage = WakingCheckStage()
    await stage.initialize(
        SimpleNamespace(
            astrbot_config={
                "platform_settings": {"friend_message_needs_wake_prefix": True},
                "admins_id": [],
                "wake_prefix": ["/"],
            },
            astrbot_config_id="default",
            db_helper=MagicMock(),
        )
    )
    stage._umo_auto_name_recorder = MagicMock()
    trusted = _event(PlatformMetadata("x", "x", "x", trusted_source=True), group=False)
    plain = _event(PlatformMetadata("x", "x", "x"), group=False)

    with _no_plugin_handlers():
        await stage.process(trusted)
        await stage.process(plain)

    assert trusted.is_wake is True
    assert plain.is_wake is False


@pytest.mark.asyncio
async def test_waking_unique_session_uses_event_hook():
    class Isolating(AstrMessageEvent):
        def build_unique_session_id(self):
            return f"{self.get_sender_id()}#{self.get_group_id()}"

    stage = WakingCheckStage()
    await stage.initialize(
        SimpleNamespace(
            astrbot_config={
                "platform_settings": {"unique_session": True},
                "admins_id": [],
                "wake_prefix": ["/"],
            },
            astrbot_config_id="default",
            db_helper=MagicMock(),
        )
    )
    stage._umo_auto_name_recorder = MagicMock()
    meta = PlatformMetadata("x", "x", "x")
    base = _event(meta)
    isolating = Isolating(base.message_str, base.message_obj, meta, "group")
    isolating.send = AsyncMock()

    with _no_plugin_handlers():
        await stage.process(base)
        await stage.process(isolating)

    assert base.session_id == "group"
    assert base.pipeline.session_isolated is False
    assert isolating.session_id == "member#group"
    assert isolating.pipeline.session_isolated is True


@pytest.mark.asyncio
async def test_pre_ack_reaction_requires_native_reaction_flag():
    stage = PreProcessStage()
    stage.config = {
        "platform_specific": {
            "x": {"pre_ack_emoji": {"enable": True, "emojis": ["👀"]}}
        }
    }
    stage.platform_settings = {}
    stage.stt_settings = {"enable": False}
    native = _event(PlatformMetadata("x", "x", "x", support_native_reaction=True))
    fallback = _event(PlatformMetadata("x", "x", "x"))
    for ev in (native, fallback):
        ev.is_at_or_wake_command = True
        ev.react = AsyncMock()

    await stage.process(native)
    await stage.process(fallback)

    native.react.assert_awaited_once_with("👀")
    fallback.react.assert_not_awaited()


def test_segmented_reply_follows_metadata_flag():
    stage = RespondStage()
    stage.enable_seg = True
    stage.only_llm_result = False
    result = MessageEventResult(chain=[Plain("hello")])
    allowed = SimpleNamespace(
        get_result=lambda: result, platform_meta=PlatformMetadata("x", "x", "x")
    )
    denied = SimpleNamespace(
        get_result=lambda: result,
        platform_meta=PlatformMetadata("x", "x", "x", support_segmented_reply=False),
    )

    assert stage.is_seg_reply_required(allowed) is True
    assert stage.is_seg_reply_required(denied) is False


def test_reasoning_component_hook():
    base = _event(PlatformMetadata("x", "x", "x"))
    lark = LarkMessageEvent.__new__(LarkMessageEvent)

    plain = base.build_reasoning_component("why")
    panel = lark.build_reasoning_component("why")

    assert isinstance(plain, Plain) and "why" in plain.text
    assert isinstance(panel, Json) and panel.data["content"] == "why"


def test_capability_defaults_keep_unlisted_adapters_unchanged():
    """An adapter that declares nothing behaves like before the flags existed."""
    meta = PlatformMetadata("x", "x", "x")
    assert meta.support_segmented_reply is True
    assert meta.support_native_reaction is False
    assert meta.support_forward_message is False
    assert meta.trusted_source is False


def test_metadata_preserves_legacy_positional_arguments():
    meta = PlatformMetadata(
        "x",
        "description",
        "id",
        None,
        None,
        None,
        True,
        True,
        "plugin.adapter",
        {"en-US": {}},
        {"key": {}},
    )
    assert meta.module_path == "plugin.adapter"
    assert meta.i18n_resources == {"en-US": {}}
    assert meta.config_metadata == {"key": {}}
    assert meta.trusted_source is False


@pytest.mark.asyncio
async def test_scheduler_calls_on_pipeline_finished():
    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = []
    event = _event(PlatformMetadata("x", "x", "x"))
    event.on_pipeline_finished = AsyncMock()

    await scheduler.execute(event)

    event.on_pipeline_finished.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_cls", [WebChatMessageEvent, WecomAIBotMessageEvent])
async def test_streaming_adapters_signal_pipeline_end(event_cls):
    event = event_cls.__new__(event_cls)
    event.send = AsyncMock()

    await event.on_pipeline_finished()

    event.send.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_platform_manager_lazy_imports_builtin_adapter():
    """Loading a built-in type imports its module and registers the class."""
    manager = PlatformManager.__new__(PlatformManager)
    manager.astrbot_config = MagicMock()
    manager.settings = {}
    manager.event_queue = MagicMock()
    manager._inst_map = {}
    manager.platform_insts = []
    manager._start_platform_task = MagicMock()
    fake_cls = MagicMock()
    fake_cls.return_value.client_self_id = "cid"

    with (
        patch(
            "astrbot.core.platform.manager.importlib.import_module",
            side_effect=lambda *_a, **_k: platform_cls_map.__setitem__(
                "telegram", fake_cls
            ),
        ) as import_module,
        patch.dict(platform_cls_map, {}, clear=False),
    ):
        platform_cls_map.pop("telegram", None)
        await manager.load_platform({"enable": True, "type": "telegram", "id": "tg"})

    import_module.assert_called_once_with(
        BUILTIN_ADAPTER_MODULES["telegram"], package="astrbot.core.platform"
    )
    fake_cls.assert_called_once()
    assert manager.platform_insts == [fake_cls.return_value]
