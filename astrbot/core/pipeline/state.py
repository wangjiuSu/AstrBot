"""Typed state that core pipeline stages hand to each other on an event.

Before this module the stages talked through ``event.set_extra("<string>")``,
which left the stage-to-stage contract implicit. ``PipelineState`` names each
field; ``AstrMessageEvent.set_extra`` / ``get_extra`` keep accepting the old
string keys so plugins and older code keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

if TYPE_CHECKING:
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.star.star_handler import StarHandlerMetadata


T = TypeVar("T")


class _ExtraField(Generic[T]):
    """Expose one legacy extra as a typed field without duplicating its value."""

    def __init__(self, key: str, default_factory: Callable[[], T]) -> None:
        self.key = key
        self.default_factory = default_factory

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> _ExtraField[T]: ...

    @overload
    def __get__(self, instance: PipelineState, owner: type | None = None) -> T: ...

    def __get__(
        self, instance: PipelineState | None, owner: type | None = None
    ) -> T | _ExtraField[T]:
        """Read the shared value, creating an event-local default if missing.

        Args:
            instance: State view, or None for access through the class.
            owner: Class performing the descriptor lookup.

        Returns:
            Stored value, or this descriptor for class-level access.
        """
        if instance is None:
            return self
        if self.key not in instance._extras:
            instance._extras[self.key] = self.default_factory()
        return instance._extras[self.key]

    def __set__(self, instance: PipelineState, value: T) -> None:
        instance._extras[self.key] = value


class PipelineState:
    """Per-event state produced and consumed by the built-in pipeline stages.

    Attributes:
        activated_handlers: Plugin handlers whose filters matched; set by the
            wake stage, run by the process stage.
        handlers_parsed_params: Parsed command arguments keyed by handler full
            name; set by the wake stage alongside ``activated_handlers``.
        session_isolated: ``True`` when ``unique_session`` rewrote the session
            id so each group member has their own session.
        provider_request: LLM request a plugin handler yielded, picked up by the
            agent sub stage.
        reasoning_content: Model reasoning captured by the agent hooks and
            rendered by the decorate stage.
        streaming_finished: Set by the respond stage once a streaming reply
            has been fully delivered.
        agent_stop_requested: Another event asked the running agent to stop.
        agent_user_aborted: The agent run ended because the user aborted it.
    """

    activated_handlers: _ExtraField[list[StarHandlerMetadata]] = _ExtraField(
        "activated_handlers", list
    )
    handlers_parsed_params: _ExtraField[dict[str, dict[str, Any]]] = _ExtraField(
        "handlers_parsed_params", dict
    )
    session_isolated: _ExtraField[bool] = _ExtraField("_session_isolated", bool)
    provider_request: _ExtraField[ProviderRequest | None] = _ExtraField(
        "provider_request", lambda: None
    )
    reasoning_content: _ExtraField[str | None] = _ExtraField(
        "_llm_reasoning_content", lambda: None
    )
    streaming_finished: _ExtraField[bool] = _ExtraField("_streaming_finished", bool)
    agent_stop_requested: _ExtraField[bool] = _ExtraField("agent_stop_requested", bool)
    agent_user_aborted: _ExtraField[bool] = _ExtraField("agent_user_aborted", bool)

    def __init__(self, extras: dict[str, Any] | None = None) -> None:
        """Share the event's extra dictionary with legacy plugin interfaces.

        Args:
            extras: Existing extra dictionary. Defaults are populated lazily
                when a typed field is read, preserving legacy missing-key and
                explicit-None semantics. Clearing the dictionary resets state.
        """
        self._extras = extras if extras is not None else {}
