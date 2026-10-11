import asyncio
import inspect
import itertools
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable
from typing import Any, cast

from aiocqhttp import CQHttp, Event
from aiocqhttp.exceptions import ActionFailed

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import *
from astrbot.api.platform import (
    AstrBotMessage,
    MessageMember,
    MessageType,
    Platform,
    PlatformMetadata,
)
from astrbot.core import sp
from astrbot.core.platform.astr_message_event import MessageSesion

from ...register import register_platform_adapter
from .aiocqhttp_message_event import *
from .aiocqhttp_message_event import AiocqhttpMessageEvent

OFFLINE_BACKLOG_MAX_AGE_SEC = 6 * 3600
OFFLINE_BACKLOG_HISTORY_COUNT = 50


@register_platform_adapter(
    "aiocqhttp",
    "适用于 OneBot V11 标准的消息平台适配器，支持反向 WebSockets。",
    support_streaming_message=False,
)
class AiocqhttpAdapter(Platform):
    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: asyncio.Queue,
    ) -> None:
        super().__init__(platform_config, event_queue)

        self.settings = platform_settings
        self.host = platform_config["ws_reverse_host"]
        self.port = platform_config["ws_reverse_port"]

        self.metadata = PlatformMetadata(
            name="aiocqhttp",
            description="适用于 OneBot 标准的消息平台适配器，支持反向 WebSockets。",
            id=cast(str, self.config.get("id")),
            support_streaming_message=False,
            support_forward_message=True,
        )

        # Newest OneBot message time seen per bot account, persisted so a
        # restart knows where the offline gap starts.
        self._last_message_time: dict[str, int] = {}
        self._seen_message_ids: deque[str] = deque(maxlen=2000)
        self._backlog_tasks: dict[str, asyncio.Task] = {}

        self.bot = CQHttp(
            use_ws_reverse=True,
            import_name="aiocqhttp",
            api_timeout_sec=180,
            access_token=platform_config.get(
                "ws_reverse_token",
            ),  # 以防旧版本配置不存在
        )

        @self.bot.on_request()
        async def request(event: Event) -> None:
            try:
                abm = await self.convert_message(event)
                if not abm:
                    return
                await self.handle_msg(abm)
            except Exception as e:
                logger.exception(f"Handle request message failed: {e}")
                return

        @self.bot.on_notice()
        async def notice(event: Event) -> None:
            try:
                abm = await self.convert_message(event)
                if abm:
                    await self.handle_msg(abm)
            except Exception as e:
                logger.exception(f"Handle notice message failed: {e}")
                return

        @self.bot.on_message("group")
        async def group(event: Event) -> None:
            try:
                abm = await self.convert_message(event)
                if abm:
                    await self.handle_msg(abm)
            except Exception as e:
                logger.exception(f"Handle group message failed: {e}")
                return

        @self.bot.on_message("private")
        async def private(event: Event) -> None:
            try:
                abm = await self.convert_message(event)
                if abm:
                    await self.handle_msg(abm)
            except Exception as e:
                logger.exception(f"Handle private message failed: {e}")
                return

        @self.bot.on_websocket_connection
        async def on_websocket_connection(event: Event) -> None:
            logger.info("aiocqhttp(OneBot v11) 适配器已连接。")
            self_id = str(event.self_id)
            running = self._backlog_tasks.get(self_id)
            if running and not running.done():
                return
            since = self._last_message_time.get(self_id) or await sp.global_get(
                f"aiocqhttp_last_message_time_{self.metadata.id}_{self_id}", 0
            )
            if not since:
                # First run for this account: nothing to compare against.
                return
            self._backlog_tasks[self_id] = asyncio.create_task(
                self._reply_offline_backlog(
                    event.self_id,
                    max(int(since), int(time.time()) - OFFLINE_BACKLOG_MAX_AGE_SEC),
                )
            )

    async def send_by_session(
        self,
        session: MessageSesion,
        message_chain: MessageChain,
    ) -> None:
        is_group = session.message_type == MessageType.GROUP_MESSAGE
        if is_group:
            session_id = session.session_id.split("_")[-1]
        else:
            session_id = session.session_id
        await AiocqhttpMessageEvent.send_message(
            bot=self.bot,
            message_chain=message_chain,
            event=None,  # 这里不需要 event，因为是通过 session 发送的
            is_group=is_group,
            session_id=session_id,
        )
        await super().send_by_session(session, message_chain)

    async def convert_message(self, event: Event) -> AstrBotMessage | None:
        logger.debug(f"[aiocqhttp] RawMessage {event}")

        if event["post_type"] == "message":
            abm = await self._convert_handle_message_event(event)
            if abm.sender.user_id == "2854196310":
                # 屏蔽 QQ 管家的消息
                return None
        elif event["post_type"] == "notice":
            abm = await self._convert_handle_notice_event(event)
        elif event["post_type"] == "request":
            abm = await self._convert_handle_request_event(event)

        return abm

    async def _convert_handle_request_event(self, event: Event) -> AstrBotMessage:
        """OneBot V11 请求类事件"""
        abm = AstrBotMessage()
        abm.self_id = str(event.self_id)
        abm.sender = MessageMember(
            user_id=str(event.user_id), nickname=str(event.user_id)
        )
        abm.type = MessageType.OTHER_MESSAGE
        if event.get("group_id"):
            abm.type = MessageType.GROUP_MESSAGE
            abm.group_id = str(event.group_id)
        else:
            abm.type = MessageType.FRIEND_MESSAGE
        abm.session_id = (
            str(event.group_id)
            if abm.type == MessageType.GROUP_MESSAGE
            else abm.sender.user_id
        )
        abm.message_str = ""
        abm.message = []
        abm.timestamp = int(time.time())
        abm.message_id = uuid.uuid4().hex
        abm.raw_message = event
        return abm

    async def _convert_handle_notice_event(self, event: Event) -> AstrBotMessage:
        """OneBot V11 通知类事件"""
        abm = AstrBotMessage()
        abm.self_id = str(event.self_id)
        abm.sender = MessageMember(
            user_id=str(event.user_id), nickname=str(event.user_id)
        )
        abm.type = MessageType.OTHER_MESSAGE
        if event.get("group_id"):
            abm.group_id = str(event.group_id)
            abm.type = MessageType.GROUP_MESSAGE
        else:
            abm.type = MessageType.FRIEND_MESSAGE
        abm.session_id = (
            str(event.group_id)
            if abm.type == MessageType.GROUP_MESSAGE
            else abm.sender.user_id
        )
        abm.message_str = ""
        abm.message = []
        abm.raw_message = event
        abm.timestamp = int(time.time())
        abm.message_id = uuid.uuid4().hex

        if "sub_type" in event:
            if event["sub_type"] == "poke" and "target_id" in event:
                abm.message.append(Poke(id=str(event["target_id"])))

        return abm

    async def _convert_handle_message_event(
        self,
        event: Event,
        get_reply=True,
    ) -> AstrBotMessage:
        """OneBot V11 消息类事件

        @param event: 事件对象
        @param get_reply: 是否获取回复消息。这个参数是为了防止多个回复嵌套。
        """
        assert event.sender is not None
        abm = AstrBotMessage()
        abm.self_id = str(event.self_id)
        abm.sender = MessageMember(
            str(event.sender["user_id"]),
            event.sender.get("card") or event.sender.get("nickname", "N/A"),
        )
        if event["message_type"] == "group":
            abm.type = MessageType.GROUP_MESSAGE
            abm.group_id = str(event.group_id)
            abm.group = Group(str(event.group_id))
            abm.group.group_name = event.get("group_name")
        elif event["message_type"] == "private":
            abm.type = MessageType.FRIEND_MESSAGE
        abm.session_id = (
            str(event.group_id)
            if abm.type == MessageType.GROUP_MESSAGE
            else abm.sender.user_id
        )

        abm.message_id = str(event.message_id)
        abm.message = []

        message_str = ""
        if not isinstance(event.message, list):
            err = f"aiocqhttp: 无法识别的消息类型: {event.message!s}，此条消息将被忽略。如果您在使用 go-cqhttp，请将其配置文件中的 message.post-format 更改为 array。"
            logger.critical(err)
            try:
                await self.bot.send(event, err)
            except BaseException as e:
                logger.error(f"回复消息失败: {e}")
            raise ValueError(err)

        # 按消息段类型类型适配
        routing_params = {"self_id": event.self_id} if event.self_id else {}
        for t, m_group in itertools.groupby(event.message, key=lambda x: x["type"]):
            a = None
            if t == "text":
                current_text = "".join(m["data"]["text"] for m in m_group).strip()
                if not current_text:
                    # 如果文本段为空，则跳过
                    continue
                message_str += current_text
                a = ComponentTypes[t](text=current_text)
                abm.message.append(a)

            elif t == "file":
                for m in m_group:
                    if m["data"].get("url") and m["data"].get("url").startswith("http"):
                        # Lagrange
                        logger.info("guessing lagrange")
                        # 检查多个可能的文件名字段
                        file_name = (
                            m["data"].get("file_name", "")
                            or m["data"].get("name", "")
                            or m["data"].get("file", "")
                            or "file"
                        )
                        abm.message.append(File(name=file_name, url=m["data"]["url"]))
                    else:
                        try:
                            # Napcat
                            ret = None
                            if abm.type == MessageType.GROUP_MESSAGE:
                                ret = await self.bot.call_action(
                                    action="get_group_file_url",
                                    file_id=event.message[0]["data"]["file_id"],
                                    group_id=event.group_id,
                                    **routing_params,
                                )
                            elif abm.type == MessageType.FRIEND_MESSAGE:
                                ret = await self.bot.call_action(
                                    action="get_private_file_url",
                                    file_id=event.message[0]["data"]["file_id"],
                                    **routing_params,
                                )
                            if ret and "url" in ret:
                                file_url = ret["url"]  # https
                                # 优先从 API 返回值获取文件名，其次从原始消息数据获取
                                file_name = (
                                    ret.get("file_name", "")
                                    or ret.get("name", "")
                                    or m["data"].get("file", "")
                                    or m["data"].get("file_name", "")
                                )
                                a = File(name=file_name, url=file_url)
                                abm.message.append(a)
                            else:
                                logger.error(f"获取文件失败: {ret}")

                        except ActionFailed as e:
                            logger.error(f"获取文件失败: {e}，此消息段将被忽略。")
                        except BaseException as e:
                            logger.error(f"获取文件失败: {e}，此消息段将被忽略。")

            elif t == "reply":
                for m in m_group:
                    if not get_reply:
                        a = ComponentTypes[t](**m["data"])
                        abm.message.append(a)
                    else:
                        try:
                            reply_event_data = await self.bot.call_action(
                                action="get_msg",
                                message_id=int(m["data"]["id"]),
                                **routing_params,
                            )
                            # 添加必要的 post_type 字段，防止 Event.from_payload 报错
                            reply_event_data["post_type"] = "message"
                            new_event = Event.from_payload(reply_event_data)
                            if not new_event:
                                logger.error(
                                    f"无法从回复消息数据构造 Event 对象: {reply_event_data}",
                                )
                                continue
                            abm_reply = await self._convert_handle_message_event(
                                new_event,
                                get_reply=False,
                            )

                            reply_seg = Reply(
                                id=abm_reply.message_id,
                                chain=abm_reply.message,
                                sender_id=abm_reply.sender.user_id,
                                sender_nickname=abm_reply.sender.nickname,
                                time=abm_reply.timestamp,
                                message_str=abm_reply.message_str,
                                text=abm_reply.message_str,  # for compatibility
                                qq=abm_reply.sender.user_id,  # for compatibility
                            )

                            abm.message.append(reply_seg)
                        except BaseException as e:
                            logger.error(f"获取引用消息失败: {e}。")
                            a = ComponentTypes[t](**m["data"])
                            abm.message.append(a)
            elif t == "at":
                first_at_self_processed = False
                # Accumulate @ mention text for efficient concatenation
                at_parts = []

                for m in m_group:
                    try:
                        if m["data"]["qq"] == "all":
                            abm.message.append(At(qq="all", name="全体成员"))
                            continue

                        at_info = await self.bot.call_action(
                            action="get_group_member_info",
                            group_id=event.group_id,
                            user_id=int(m["data"]["qq"]),
                            no_cache=False,
                            **routing_params,
                        )
                        if at_info:
                            nickname = at_info.get("card", "")
                            if nickname == "":
                                at_info = await self.bot.call_action(
                                    action="get_stranger_info",
                                    user_id=int(m["data"]["qq"]),
                                    no_cache=False,
                                    **routing_params,
                                )
                                nickname = at_info.get("nick", "") or at_info.get(
                                    "nickname",
                                    "",
                                )
                            is_at_self = str(m["data"]["qq"]) in {abm.self_id, "all"}

                            abm.message.append(
                                At(
                                    qq=m["data"]["qq"],
                                    name=nickname,
                                ),
                            )

                            if is_at_self and not first_at_self_processed:
                                # 第一个@是机器人，不添加到message_str
                                first_at_self_processed = True
                            else:
                                # 非第一个@机器人或@其他用户，添加到message_str
                                at_parts.append(f" @{nickname}({m['data']['qq']}) ")
                        else:
                            abm.message.append(At(qq=str(m["data"]["qq"]), name=""))
                    except ActionFailed as e:
                        logger.error(f"获取 @ 用户信息失败: {e}，此消息段将被忽略。")
                    except BaseException as e:
                        logger.error(f"获取 @ 用户信息失败: {e}，此消息段将被忽略。")

                message_str += "".join(at_parts)
            elif t == "mface":
                continue
            elif t == "markdown":
                for m in m_group:
                    text = m["data"].get("markdown") or m["data"].get("content", "")
                    abm.message.append(Plain(text=text))
                    message_str += text
            else:
                for m in m_group:
                    try:
                        if t not in ComponentTypes:
                            logger.warning(
                                f"不支持的消息段类型，已忽略: {t}, data={m['data']}"
                            )
                            continue
                        a = ComponentTypes[t](**m["data"])
                        abm.message.append(a)
                    except Exception as e:
                        logger.exception(
                            f"消息段解析失败: type={t}, data={m['data']}. {e}"
                        )
                        continue

        abm.timestamp = int(time.time())
        abm.message_str = message_str
        abm.raw_message = event

        return abm

    def run(self) -> Awaitable[Any]:
        if not self.host or not self.port:
            logger.warning(
                "aiocqhttp: 未配置 ws_reverse_host 或 ws_reverse_port，将使用默认值：http://0.0.0.0:6199",
            )
            self.host = "0.0.0.0"
            self.port = 6199

        coro = self.bot.run_task(
            host=self.host,
            port=int(self.port),
            shutdown_trigger=self.shutdown_trigger_placeholder,
        )

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.getLogger("aiocqhttp").setLevel(logging.ERROR)
        self.shutdown_event = asyncio.Event()
        return coro

    async def terminate(self) -> None:
        if hasattr(self, "shutdown_event"):
            self.shutdown_event.set()
        for task in self._backlog_tasks.values():
            task.cancel()
        await self._close_reverse_ws_connections()

    async def _close_reverse_ws_connections(self) -> None:
        api_clients = getattr(self.bot, "_wsr_api_clients", None)
        event_clients = getattr(self.bot, "_wsr_event_clients", None)

        ws_clients: set[Any] = set()
        if isinstance(api_clients, dict):
            ws_clients.update(api_clients.values())
        if isinstance(event_clients, set):
            ws_clients.update(event_clients)

        close_tasks: list[Awaitable[Any]] = []
        for ws in ws_clients:
            close_func = getattr(ws, "close", None)
            if not callable(close_func):
                continue
            try:
                close_result = close_func(code=1000, reason="Adapter shutdown")
            except TypeError:
                close_result = close_func()
            except Exception:
                continue

            if inspect.isawaitable(close_result):
                close_tasks.append(close_result)

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        if isinstance(api_clients, dict):
            api_clients.clear()
        if isinstance(event_clients, set):
            event_clients.clear()

    async def shutdown_trigger_placeholder(self) -> None:
        await self.shutdown_event.wait()
        logger.info("aiocqhttp 适配器已被关闭")

    def meta(self) -> PlatformMetadata:
        return self.metadata

    def create_event(self, message: AstrBotMessage) -> AiocqhttpMessageEvent:
        """Creates an aiocqhttp message event.

        Args:
            message: AstrBot message object to wrap.

        Returns:
            Created aiocqhttp message event.
        """
        return AiocqhttpMessageEvent(
            message_str=message.message_str,
            message_obj=message,
            platform_meta=self.meta(),
            session_id=message.session_id,
            bot=self.bot,
        )

    async def handle_msg(
        self, message: AstrBotMessage, offline_backlog: dict | None = None
    ) -> None:
        """Commit a converted message and remember how far messages were handled.

        Args:
            message: Converted message.
            offline_backlog: Set when the message was missed while offline.
                ``sent_at`` is the original Unix time and ``earlier`` lists
                older unanswered messages from the same sender.
        """
        raw = message.raw_message
        if isinstance(raw, dict) and raw.get("post_type") == "message":
            self._seen_message_ids.append(str(message.message_id))
            sent_at = int(raw.get("time") or time.time())
            if sent_at > self._last_message_time.get(message.self_id, 0):
                self._last_message_time[message.self_id] = sent_at
                await sp.global_put(
                    f"aiocqhttp_last_message_time_{self.metadata.id}_{message.self_id}",
                    sent_at,
                )
        event = self.create_event(message)
        if offline_backlog:
            event.set_extra("offline_backlog", offline_backlog)
        self.commit_event(event)

    async def _reply_offline_backlog(self, self_id: int | str, since: int) -> None:
        """Replay mentions and private messages that arrived while offline.

        NapCat does not resend events missed by a closed reverse WebSocket, so
        recent chats are read back through its history API. Per sender, only
        the newest unanswered message is replayed; older ones ride along as
        ``earlier`` context. A chat counts as answered once the bot spoke in it.

        Args:
            self_id: Bot account that just connected.
            since: Unix time of the last handled message, capped by
                ``OFFLINE_BACKLOG_MAX_AGE_SEC``.
        """
        # Let live events from the new connection land first so they are
        # recognised as already handled.
        await asyncio.sleep(5)
        bot_id = str(self_id)
        try:
            contacts = await self.bot.call_action(
                action="get_recent_contact", count=50, self_id=self_id
            )
        except Exception as e:
            logger.warning(
                f"aiocqhttp: offline backlog skipped, get_recent_contact failed: {e}"
            )
            return

        replayed = 0
        for contact in contacts or []:
            try:
                if int(contact.get("msgTime") or 0) <= since:
                    continue
                peer = int(contact["peerUin"])
                is_group = contact.get("chatType") == 2
                if is_group:
                    ret = await self.bot.call_action(
                        action="get_group_msg_history",
                        group_id=peer,
                        count=OFFLINE_BACKLOG_HISTORY_COUNT,
                        self_id=self_id,
                    )
                elif contact.get("chatType") == 1:
                    ret = await self.bot.call_action(
                        action="get_friend_msg_history",
                        user_id=peer,
                        count=OFFLINE_BACKLOG_HISTORY_COUNT,
                        self_id=self_id,
                    )
                else:
                    continue

                history = sorted(
                    (ret or {}).get("messages") or [], key=lambda m: m.get("time", 0)
                )
                bot_message_ids = {
                    str(m.get("message_id"))
                    for m in history
                    if str(m.get("user_id")) == bot_id
                }
                pending: dict[str, list[dict]] = {}
                for m in history:
                    sender = str(m.get("user_id"))
                    if sender == bot_id:
                        pending.clear()
                    elif str(m.get("message_id")) in self._seen_message_ids:
                        pending.pop(sender, None)
                    elif int(m.get("time") or 0) > since and (
                        not is_group
                        or any(
                            (
                                seg.get("type") == "at"
                                and str(seg.get("data", {}).get("qq")) == bot_id
                            )
                            or (
                                seg.get("type") == "reply"
                                and str(seg.get("data", {}).get("id"))
                                in bot_message_ids
                            )
                            for seg in m.get("message") or []
                            if isinstance(seg, dict)
                        )
                    ):
                        pending.setdefault(sender, []).append(m)

                for messages in pending.values():
                    latest = messages[-1]
                    if str(latest.get("message_id")) in self._seen_message_ids:
                        continue
                    earlier = []
                    for m in messages[:-1]:
                        text = "".join(
                            seg.get("data", {}).get("text", "")
                            if seg.get("type") == "text"
                            else f"[{seg.get('type')}]"
                            for seg in m.get("message") or []
                            if isinstance(seg, dict)
                            and not (
                                seg.get("type") == "at"
                                and str(seg.get("data", {}).get("qq")) == bot_id
                            )
                        ).strip()
                        minutes = max(1, (int(time.time()) - int(m["time"])) // 60)
                        earlier.append(f"({minutes} min ago) {text}")

                    ev = Event.from_payload({**latest, "post_type": "message"})
                    if not ev:
                        continue
                    abm = await self.convert_message(ev)
                    if not abm:
                        continue
                    abm.timestamp = int(latest["time"])
                    await self.handle_msg(
                        abm,
                        offline_backlog={"sent_at": abm.timestamp, "earlier": earlier},
                    )
                    replayed += 1
                    await asyncio.sleep(1)
            except Exception as e:
                logger.warning(
                    f"aiocqhttp: offline backlog failed for {contact.get('peerUin')}: {e}"
                )

        if replayed:
            logger.info(
                f"aiocqhttp: replayed {replayed} offline message(s) for {bot_id}"
            )

    def get_client(self) -> CQHttp:
        return self.bot
