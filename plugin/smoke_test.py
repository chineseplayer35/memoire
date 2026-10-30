# -*- coding: utf-8 -*-
"""Smoke test for the bot2_memory plugin.

Runs the plugin's real storage, injection, distillation and trimming code
against a temporary memory directory and a fake provider. No network call and
no production data are touched. Run from the AstrBot root:

    python -X utf8 data/plugins/bot2_memory/smoke_test.py
"""

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ASTRBOT_ROOT = Path(__file__).resolve().parents[3]
if str(ASTRBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ASTRBOT_ROOT))

from astrbot.core.provider.entities import (  # noqa: E402
    LLMResponse,
    TokenUsage,
)

PLUGIN_PATH = Path(__file__).resolve().parent / "main.py"
UMO = "napcat2:GroupMessage:10001"
UMO_IDLE = "napcat2:GroupMessage:10002"
DAY = date.today().isoformat()


def load_plugin_module():
    """Import the plugin module without going through AstrBot's loader."""
    spec = importlib.util.spec_from_file_location("bot2_memory_smoke", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text_message(role: str, text: str) -> dict:
    """Build a stored message in the format AstrBot writes."""
    return {"role": role, "content": [{"type": "text", "text": text}]}


class FakeProvider:
    """Deterministic provider that records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.provider_config = {"id": "fake"}
        self.summaries = 0
        self.impressions = 0

    def get_model(self) -> str:
        return "fake-model"

    def meta(self):
        return SimpleNamespace(id="fake")

    async def text_chat(self, prompt=None, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt or "", "system": system_prompt or ""})
        if "压缩成当天的一条事记" in (prompt or ""):
            self.summaries += 1
            text = (
                f"第{self.summaries}版事记：大家聊了周末调试机器人的安排，"
                "约定周六下午带好电脑，晚饭一起吃。"
            )
        else:
            self.impressions += 1
            text = (
                f"第{self.impressions}版印象：我觉得他是个说话直接但可靠的人，"
                "聊技术时会认真解释，也会记得别人的偏好。"
            )
        return LLMResponse(
            role="assistant",
            completion_text=text,
            usage=TokenUsage(input_other=120, input_cached=30, output=40),
        )


class FakeConversation:
    """Minimal stand-in for conversation_manager.Conversation."""

    def __init__(self, umo: str, cid: str, history: list[dict], updated_at: float):
        self.user_id = umo
        self.cid = cid
        self.platform_id = "napcat2"
        self.history = json.dumps(history, ensure_ascii=False)
        self.updated_at = updated_at


class FakeConversationManager:
    """Minimal conversation manager backed by a dictionary."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], FakeConversation] = {}

    def add(self, umo: str, cid: str, history: list[dict], updated_at: float):
        conversation = FakeConversation(umo, cid, history, updated_at)
        self.items[(umo, cid)] = conversation
        return conversation

    async def get_conversations(self, *args, **kwargs):
        return list(self.items.values())

    async def get_conversation(self, umo: str, cid: str):
        return self.items.get((umo, cid))

    async def update_conversation(
        self,
        unified_msg_origin: str,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
        token_usage: int | None = None,
        **kwargs,
    ):
        conversation = self.items[(unified_msg_origin, conversation_id)]
        if history is not None:
            conversation.history = json.dumps(history, ensure_ascii=False)
        conversation.updated_at = time.time()


class FakeProviderManager:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    async def get_provider_by_id(self, provider_id: str):
        return self.provider if provider_id == "fake" else None


class FakeContext:
    def __init__(self, provider: FakeProvider, manager: FakeConversationManager):
        self.provider_manager = FakeProviderManager(provider)
        self.conversation_manager = manager
        self.provider = provider

    def get_using_provider(self, umo=None):
        return self.provider


class FakeEvent:
    def __init__(self, umo: str, qq: str, name: str, text: str = ""):
        self.unified_msg_origin = umo
        self._qq = qq
        self._name = name
        self.message_str = text
        self.message_obj = SimpleNamespace(self_id="999")

    def get_self_id(self) -> str:
        return "999"

    def get_sender_id(self) -> str:
        return self._qq

    def get_sender_name(self) -> str:
        return self._name

    def get_messages(self):
        return []

    def get_group_id(self) -> str:
        return "10001"

    def is_private_chat(self) -> bool:
        return False


def make_module(temp_root: Path):
    """Load the plugin module with every storage path redirected to temp_root."""
    module = load_plugin_module()
    module.MEMORY_DIR = temp_root / "memory"
    module.PEOPLE_DIR = module.MEMORY_DIR / "people"
    module.SESSIONS_DIR = module.MEMORY_DIR / "sessions"
    module.DIARY_PATH = module.MEMORY_DIR / "diary.md"
    module.PEOPLE_INDEX_PATH = module.MEMORY_DIR / "people_index.json"
    module.STATE_PATH = module.MEMORY_DIR / "state.json"
    module.USAGE_PATH = module.MEMORY_DIR / "usage.jsonl"
    module.CONFIG_PATH = temp_root / "config" / "bot2_memory.json"
    return module


async def main() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="bot2_memory_smoke_"))
    try:
        module = make_module(temp_root)
        provider = FakeProvider()
        manager = FakeConversationManager()
        manager.add(
            UMO,
            "cid-1",
            [
                text_message("user", "大家早"),
                text_message("assistant", "早呀"),
                text_message("user", "周末一起调试机器人吗"),
                text_message("assistant", "可以"),
            ],
            time.time(),
        )
        plugin = module.Bot2MemoryPlugin(FakeContext(provider, manager))

        # Replace the real DB writer; the usage log still exercises its code.
        async def no_provider_stat(**kwargs):
            return None

        plugin._record_provider_stat = no_provider_stat

        assert module.CONFIG_PATH.exists(), "config should be created"
        assert json.loads(module.CONFIG_PATH.read_text())["daily_time"] == "22:30"

        # Two people speak in the session.
        await plugin.track_message(FakeEvent(UMO, "111", "小A", "周末一起调试机器人吗"))
        await plugin.track_message(FakeEvent(UMO, "222", "小B", "我也来，我带零食"))
        state = json.loads(module.STATE_PATH.read_text())
        assert state["sessions"][UMO]["participants"], "participants should be saved"
        index = json.loads(module.PEOPLE_INDEX_PATH.read_text())
        assert index["111"]["nickname"] == "小A", [
            hex(ord(ch)) for ch in index["111"]["nickname"]
        ]
        assert index["222"]["nickname"] == "小B", [
            hex(ord(ch)) for ch in index["222"]["nickname"]
        ]

        # Daily run: one journal + two first impressions.
        await plugin._run_daily(DAY)
        entry_path = plugin._session_dir(UMO) / f"{DAY}.md"
        card_a = module.PEOPLE_DIR / "111.md"
        card_b = module.PEOPLE_DIR / "222.md"
        assert entry_path.exists(), "daily journal should be written"
        assert card_a.exists() and card_b.exists(), "cards should be created"
        assert len(provider.calls) == 3, f"expected 3 calls, got {len(provider.calls)}"
        state = json.loads(module.STATE_PATH.read_text())
        assert state["sessions"][UMO]["last_daily_run"] == DAY
        assert state["sessions"][UMO]["conversations"]["cid-1"]["cursor"] == 4
        first_entry = entry_path.read_text()
        assert "第1版事记" in first_entry

        # Second daily run without new messages must cost nothing.
        calls_before = len(provider.calls)
        await plugin._run_daily(DAY)
        assert len(provider.calls) == calls_before, (
            "no new messages must skip model calls"
        )

        # New messages: journal is rewritten and only the speaking person's
        # card is refreshed.
        conversation = manager.items[(UMO, "cid-1")]
        history = json.loads(conversation.history)
        history.extend(
            [
                text_message("user", "小A说他周六下午两点到"),
                text_message("assistant", "好，我把时间记下了"),
            ]
        )
        conversation.history = json.dumps(history, ensure_ascii=False)
        await plugin.track_message(FakeEvent(UMO, "111", "小A", "我周六下午两点到"))
        calls_before = len(provider.calls)
        await plugin._run_daily(DAY)
        assert len(provider.calls) == calls_before + 2, (
            f"expected summary + one impression, got "
            f"{len(provider.calls) - calls_before}"
        )
        assert "第2版事记" in entry_path.read_text(), "journal should be revised"
        assert len((card_a.read_text()).split("---")[-1].strip()) <= 300

        # Recent seven entries: write seven older ones and check the window.
        oldest_day = (date.today() - timedelta(days=7)).isoformat()
        for offset in range(7, 0, -1):
            day = (date.today() - timedelta(days=offset)).isoformat()
            module._atomic_write(
                plugin._session_dir(UMO) / f"{day}.md",
                f"# {day}\n更新时间：{day} 20:00\n\n{offset}天前的事记\n",
            )
        recent = plugin._recent_entry_texts(UMO)
        assert len(recent) == 7, f"expected 7 recent entries, got {len(recent)}"
        assert recent[0].startswith(DAY), recent[0]
        assert all(oldest_day not in item for item in recent), "oldest must be out"

        # Injection contains the journal and the cards of people in scene.
        request = SimpleNamespace(extra_user_content_parts=[], system_prompt="")
        await plugin.inject_memory(FakeEvent(UMO, "111", "小A", "周六见"), request)
        assert request.extra_user_content_parts, "injection should add a part"
        part = request.extra_user_content_parts[0]
        injected = part.text
        assert "最近事记" in injected and DAY in injected
        assert "小A（QQ 111）" in injected and "小B（QQ 222）" in injected
        assert getattr(part, "_no_save", False), "injected part must be temporary"

        # People window: after 16 senders, the oldest one drops out.
        for number in range(1001, 1017):
            await plugin.track_message(
                FakeEvent(UMO, str(number), f"P{number}", f"消息{number}")
            )
            plugin._write_card(str(number), f"P{number}", f"对P{number}的印象")
        request = SimpleNamespace(extra_user_content_parts=[], system_prompt="")
        await plugin.inject_memory(FakeEvent(UMO, "1016", "P1016", "最后一句"), request)
        injected = request.extra_user_content_parts[0].text
        assert "P1016（QQ 1016）" in injected
        assert "P1001（QQ 1001）" not in injected, "outside 15-message window"

        # Tools.
        read = await plugin.memory_read_impression(FakeEvent(UMO, "111", "小A"), "小A")
        assert "印象" in read and "第" in read, read
        missing = await plugin.memory_read_impression(
            FakeEvent(UMO, "111", "小A"), "查无此人"
        )
        assert missing == "没有此人的印象。", missing
        entries = await plugin.memory_read_entries(
            FakeEvent(UMO, "111", "小A"), DAY, DAY
        )
        assert "第2版事记" in entries, entries
        assert module.DIARY_PATH.is_relative_to(temp_root)
        written = await plugin.memory_write_diary(
            FakeEvent(UMO, "111", "小A"), "周六下午一起调试机器人"
        )
        assert written == "已记入日记。", written
        diary = await plugin.memory_read_diary(FakeEvent(UMO, "111", "小A"))
        assert "周六下午一起调试机器人" in diary, diary

        calls_before = len(provider.calls)
        updated = await plugin.memory_update_impression(
            FakeEvent(UMO, "111", "小A"),
            "222",
            "他开始主动帮大家带东西",
            "今天主动说带零食",
        )
        assert "已更新对" in updated, updated
        assert len(provider.calls) == calls_before + 1
        assert "第" in card_b.read_text()
        missing_update = await plugin.memory_update_impression(
            FakeEvent(UMO, "111", "小A"), "查无此人", "变化", "原因"
        )
        assert "找不到这个人" in missing_update, missing_update

        # Daily trigger: not due before 22:30, due once after.
        plugin._daily_ran_date = None
        assert not plugin._daily_due(datetime(2026, 10, 26, 22, 29))
        assert plugin._daily_due(datetime(2026, 10, 26, 22, 30))
        plugin._daily_ran_date = "2026-10-26"
        assert not plugin._daily_due(datetime(2026, 10, 26, 23, 30))

        # Idle scan: distill first, then trim 60 messages to 50.
        idle_history = [
            text_message("user" if index % 2 == 0 else "assistant", f"旧消息 {index}")
            for index in range(60)
        ]
        manager.add(UMO_IDLE, "cid-idle", idle_history, time.time() - 5 * 3600)
        await plugin.track_message(
            FakeEvent(UMO_IDLE, "333", "小C", "四小时前的最后一句")
        )
        calls_before = len(provider.calls)
        await plugin._idle_scan()
        trimmed = json.loads(manager.items[(UMO_IDLE, "cid-idle")].history)
        assert len(trimmed) == 50, (
            f"expected 50 messages after trim, got {len(trimmed)}"
        )
        purposes = [
            json.loads(line)["purpose"]
            for line in module.USAGE_PATH.read_text().splitlines()
        ]
        assert purposes.count("session_summary") >= 1
        assert len(provider.calls) > calls_before, "idle scan must distill before trim"
        idle_state = json.loads(module.STATE_PATH.read_text())
        idle_cursor = idle_state["sessions"][UMO_IDLE]["conversations"]["cid-idle"]
        assert idle_cursor["cursor"] == 50, idle_cursor

        # Usage log must carry tokens and purpose for every call.
        records = [
            json.loads(line) for line in module.USAGE_PATH.read_text().splitlines()
        ]
        assert len(records) == len(provider.calls), (len(records), len(provider.calls))
        assert all(record["input_other"] == 120 for record in records)
        assert all(record["output"] == 40 for record in records)
        assert all(record["status"] == "completed" for record in records)
        print("SMOKE OK")
        print(f"provider calls: {len(provider.calls)}")
        purpose_counts = {purpose: purposes.count(purpose) for purpose in set(purposes)}
        print(f"purposes: {purpose_counts}")
        print(
            f"entries: {sorted(path.name for path in plugin._session_dir(UMO).glob('*.md'))}"
        )
        print(f"cards: {sorted(path.name for path in module.PEOPLE_DIR.glob('*.md'))}")
        print(
            f"state sessions: {sorted(json.loads(module.STATE_PATH.read_text())['sessions'])}"
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
