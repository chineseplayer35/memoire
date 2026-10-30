"""Lightweight long-term memory for bot2.

Three layers, all stored as small Markdown/JSON files under
``data/memory``:

1. One impression card per person who has interacted with the bot
   (``people/<qqid>.md``, at most 240 characters), indexed by
   ``people_index.json`` for nickname lookups.
2. One short session journal per day (``sessions/<safe umo>/YYYY-MM-DD.md``),
   rewritten a few times a day instead of appended forever.
3. The live context: after four idle hours the journal and impressions are
   distilled from the full AstrBot conversation history first, then the
   history is trimmed to 50 messages.

Background distillation calls the provider only when a session has messages
since its last distillation. Every such call is written to ``usage.jsonl``
and to ``provider_stats`` so cost_guard and the dashboard can see it.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import Message, TextPart

DATA_DIR = Path(__file__).resolve().parents[2]
MEMORY_DIR = DATA_DIR / "memory"
PEOPLE_DIR = MEMORY_DIR / "people"
SESSIONS_DIR = MEMORY_DIR / "sessions"
DIARY_PATH = MEMORY_DIR / "diary.md"
DIARY_INJECT_COUNT = 10
PEOPLE_INDEX_PATH = MEMORY_DIR / "people_index.json"
STATE_PATH = MEMORY_DIR / "state.json"
USAGE_PATH = MEMORY_DIR / "usage.jsonl"
CONFIG_PATH = DATA_DIR / "config" / "bot2_memory.json"

DEFAULT_CONFIG: dict = {
    "enabled": True,
    "daily_time": "22:30",
    "idle_hours": 4,
    "keep_messages": 50,
    "recent_entries": 7,
    "people_window": 15,
    "impression_max_chars": 240,
    "summary_provider": "",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
IDLE_SCAN_INTERVAL_SECONDS = 300
DAILY_CHECK_INTERVAL_SECONDS = 60
PARTICIPANT_RETENTION_DAYS = 14
# Bounds the context sent to a distillation call without cutting the newest
# messages: a single message is shortened, the day's whole stream is kept.
MAX_MESSAGE_CHARS = 1000
MAX_ENTRY_CHARS = 500


def _atomic_write(path: Path, text: str) -> None:
    """Write text to ``path`` through a temporary file in the same directory.

    Args:
        path: Destination file.
        text: Full new content.

    Raises:
        OSError: If the directory, temporary file, or rename fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _safe_dir_name(umo: str) -> str:
    """Return a filesystem-safe, collision-free directory name for a session.

    Args:
        umo: Unified message origin.

    Returns:
        A readable ASCII prefix plus a short hash of the original umo.
    """
    prefix = re.sub(r"[^0-9A-Za-z._-]+", "_", umo).strip("._-")[:64]
    digest = hashlib.sha1(umo.encode("utf-8")).hexdigest()[:8]
    return f"{prefix or 'session'}_{digest}"


def _msg_anchor(message: Any) -> str:
    """Return a short stable fingerprint for one history message."""
    raw = json.dumps(message, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parse_history(raw: Any) -> list[dict]:
    """Parse the JSON history string stored by AstrBot.

    Args:
        raw: ``conversation.history`` value, normally a JSON list string.

    Returns:
        A list of message dictionaries, empty when the value is unusable.
    """
    try:
        history = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, dict)]


def _content_text(content: Any) -> str:
    """Extract visible text from a stored message content value.

    Think parts, tool calls and non-text parts are skipped. Text parts are
    joined with newlines, matching how the agent sees a user message.

    Args:
        content: A string or a list of content-part dictionaries.

    Returns:
        The visible text, or an empty string.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)


def _format_history(messages: list[dict]) -> str:
    """Format stored messages as a user/assistant transcript for the model.

    Args:
        messages: Stored message dictionaries.

    Returns:
        A transcript with ``用户：`` / ``你：`` prefixes.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_text(message.get("content")).strip()
        if not text:
            continue
        label = "用户" if role == "user" else "你"
        lines.append(f"{label}：{text[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines)


def _history_since(history: list[dict], cursor: dict, recent_people: list[dict] | None = None) -> tuple[list[dict], int]:
    """Return messages added since the previous distillation.

    The history list has no per-message timestamps. Instead, the state stores
    the processed message count plus the fingerprint of the last processed
    message. A normal run appends to the list, so the cursor is exact. When a
    framework compression shortened the list, the stored anchor is looked up
    and everything after it is new. If the anchor is gone, only messages from
    today 08:00 onwards are distilled (estimated via recent_people); with no
    chat today, the run is skipped. Never an all-in fallback.

    Args:
        history: Current raw history list.
        cursor: Stored ``{"cursor": int, "anchor": str}`` dictionary.

    Returns:
        ``(new messages, start index)``.
    """
    cursor_index = int(cursor.get("cursor", 0) or 0)
    anchor = str(cursor.get("anchor", "") or "")
    if cursor_index <= 0:
        return history, 0
    if cursor_index <= len(history):
        if not anchor or _msg_anchor(history[cursor_index - 1]) == anchor:
            return history[cursor_index:], cursor_index
    if anchor:
        for index in range(len(history) - 1, -1, -1):
            if _msg_anchor(history[index]) == anchor:
                return history[index + 1 :], index + 1
    # Anchor lost: the all-in fallback is banned. Instead, distill only the
    # messages from today 08:00 onwards (estimated via the session's own
    # timestamped recent_people notes). If there was no chat today, skip.
    today_8am = datetime.combine(date.today(), datetime.min.time()).timestamp() + 8 * 3600
    recent = recent_people or []
    today_count = sum(1 for item in recent if float(item.get("ts", 0)) >= today_8am)
    if today_count == 0:
        logger.info("bot2_memory: cursor lost, but no chat since 08:00; skip distillation")
        return [], 0
    # History also contains bot replies (~2x the inbound notes), so cover
    # roughly the same span. Bounded by today's activity, never all-in.
    start = max(0, len(history) - 2 * today_count)
    logger.warning(
        "bot2_memory: cursor lost; distilling only the last %d messages (today 08:00+)",
        len(history) - start,
    )
    return history[start:], start


def _entry_body(text: str) -> str:
    """Strip the journal file header and return the entry body."""
    lines = text.splitlines()
    body: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped.startswith("# "):
            continue
        if stripped.startswith("更新时间："):
            continue
        if stripped:
            body.append(line)
    return "\n".join(body).strip()


def _clip(text: str, limit: int) -> str:
    """Trim model output to ``limit`` characters at a sentence boundary.

    Args:
        text: Raw model output.
        limit: Maximum number of characters to keep.

    Returns:
        The clipped text, never longer than ``limit`` characters.
    """
    text = text.strip().strip('"“”')
    for prefix in ("印象卡：", "印象卡:", "更新后的印象卡：", "更新后的印象卡:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip()
            break
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(mark) for mark in "。！？；\n")
    if boundary >= int(limit * 0.6):
        return cut[: boundary + 1]
    return cut[: limit - 1].rstrip() + "…"


def _front_matter_name(text: str) -> str:
    """Return the nickname from a card's front matter, if present."""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    if end == -1:
        return ""
    for line in text[3:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "nickname":
            return value.strip()
    return ""


def _card_body(text: str) -> str:
    """Return the card body without its front matter."""
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + len("\n---") :].lstrip("\n").strip()


@register("bot2_memory", "yqb", "轻量长期记忆：印象卡、会话事记与上下文提炼", "1.0.0")
class Bot2MemoryPlugin(Star):
    """Maintain impression cards, daily journals, and bounded context."""

    def __init__(self, context: Context) -> None:
        """Load config and state, then prepare in-memory recent-person windows.

        Args:
            context: AstrBot plugin context.
        """
        super().__init__(context)
        self.enabled = DEFAULT_CONFIG["enabled"]
        self.daily_time = DEFAULT_CONFIG["daily_time"]
        self.idle_hours = DEFAULT_CONFIG["idle_hours"]
        self.keep_messages = DEFAULT_CONFIG["keep_messages"]
        self.recent_entries = DEFAULT_CONFIG["recent_entries"]
        self.people_window = DEFAULT_CONFIG["people_window"]
        self.impression_max_chars = DEFAULT_CONFIG["impression_max_chars"]
        self.summary_provider = DEFAULT_CONFIG["summary_provider"]
        self._config_stat: tuple[float, int] | None = None
        self._invalid_stat: tuple[float, int] | None = None

        # qq -> [(timestamp, umo, text)] used as today's raw material when
        # rewriting an impression card.
        self._person_notes: dict[str, list[tuple[float, str, str]]] = {}

        self._daily_ran_date: str | None = None
        self._warned_unknown_provider = False
        self._daily_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._maintenance_lock = asyncio.Lock()
        self._impression_lock = asyncio.Lock()

        PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._refresh_config()
        self._state = self._load_state()

        self.logger.info(
            "bot2_memory: registry ready at %s, config %s (enabled=%s)",
            MEMORY_DIR,
            CONFIG_PATH,
            self.enabled,
        )
        self.logger.info(
            "bot2_memory: registered LLM tools memory_read_impression, "
            "memory_read_entries, memory_update_impression",
        )

    # ---------------------------------------------------------------- config

    def _refresh_config(self) -> None:
        """Reload ``bot2_memory.json`` when it is missing or changed.

        A missing file is created from ``DEFAULT_CONFIG``. A malformed value
        falls back to the built-in defaults and is reported once per file
        version, so a broken edit cannot turn missing fields into crashes.
        """
        try:
            stat = CONFIG_PATH.stat()
        except FileNotFoundError:
            stat = None
        except OSError as exc:
            self.logger.warning("bot2_memory: cannot stat %s: %s", CONFIG_PATH, exc)
            return

        signature = (stat.st_mtime, stat.st_size) if stat is not None else None
        if signature is not None and signature == self._config_stat:
            return

        if signature is None:
            try:
                _atomic_write(
                    CONFIG_PATH,
                    json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n",
                )
                stat = CONFIG_PATH.stat()
            except OSError as exc:
                self.logger.warning(
                    "bot2_memory: cannot create %s: %s", CONFIG_PATH, exc
                )
                return
            self._config_stat = (stat.st_mtime, stat.st_size)
            self._invalid_stat = None
            self.logger.info("bot2_memory: created %s with defaults", CONFIG_PATH)
            return

        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise TypeError("top level must be an object")
            enabled = data.get("enabled", DEFAULT_CONFIG["enabled"])
            if not isinstance(enabled, bool):
                raise TypeError("'enabled' must be a boolean")
            daily_time = str(data.get("daily_time", DEFAULT_CONFIG["daily_time"]))
            if not TIME_RE.fullmatch(daily_time):
                raise ValueError("'daily_time' must be HH:MM")
            idle_hours = float(data.get("idle_hours", DEFAULT_CONFIG["idle_hours"]))
            if idle_hours <= 0:
                raise ValueError("'idle_hours' must be positive")
            keep_messages = data.get("keep_messages", DEFAULT_CONFIG["keep_messages"])
            recent_entries = data.get(
                "recent_entries", DEFAULT_CONFIG["recent_entries"]
            )
            people_window = data.get("people_window", DEFAULT_CONFIG["people_window"])
            impression_max_chars = data.get(
                "impression_max_chars", DEFAULT_CONFIG["impression_max_chars"]
            )
            for key, value in (
                ("keep_messages", keep_messages),
                ("recent_entries", recent_entries),
                ("people_window", people_window),
                ("impression_max_chars", impression_max_chars),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"'{key}' must be an integer")
                if value <= 0:
                    raise ValueError(f"'{key}' must be positive")
            summary_provider = str(
                data.get("summary_provider", DEFAULT_CONFIG["summary_provider"])
            ).strip()
        except (OSError, ValueError, TypeError) as exc:
            self.enabled = DEFAULT_CONFIG["enabled"]
            self.daily_time = DEFAULT_CONFIG["daily_time"]
            self.idle_hours = DEFAULT_CONFIG["idle_hours"]
            self.keep_messages = DEFAULT_CONFIG["keep_messages"]
            self.recent_entries = DEFAULT_CONFIG["recent_entries"]
            self.people_window = DEFAULT_CONFIG["people_window"]
            self.impression_max_chars = DEFAULT_CONFIG["impression_max_chars"]
            self.summary_provider = DEFAULT_CONFIG["summary_provider"]
            if signature != self._invalid_stat:
                self._invalid_stat = signature
                self.logger.error(
                    "bot2_memory: invalid config %s (%s); using defaults",
                    CONFIG_PATH,
                    exc,
                )
            return

        self._config_stat = signature
        self._invalid_stat = None
        self.enabled = enabled
        self.daily_time = daily_time
        self.idle_hours = idle_hours
        self.keep_messages = int(keep_messages)
        self.recent_entries = int(recent_entries)
        self.people_window = int(people_window)
        self.impression_max_chars = int(impression_max_chars)
        self.summary_provider = summary_provider
        self.logger.info(
            "bot2_memory: loaded config enabled=%s daily_time=%s idle_hours=%.2f "
            "keep_messages=%d recent_entries=%d people_window=%d "
            "impression_max_chars=%d summary_provider=%r",
            self.enabled,
            self.daily_time,
            self.idle_hours,
            self.keep_messages,
            self.recent_entries,
            self.people_window,
            self.impression_max_chars,
            self.summary_provider or "(session)",
        )

    # ----------------------------------------------------------------- state

    def _load_state(self) -> dict:
        """Load ``state.json``, falling back to an empty valid structure."""
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        sessions = data.get("sessions")
        people = data.get("people")
        return {
            "version": 1,
            "sessions": sessions if isinstance(sessions, dict) else {},
            "people": people if isinstance(people, dict) else {},
        }

    def _session_state(self, umo: str) -> dict:
        """Return (creating if needed) the state dictionary for one session."""
        session = self._state["sessions"].setdefault(umo, {})
        if not isinstance(session, dict):
            session = {}
            self._state["sessions"][umo] = session
        session.setdefault("last_summary_ts", 0.0)
        session.setdefault("last_daily_run", "")
        session.setdefault("last_activity_ts", 0.0)
        session.setdefault("participants", {})
        session.setdefault("conversations", {})
        session.setdefault("recent_people", [])
        return session

    def _person_state(self, qq: str) -> dict:
        """Return (creating if needed) the state dictionary for one person."""
        person = self._state["people"].setdefault(qq, {})
        if not isinstance(person, dict):
            person = {}
            self._state["people"][qq] = person
        person.setdefault("last_seen_ts", 0.0)
        person.setdefault("last_impression_ts", 0.0)
        return person

    def _save_state(self) -> None:
        """Persist state atomically and prune old participant dates."""
        cutoff = date.today() - timedelta(days=PARTICIPANT_RETENTION_DAYS)
        for session in self._state["sessions"].values():
            if not isinstance(session, dict):
                continue
            participants = session.get("participants")
            if not isinstance(participants, dict):
                continue
            for day in list(participants):
                if not DATE_RE.fullmatch(str(day)):
                    participants.pop(day, None)
                    continue
                try:
                    if date.fromisoformat(str(day)) < cutoff:
                        participants.pop(day, None)
                except ValueError:
                    participants.pop(day, None)
        try:
            _atomic_write(
                STATE_PATH,
                json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            self.logger.warning("bot2_memory: failed to save state: %s", exc)

    # -------------------------------------------------------- paths/indexing

    def _session_dir(self, umo: str) -> Path:
        """Return the journal directory for one session."""
        return SESSIONS_DIR / _safe_dir_name(umo)

    def _card_path(self, qq: str) -> Path:
        """Return the impression-card path for one QQ number."""
        return PEOPLE_DIR / f"{qq}.md"

    def _read_index(self) -> dict:
        """Read ``people_index.json``, returning an empty mapping on failure."""
        try:
            data = json.loads(PEOPLE_INDEX_PATH.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(qq): info for qq, info in data.items() if isinstance(info, dict)}

    def _update_index(
        self, qq: str, name: str, aliases: list[str] | None = None
    ) -> None:
        """Insert or refresh one person in ``people_index.json``.

        Only nickname changes are written, so a chatty group does not rewrite
        the index on every message.

        Args:
            qq: QQ number as a string.
            name: Current group card or nickname.
            aliases: Optional extra names for lookup.
        """
        index = self._read_index()
        info = index.get(qq)
        changed = False
        if not isinstance(info, dict):
            info = {"nickname": name, "aliases": []}
            changed = True
        else:
            info = dict(info)
            old_name = str(info.get("nickname") or "").strip()
            if name and old_name != name:
                info["nickname"] = name
                if old_name:
                    current = [
                        str(alias) for alias in info.get("aliases", []) if str(alias)
                    ]
                    if old_name not in current:
                        current.append(old_name)
                        info["aliases"] = current[-5:]
                changed = True
            if not isinstance(info.get("aliases"), list):
                info["aliases"] = []
                changed = True
        if aliases:
            current = [str(alias) for alias in info.get("aliases", []) if str(alias)]
            for alias in aliases:
                if alias and alias not in current:
                    current.append(alias)
                    changed = True
            info["aliases"] = current
        if not changed:
            return
        index[qq] = info
        try:
            _atomic_write(
                PEOPLE_INDEX_PATH,
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            self.logger.warning("bot2_memory: failed to save people index: %s", exc)

    @staticmethod
    def _name_matches(info: dict, needle: str, *, exact: bool) -> bool:
        """Check one index entry against a case-folded name."""
        names = [str(info.get("nickname") or "")]
        aliases = info.get("aliases")
        if isinstance(aliases, list):
            names.extend(str(alias) for alias in aliases)
        for name in names:
            folded = name.strip().casefold()
            if not folded:
                continue
            if folded == needle or (
                not exact and len(needle) >= 2 and needle in folded
            ):
                return True
        return False

    def _resolve_person(
        self, name_or_qq: str, event: AstrMessageEvent | None = None
    ) -> tuple[str | None, str, list[str]]:
        """Resolve a nickname or QQ number to a QQ id.

        Args:
            name_or_qq: Raw tool argument.
            event: Current event, used to name a brand-new QQ id.

        Returns:
            ``(qq, display name, ambiguous qq list)``. ``qq`` is None when the
            person cannot be resolved.
        """
        raw = str(name_or_qq or "").strip()
        if not raw:
            return None, "", []
        index = self._read_index()
        if raw.isdigit():
            info = index.get(raw)
            name = str((info or {}).get("nickname") or "").strip()
            if not name and event is not None:
                if str(event.get_sender_id()) == raw:
                    name = str(event.get_sender_name() or "").strip()
            return raw, name or raw, []
        needle = raw.casefold()
        exact = [
            qq
            for qq, info in index.items()
            if self._name_matches(info, needle, exact=True)
        ]
        if len(exact) == 1:
            qq = exact[0]
            return qq, str(index[qq].get("nickname") or raw), []
        if len(exact) > 1:
            return None, raw, exact
        partial = [
            qq
            for qq, info in index.items()
            if self._name_matches(info, needle, exact=False)
        ]
        if len(partial) == 1:
            qq = partial[0]
            return qq, str(index[qq].get("nickname") or raw), []
        if len(partial) > 1:
            return None, raw, partial
        # The current sender may not be in the index yet (for example a
        # nickname passed by the model right after a restart).
        if event is not None:
            sender = str(event.get_sender_id() or "")
            sender_name = str(event.get_sender_name() or "").strip()
            if sender and sender_name.casefold() == needle:
                return sender, sender_name, []
        return None, raw, []

    @staticmethod
    def _ambiguous_message(names: dict, candidates: list[str]) -> str:
        """Build a short disambiguation message for an ambiguous nickname."""
        labels = []
        for qq in candidates[:5]:
            info = names.get(qq) or {}
            labels.append(f"{info.get('nickname') or '未知'}（{qq}）")
        return "匹配到多个人：" + "、".join(labels) + "。请用 QQ 号指定。"

    # -------------------------------------------------------------- tracking

    @filter.event_message_type(
        EventMessageType.ALL,
        priority=90,
    )
    async def track_message(self, event: AstrMessageEvent) -> None:
        """Remember the sender, today's participant, and recent message text.

        This runs after session_guard and cost_guard, so dropped messages are
        not recorded. Only in-memory windows and a small state file are
        touched; no model call happens on the hot path.

        Args:
            event: Incoming group or private message event.
        """
        if not self.enabled:
            return
        self_id = str(event.get_self_id() or "")
        sender = str(event.get_sender_id() or "")
        if not sender or sender == self_id:
            return
        umo = event.unified_msg_origin
        name = str(event.get_sender_name() or "").strip() or sender
        now = time.time()

        text = str(event.message_str or "").strip()
        if text:
            notes = self._person_notes.setdefault(sender, [])
            notes.append((now, umo, text[:300]))
            del notes[:-20]

        session = self._session_state(umo)
        session["last_activity_ts"] = now
        recent = session["recent_people"]
        if not isinstance(recent, list):
            recent = []
            session["recent_people"] = recent
        recent.append({"ts": now, "qq": sender, "name": name})
        del recent[:-50]
        today = date.today().isoformat()
        participants = session["participants"].setdefault(today, {})
        if not isinstance(participants, dict):
            participants = {}
            session["participants"][today] = participants
        participants[sender] = name

        person = self._person_state(sender)
        person["last_seen_ts"] = now
        self._save_state()
        self._update_index(sender, name)

    # ------------------------------------------------------------- injection

    def _recent_entry_texts(self, umo: str) -> list[str]:
        """Return the latest journal entries for one session, newest first."""
        directory = self._session_dir(umo)
        if not directory.is_dir():
            return []
        files = [
            path for path in directory.glob("*.md") if DATE_RE.fullmatch(path.stem)
        ]
        entries: list[str] = []
        for path in sorted(files, key=lambda item: item.stem, reverse=True)[
            : self.recent_entries
        ]:
            try:
                body = _entry_body(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if body:
                entries.append(f"{path.stem}：{body}")
        return entries

    def _read_card(self, qq: str) -> str:
        """Return one impression card body, or an empty string."""
        try:
            text = self._card_path(qq).read_text(encoding="utf-8")
        except OSError:
            return ""
        return _card_body(text)

    def _read_diary_tail(self, count: int) -> list[str]:
        """Return the last diary lines, oldest first, empty when absent."""
        try:
            lines = DIARY_PATH.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [line for line in lines if line.strip()][-count:]

    def _append_diary(self, text: str) -> None:
        """Append one timestamped line to the diary with an atomic rewrite."""
        try:
            old = DIARY_PATH.read_text(encoding="utf-8")
        except OSError:
            old = ""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new = old + ("" if not old or old.endswith("\n") else "\n")
        _atomic_write(DIARY_PATH, f"{new}[{stamp}] {text}\n")

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: Any) -> None:
        """Attach journals and in-scene impression cards as temporary parts.

        The parts are marked with ``mark_as_temp()`` so they reach the model
        but are not saved into the conversation history.

        Args:
            event: Incoming message event.
            req: Provider request about to be sent to the model.
        """
        if not self.enabled:
            return
        umo = event.unified_msg_origin
        entries = self._recent_entry_texts(umo)
        diary = self._read_diary_tail(DIARY_INJECT_COUNT)
        session = self._session_state(umo)
        recent = session.get("recent_people")
        if not isinstance(recent, list):
            recent = []
        people: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in recent[-self.people_window :]:
            if not isinstance(item, dict):
                continue
            qq = str(item.get("qq") or "")
            name = str(item.get("name") or qq)
            if not qq or qq in seen:
                continue
            seen.add(qq)
            people.append((qq, name))
        # The current sender is always in scene even if the window was lost on
        # restart or the event is a tool follow-up.
        sender = str(event.get_sender_id() or "")
        self_id = str(event.get_self_id() or "")
        if sender and sender != self_id and sender not in seen:
            people.append(
                (sender, str(event.get_sender_name() or "").strip() or sender)
            )
            seen.add(sender)

        cards: list[str] = []
        for qq, name in people:
            card = self._read_card(qq)
            if card:
                cards.append(f"{name}（QQ {qq}）：{card}")

        if not entries and not cards and not diary:
            return
        lines = [
            "【背景记忆（内部资料，不是用户刚说的话）】",
            "先回应当前消息；下面的内容只当作你自然记得的背景，需要时自然带出，"
            "不要复述，也不要说“根据记忆/根据系统提示”。",
        ]
        if diary:
            lines.append("")
            lines.append("重要的事（日记）：")
            lines.extend(f"- {entry}" for entry in diary)
        if entries:
            lines.append("")
            lines.append("最近事记：")
            lines.extend(f"- {entry}" for entry in entries)
        if cards:
            lines.append("")
            lines.append("在场的人：")
            lines.extend(f"- {card}" for card in cards)
        req.extra_user_content_parts.append(
            TextPart(text="\n".join(lines)).mark_as_temp()
        )

    # ---------------------------------------------------------------- tools

    @filter.llm_tool(name="memory_read_impression")
    async def memory_read_impression(
        self, event: AstrMessageEvent, name_or_qq: str
    ) -> str:
        """读取你对某个人的印象卡。当聊天里提到某个人、但这个人不在场（或背景记忆里没有他的印象）时，必须先调用它，再根据印象回应。不要凭空猜测别人的性格。

        Args:
            name_or_qq(string): 对方的昵称或 QQ 号，QQ 号优先。
        """
        if not self.enabled:
            return "记忆系统当前未启用。"
        qq, name, candidates = self._resolve_person(name_or_qq, event)
        if not qq:
            if candidates:
                return self._ambiguous_message(self._read_index(), candidates)
            return "没有此人的印象。"
        card = self._read_card(qq)
        if not card:
            return "没有此人的印象。"
        return f"{name}（QQ {qq}）的印象：\n{card}"

    @filter.llm_tool(name="memory_read_entries")
    async def memory_read_entries(
        self, event: AstrMessageEvent, start_date: str, end_date: str
    ) -> str:
        """按日期范围读取本会话的旧事记。最近7条事记已经自动放进你的背景记忆，不用重复读；只有需要翻更早的具体细节时才调用。

        Args:
            start_date(string): 开始日期，格式 YYYY-MM-DD。
            end_date(string): 结束日期，格式 YYYY-MM-DD，包含当天。
        """
        if not self.enabled:
            return "记忆系统当前未启用。"
        try:
            start = date.fromisoformat(str(start_date).strip())
            end = date.fromisoformat(str(end_date).strip())
        except ValueError:
            return "日期格式不对，请用 YYYY-MM-DD。"
        if start > end:
            return "开始日期不能晚于结束日期。"
        if (end - start).days > 60:
            return "日期范围太长，请缩小到 60 天以内。"
        umo = event.unified_msg_origin
        directory = self._session_dir(umo)
        chunks: list[str] = []
        cursor = start
        while cursor <= end:
            path = directory / f"{cursor.isoformat()}.md"
            try:
                body = _entry_body(path.read_text(encoding="utf-8"))
            except OSError:
                body = ""
            if body:
                chunks.append(f"{cursor.isoformat()}：{body}")
            cursor += timedelta(days=1)
        if not chunks:
            return "这个日期范围没有事记。"
        return "\n\n".join(chunks)[:8000]

    @filter.llm_tool(name="memory_update_impression")
    async def memory_update_impression(
        self,
        event: AstrMessageEvent,
        name_or_qq: str,
        change: str,
        reason: str,
    ) -> str:
        """记录你对某个人的印象变化，并立即更新他的印象卡。当某人的言行让你对他产生新的好印象或坏印象、你了解到他的性格/偏好/重要事实、或你们的关系发生变化时，主动调用。不要用于琐碎闲聊。

        Args:
            name_or_qq(string): 对方的昵称或 QQ 号，QQ 号优先。
            change(string): 新的印象变化，一两句话说清。
            reason(string): 变化的原因或依据。
        """
        if not self.enabled:
            return "记忆系统当前未启用。"
        qq, name, candidates = self._resolve_person(name_or_qq, event)
        if not qq:
            if candidates:
                return self._ambiguous_message(self._read_index(), candidates)
            return "找不到这个人。如果对方还没有昵称记录，请直接提供 QQ 号。"
        change = str(change or "").strip()
        reason = str(reason or "").strip()
        if not change:
            return "没有提供印象变化，未更新。"
        async with self._impression_lock:
            old = self._read_card(qq)
            new = await self._rewrite_impression(
                qq,
                name,
                old,
                change,
                reason=reason,
                umo=event.unified_msg_origin,
                purpose="impression_tool",
            )
            if not new:
                return "印象更新失败，请稍后再试。"
            self._write_card(qq, name, new)
            person = self._person_state(qq)
            person["last_impression_ts"] = time.time()
            self._save_state()
            self._update_index(qq, name)
        return f"已更新对{name}的印象：\n{new}"

    @filter.llm_tool(name="memory_write_diary")
    async def memory_write_diary(self, event: AstrMessageEvent, text: str) -> str:
        """把重要的事写进你的日记（长期记忆）。用于：重要的约定/承诺、谁对你好或欺负你、重要的情绪节点、值得长期记住的事。不要记录琐碎闲聊或每天的流水账（那些由事记自动整理）。

        Args:
            text(string): 一条简短的重要记录，一两句话。
        """
        if not self.enabled:
            return "记忆系统当前未启用。"
        text = str(text or "").strip()
        if not text:
            return "没有内容，未写入。"
        self._append_diary(text[:300])
        return "已记入日记。"

    @filter.llm_tool(name="memory_read_diary")
    async def memory_read_diary(self, event: AstrMessageEvent, limit: str = "30") -> str:
        """读取你的日记（重要的事）。最近10条已自动在背景记忆里，需要回忆更早的重要事件时调用。

        Args:
            limit(string): 最多读取多少条，默认 30，最大 100。
        """
        if not self.enabled:
            return "记忆系统当前未启用。"
        try:
            n = max(1, min(int(str(limit or "30").strip()), 100))
        except (TypeError, ValueError):
            n = 30
        entries = self._read_diary_tail(n)
        if not entries:
            return "日记还是空的。"
        return "\n".join(entries)

    # --------------------------------------------------------- model calling

    async def _get_provider(self, umo: str | None) -> Any | None:
        """Resolve the configured summary provider, falling back per session."""
        if self.summary_provider:
            try:
                provider = await self.context.provider_manager.get_provider_by_id(
                    self.summary_provider
                )
            except Exception as exc:
                provider = None
                if not self._warned_unknown_provider:
                    self._warned_unknown_provider = True
                    self.logger.warning(
                        "bot2_memory: cannot resolve summary provider %r: %s",
                        self.summary_provider,
                        exc,
                    )
            if provider is not None:
                return provider
            if not self._warned_unknown_provider:
                self._warned_unknown_provider = True
                self.logger.warning(
                    "bot2_memory: summary provider %r not found, using the "
                    "session provider",
                    self.summary_provider,
                )
        try:
            return self.context.get_using_provider(umo)
        except Exception as exc:
            self.logger.warning("bot2_memory: no provider available: %s", exc)
            return None

    @staticmethod
    def _usage_field(usage: Any, key: str) -> int:
        """Read one token count from an object or dictionary usage."""
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(key, 0) or 0)
        return int(getattr(usage, key, 0) or 0)

    def _log_usage(self, record: dict) -> None:
        """Append one usage record to ``usage.jsonl`` atomically."""
        try:
            existing = (
                USAGE_PATH.read_text(encoding="utf-8") if USAGE_PATH.exists() else ""
            )
            _atomic_write(
                USAGE_PATH,
                existing + json.dumps(record, ensure_ascii=False) + "\n",
            )
        except OSError as exc:
            self.logger.warning("bot2_memory: failed to write usage log: %s", exc)

    async def _record_provider_stat(
        self,
        *,
        umo: str,
        provider_id: str,
        provider_model: str,
        usage: dict,
        start_time: float,
        end_time: float,
        status: str,
    ) -> None:
        """Write one background call into ``provider_stats`` for cost_guard."""
        try:
            from astrbot.core import db_helper

            await db_helper.insert_provider_stat(
                umo=umo or "bot2_memory",
                provider_id=provider_id,
                provider_model=provider_model,
                conversation_id=None,
                status=status,
                stats={
                    "token_usage": usage,
                    "start_time": start_time,
                    "end_time": end_time,
                    "time_to_first_token": 0.0,
                },
                agent_type="internal",
            )
        except Exception as exc:
            self.logger.debug("bot2_memory: provider stat not recorded (%s)", exc)

    async def _call_llm(
        self,
        *,
        purpose: str,
        umo: str,
        system_prompt: str,
        prompt: str,
    ) -> str | None:
        """Call the provider for one background distillation.

        Args:
            purpose: Short label for the usage log.
            umo: Session used to resolve the provider and attribute cost.
            system_prompt: System instruction.
            prompt: User content of the distillation request.

        Returns:
            The response text, or None on failure.
        """
        provider = await self._get_provider(umo or None)
        if provider is None:
            return None
        provider_config = getattr(provider, "provider_config", {}) or {}
        provider_id = str(provider_config.get("id") or "")
        provider_model = ""
        try:
            provider_model = str(provider.get_model() or "")
            if not provider_id:
                provider_id = str(provider.meta().id)
        except Exception:
            provider_id = provider_id or "unknown"

        start = time.time()
        try:
            response = await provider.text_chat(
                prompt=prompt,
                system_prompt=system_prompt,
                contexts=[],
                request_max_retries=2,
            )
        except Exception:
            self.logger.exception(
                "bot2_memory: %s call failed for %s", purpose, umo or "global"
            )
            self._log_usage(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "purpose": purpose,
                    "umo": umo,
                    "provider": provider_id,
                    "model": provider_model,
                    "input_other": 0,
                    "input_cached": 0,
                    "output": 0,
                    "elapsed": round(time.time() - start, 3),
                    "status": "error",
                }
            )
            return None
        end = time.time()

        usage = getattr(response, "usage", None)
        counts = {
            "input_other": self._usage_field(usage, "input_other"),
            "input_cached": self._usage_field(usage, "input_cached"),
            "output": self._usage_field(usage, "output"),
        }
        text = str(getattr(response, "completion_text", "") or "").strip()
        if not text:
            chain = getattr(response, "result_chain", None)
            if chain is not None:
                try:
                    text = str(chain.get_plain_text() or "").strip()
                except Exception:
                    text = ""
        self._log_usage(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "purpose": purpose,
                "umo": umo,
                "provider": provider_id,
                "model": provider_model,
                **counts,
                "elapsed": round(end - start, 3),
                "status": "completed" if text else "empty",
            }
        )
        await self._record_provider_stat(
            umo=umo,
            provider_id=provider_id or "unknown",
            provider_model=provider_model,
            usage=counts,
            start_time=start,
            end_time=end,
            status="completed" if text else "error",
        )
        return text or None

    # --------------------------------------------------------------- writers

    def _write_card(self, qq: str, name: str, content: str) -> None:
        """Write one impression card atomically with a small header."""
        safe_name = " ".join(str(name or qq).split())
        header = (
            f"---\nnickname: {safe_name}\n"
            f"updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n---\n\n"
        )
        _atomic_write(self._card_path(qq), header + content.strip() + "\n")

    def _read_entry(self, umo: str, day: str) -> str:
        """Return the body of one day's journal entry, or an empty string."""
        path = self._session_dir(umo) / f"{day}.md"
        try:
            return _entry_body(path.read_text(encoding="utf-8"))
        except OSError:
            return ""

    def _write_entry(self, umo: str, day: str, content: str) -> None:
        """Write one day's journal entry atomically."""
        path = self._session_dir(umo) / f"{day}.md"
        text = (
            f"# {day}\n"
            f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{content.strip()}\n"
        )
        _atomic_write(path, text)

    # ------------------------------------------------------------- distillation

    async def _summarize_entry(
        self, umo: str, day: str, material: str, existing: str
    ) -> str | None:
        """Rewrite one day's journal entry from new messages.

        Args:
            umo: Session being summarized.
            day: Target date (YYYY-MM-DD).
            material: New transcript since the previous distillation.
            existing: Current entry body, if any.

        Returns:
            The new entry body, or None when the model call failed.
        """
        prompt = (
            "你在维护一个群聊的长期记忆，只保留以后还有用的事。\n"
            f"日期：{day}\n\n"
            "已有事记（可能为空）：\n"
            f"{existing or '（无）'}\n\n"
            "自上次提炼以来的新对话：\n"
            f"{material}\n\n"
            "请把新对话里的有效信息合并进已有事记，压缩成当天的一条事记。\n"
            "要求：\n"
            "1. 保留人物、事实、承诺、偏好、关系变化、情绪转折和梗，删掉寒暄与重复。\n"
            "2. 不编造；不写“根据记录/根据对话”；不要写成流水账。\n"
            "3. 第三人称中文，200~400字，信息密度高。\n"
            "只输出事记正文。"
        )
        text = await self._call_llm(
            purpose="session_summary",
            umo=umo,
            system_prompt="你是一个只输出结果的记忆整理器，不寒暄、不解释。",
            prompt=prompt,
        )
        if not text:
            return None
        text = re.sub(rf"^{re.escape(day)}[：:]\s*", "", text)
        return _clip(text, MAX_ENTRY_CHARS)

    async def _rewrite_impression(
        self,
        qq: str,
        name: str,
        old: str,
        material: str,
        *,
        reason: str = "",
        umo: str | None = None,
        purpose: str,
    ) -> str | None:
        """Rewrite one impression card from the old card and new evidence.

        Args:
            qq: QQ number.
            name: Display name.
            old: Old card body, or an empty string.
            material: New observations or the day's journal.
            reason: Why the impression changed (tool path only).
            umo: Session to attribute the provider call to.
            purpose: Usage-log label.

        Returns:
            A clipped card body, or None when the model call failed.
        """
        extra = f"\n变化依据：{reason}\n" if reason else ""
        prompt = (
            "你在维护自己对一位群友的印象卡。\n"
            f"对方：{name}（QQ {qq}）\n\n"
            "旧印象卡：\n"
            f"{old or '（无，这是初印象）'}\n\n"
            "新的观察/变化：\n"
            f"{material}\n"
            f"{extra}\n"
            "请写出更新后的印象卡。要求：\n"
            "1. 用第一人称“我”写你对他的看法，保留“初印象→后来改观”的演化感。\n"
            "2. 只写有依据的性格、偏好、相处方式和值得记住的事，不编造。\n"
            "3. 不写日期，不写“根据聊天记录/根据记忆”，不罗列流水账。\n"
            "4. 在旧卡基础上逐点修订，不要推倒重写；没有新认知就少改或不改。\n"
            f"5. 中文，{self.impression_max_chars}字以内、越精炼越好，只输出卡片正文。"
        )
        text = await self._call_llm(
            purpose=purpose,
            umo=umo or "",
            system_prompt="你是一个只输出结果的记忆整理器，不寒暄、不解释。",
            prompt=prompt,
        )
        if not text:
            return None
        return _clip(text, self.impression_max_chars)

    async def _update_impressions(self, umo: str, day: str, entry: str) -> None:
        """Refresh cards of people who interacted on ``day`` and have new words.

        People with no new messages since their last card update are skipped,
        so an idle scan followed by the evening run does not pay twice.

        Args:
            umo: Session whose participants are considered.
            day: Date of the interaction.
            entry: The day's journal body used as shared context.
        """
        session = self._session_state(umo)
        participants = session["participants"].get(day)
        if not isinstance(participants, dict) or not participants:
            return
        for qq, name in list(participants.items()):
            qq = str(qq)
            name = str(name or qq)
            person = self._person_state(qq)
            has_card = self._card_path(qq).exists()
            last_impression = float(person.get("last_impression_ts", 0) or 0)
            last_seen = float(person.get("last_seen_ts", 0) or 0)
            if has_card and last_seen <= last_impression:
                continue
            old = self._read_card(qq)
            notes = [
                text
                for timestamp, _, text in self._person_notes.get(qq, [])
                if timestamp > last_impression
            ][-10:]
            material = f"日期：{day}\n这个会话今天的事记：\n{entry}"
            if notes:
                material += "\n\n他今天说过的话：\n" + "\n".join(
                    f"- {note}" for note in notes
                )
            async with self._impression_lock:
                new = await self._rewrite_impression(
                    qq,
                    name,
                    old,
                    material,
                    umo=umo,
                    purpose="impression_daily",
                )
                if not new:
                    self.logger.warning(
                        "bot2_memory: impression refresh failed for %s in %s",
                        qq,
                        umo,
                    )
                    continue
                self._write_card(qq, name, new)
                person["last_impression_ts"] = time.time()
                self._save_state()
                self._update_index(qq, name)
                self.logger.info(
                    "bot2_memory: refreshed impression for %s (%s)", name, qq
                )

    # ---------------------------------------------------------- maintenance

    async def _process_daily_session(
        self, umo: str, conversations: list, day: str
    ) -> bool:
        """Distill one session at the daily run.

        Args:
            umo: Session origin.
            conversations: All stored conversations of that session.
            day: Target journal date.

        Returns:
            True when new messages were found and processed.
        """
        manager = self.context.conversation_manager
        new_parts: list[str] = []
        cursors: list[tuple[dict, int, str]] = []
        for conversation in conversations:
            current = await manager.get_conversation(umo, conversation.cid)
            if current is None:
                continue
            history = _parse_history(current.history)
            if not history:
                continue
            session = self._session_state(umo)
            cursor = session["conversations"].setdefault(
                conversation.cid, {"cursor": 0, "anchor": ""}
            )
            if not isinstance(cursor, dict):
                cursor = {"cursor": 0, "anchor": ""}
                session["conversations"][conversation.cid] = cursor
            new, _ = _history_since(history, cursor, session.get("recent_people"))
            if not new:
                continue
            cursors.append((cursor, len(history), _msg_anchor(history[-1])))
            text = _format_history(new)
            if text.strip():
                new_parts.append(text)

        if not cursors:
            return False
        material = "\n\n".join(new_parts)
        content = ""
        if material.strip():
            existing = self._read_entry(umo, day)
            summarized = await self._summarize_entry(umo, day, material, existing)
            if summarized is None:
                self.logger.warning(
                    "bot2_memory: daily summary failed for %s, cursors kept", umo
                )
                return True
            content = summarized
            self._write_entry(umo, day, content)

        session = self._session_state(umo)
        for cursor, length, anchor in cursors:
            cursor["cursor"] = length
            cursor["anchor"] = anchor
        session["last_summary_ts"] = time.time()
        session["last_daily_run"] = day
        self._save_state()
        self.logger.info(
            "bot2_memory: daily distilled %s on %s (%d conversation(s), %d chars)",
            umo,
            day,
            len(cursors),
            len(material),
        )
        if content:
            await self._update_impressions(umo, day, content)
        return True

    async def _run_daily(self, day: str | None = None) -> None:
        """Run the daily distillation for every session with new messages.

        Args:
            day: Journal date, defaults to today.
        """
        if not self.enabled:
            return
        day = day or date.today().isoformat()
        async with self._maintenance_lock:
            conversations = await self.context.conversation_manager.get_conversations()
            grouped: dict[str, list] = {}
            for conversation in conversations:
                grouped.setdefault(conversation.user_id, []).append(conversation)
            processed = 0
            for umo, items in grouped.items():
                try:
                    if await self._process_daily_session(umo, items, day):
                        processed += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.logger.exception(
                        "bot2_memory: daily distillation failed for %s", umo
                    )
            self.logger.info(
                "bot2_memory: daily run %s finished, %d/%d session(s) had new messages",
                day,
                processed,
                len(grouped),
            )

    async def _idle_scan(self) -> None:
        """Distill idle sessions, then trim their history to the message cap."""
        if not self.enabled:
            return
        manager = self.context.conversation_manager
        cutoff = time.time() - self.idle_hours * 3600
        async with self._maintenance_lock:
            conversations = await manager.get_conversations()
            for conversation in conversations:
                try:
                    if conversation.updated_at > cutoff:
                        continue
                    umo = conversation.user_id
                    current = await manager.get_conversation(umo, conversation.cid)
                    if current is None:
                        continue
                    history = _parse_history(current.history)
                    if not history:
                        continue
                    session = self._session_state(umo)
                    cursor = session["conversations"].setdefault(
                        conversation.cid, {"cursor": 0, "anchor": ""}
                    )
                    if not isinstance(cursor, dict):
                        cursor = {"cursor": 0, "anchor": ""}
                        session["conversations"][conversation.cid] = cursor
                    new, _ = _history_since(history, cursor, session.get("recent_people"))
                    if not new and len(history) <= self.keep_messages:
                        continue

                    day = (
                        datetime.fromtimestamp(
                            int(current.updated_at or conversation.updated_at)
                        )
                        .date()
                        .isoformat()
                    )
                    content = ""
                    if new:
                        material = _format_history(new)
                        if material.strip():
                            existing = self._read_entry(umo, day)
                            summarized = await self._summarize_entry(
                                umo, day, material, existing
                            )
                            if summarized is None:
                                self.logger.warning(
                                    "bot2_memory: idle summary failed for %s, "
                                    "skipping trim this round",
                                    umo,
                                )
                                continue
                            content = summarized
                            self._write_entry(umo, day, content)
                        cursor["cursor"] = len(history)
                        cursor["anchor"] = _msg_anchor(history[-1])
                        session["last_summary_ts"] = time.time()
                        self._save_state()
                        if content:
                            await self._update_impressions(umo, day, content)

                    if len(history) > self.keep_messages:
                        latest = await manager.get_conversation(umo, conversation.cid)
                        if latest is None or latest.updated_at != current.updated_at:
                            self.logger.info(
                                "bot2_memory: %s/%s changed during idle scan, "
                                "skipping trim this round",
                                umo,
                                conversation.cid,
                            )
                            continue
                        start = len(history) - self.keep_messages
                        while (
                            start < len(history)
                            and history[start].get("role") == "tool"
                        ):
                            start += 1
                        trimmed = history[start:]
                        if not trimmed:
                            continue
                        messages = []
                        for entry in trimmed:
                            try:
                                messages.append(Message.model_validate(entry))
                            except ValueError:
                                continue
                        token_usage = EstimateTokenCounter().count_tokens(messages)
                        await manager.update_conversation(
                            unified_msg_origin=umo,
                            conversation_id=conversation.cid,
                            history=trimmed,
                            token_usage=token_usage,
                        )
                        cursor["cursor"] = len(trimmed)
                        cursor["anchor"] = _msg_anchor(trimmed[-1])
                        self._save_state()
                        self.logger.info(
                            "bot2_memory: trimmed %s/%s from %d to %d messages",
                            umo,
                            conversation.cid,
                            len(history),
                            len(trimmed),
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.logger.exception(
                        "bot2_memory: idle scan failed for %s",
                        getattr(conversation, "user_id", "?"),
                    )

    def _daily_due(self, now: datetime) -> bool:
        """Check whether today's daily run is due but has not started yet.

        Args:
            now: Current local time.

        Returns:
            True at or after ``daily_time`` once per process day.
        """
        if not self.enabled:
            return False
        if now.strftime("%H:%M") < self.daily_time:
            return False
        return self._daily_ran_date != now.date().isoformat()

    async def _daily_loop(self) -> None:
        """Run the daily distillation at the configured time."""
        await asyncio.sleep(20)
        while True:
            try:
                self._refresh_config()
                now = datetime.now()
                if self._daily_due(now):
                    await self._run_daily(now.date().isoformat())
                    self._daily_ran_date = now.date().isoformat()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("bot2_memory: daily scheduler failed")
            await asyncio.sleep(DAILY_CHECK_INTERVAL_SECONDS)

    async def _idle_loop(self) -> None:
        """Scan for idle conversations and trim them after distillation."""
        await asyncio.sleep(60)
        while True:
            try:
                await self._idle_scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("bot2_memory: idle scanner failed")
            await asyncio.sleep(IDLE_SCAN_INTERVAL_SECONDS)

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        """Start the daily and idle background tasks."""
        self._refresh_config()
        self._state = self._load_state()
        self._daily_task = asyncio.create_task(
            self._daily_loop(), name="bot2_memory_daily"
        )
        self._idle_task = asyncio.create_task(
            self._idle_loop(), name="bot2_memory_idle"
        )
        self.logger.info(
            "bot2_memory: background tasks started (daily %s, idle %.2fh, "
            "keep %d messages)",
            self.daily_time,
            self.idle_hours,
            self.keep_messages,
        )

    async def terminate(self) -> None:
        """Cancel background tasks when the plugin is unloaded."""
        for task in (self._daily_task, self._idle_task):
            if task is None or task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._daily_task = None
        self._idle_task = None
        self._save_state()
        self.logger.info("bot2_memory: background tasks stopped")
