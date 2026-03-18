import asyncio
import io
import json
import logging
import re
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from config import (
    ANALYSIS_COOLDOWN_SECONDS,
    ANALYSIS_QUEUE_TIMEOUT_SECONDS,
    ANALYSIS_TIMEOUT_SECONDS,
    BOT_TOKEN,
    LOG_LEVEL,
    MAX_ANALYSIS_TEXT_CHARS,
    MAX_DIALOG_MESSAGES,
    MAX_CONCURRENT_ANALYSES,
    MIN_VOICE_DURATION_SECONDS,
    MAX_VOICE_DURATION_SECONDS,
    MESSAGE_RATE_LIMIT_COUNT,
    MESSAGE_RATE_LIMIT_NOTICE_COOLDOWN,
    MESSAGE_RATE_LIMIT_WINDOW_SECONDS,
)
from ai_analyzer import analyze_chat, analyze_followup, transcribe_voice
from database import get_analysis_run_by_id, get_latest_analysis_run, init_db, save_analysis_run


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Keep our app logs visible while muting noisy third-party request/update logs.
    noisy_loggers = (
        "aiogram",
        "httpx",
        "openai",
        "httpcore",
    )
    quiet_level = max(level, logging.WARNING)

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(quiet_level)


logger = logging.getLogger(__name__)


class AntiSpamMiddleware(BaseMiddleware):
    def __init__(
        self,
        limit_count: int,
        window_seconds: int,
        notice_cooldown: int,
    ) -> None:
        self.limit_count = limit_count
        self.window_seconds = window_seconds
        self.notice_cooldown = notice_cooldown
        self.user_events: dict[int, deque[float]] = defaultdict(deque)
        self.notice_sent_at: dict[int, float] = {}

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if event.chat.type != "private":
            return None

        user = event.from_user
        if user is None:
            return await handler(event, data)

        now = time.monotonic()
        timestamps = self.user_events[user.id]
        cutoff = now - self.window_seconds

        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= self.limit_count:
            last_notice = self.notice_sent_at.get(user.id, 0.0)
            if now - last_notice >= self.notice_cooldown:
                self.notice_sent_at[user.id] = now
                await event.answer(
                    "Слишком много сообщений за короткое время. Подожди несколько секунд и продолжай."
                )
            logger.warning("Rate limit triggered for user_id=%s", user.id)
            return None

        timestamps.append(now)
        return await handler(event, data)


analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
analysis_cooldowns: dict[int, float] = {}

dp = Dispatcher(
    storage=MemoryStorage(),
    events_isolation=SimpleEventIsolation()
)
dp.message.middleware(
    AntiSpamMiddleware(
        limit_count=MESSAGE_RATE_LIMIT_COUNT,
        window_seconds=MESSAGE_RATE_LIMIT_WINDOW_SECONDS,
        notice_cooldown=MESSAGE_RATE_LIMIT_NOTICE_COOLDOWN,
    )
)

BUTTON_START = "Старт"
BUTTON_ANALYZE = "Разобрать переписку"
BUTTON_NEW_ANALYSIS = "Новый анализ"
BUTTON_FOLLOWUP = "Ответил — проверить что изменилось"
BUTTON_FINISH_FOLLOWUP = "Проверить что изменилось"

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BUTTON_START)]
    ],
    resize_keyboard=True,
    input_field_placeholder="Нажми «Старт», чтобы начать"
)


collect_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Закончить и проанализировать")],
        [KeyboardButton(text="Отмена")]
    ],
    resize_keyboard=True
)

preview_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Отмена")]
    ],
    resize_keyboard=True
)

result_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BUTTON_FOLLOWUP)],
        [KeyboardButton(text=BUTTON_NEW_ANALYSIS)]
    ],
    resize_keyboard=True
)

followup_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BUTTON_FINISH_FOLLOWUP)],
        [KeyboardButton(text="Отмена")]
    ],
    resize_keyboard=True
)


class AnalyzeState(StatesGroup):
    collecting_preview = State()
    choosing_role = State()
    collecting_messages = State()
    collecting_followup = State()


class VoiceProcessingError(Exception):
    pass


VOICE_FILLER_WORDS = {
    "а", "ага", "да", "неа", "угу", "ок", "окей", "ясно", "понял", "пон", "понятно",
    "мм", "м", "эм", "ээ", "ну", "нуда", "ладно", "хорошо", "щас", "сейчас",
}
VOICE_NOISE_WORDS = {
    "шум", "музыка", "тишина", "смех", "вздох", "кашель",
}


def normalize_score_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def extract_forward_timestamp(message: Message) -> float | None:
    forward_date = getattr(message, "forward_date", None)
    if forward_date is not None:
        return float(forward_date.timestamp())

    forward_origin = getattr(message, "forward_origin", None)
    origin_date = getattr(forward_origin, "date", None)
    if origin_date is not None:
        return float(origin_date.timestamp())

    return None


def build_message_item(
    message: Message,
    text: str,
    *,
    sender_name: str | None = None,
    sender_label: str | None = None,
) -> dict:
    return {
        "text": text,
        "date": float(message.date.timestamp()),
        "message_id": message.message_id,
        "chat_id": message.chat.id if message.chat else None,
        "sender_name": sender_name,
        "sender_label": sender_label,
        "forward_date": extract_forward_timestamp(message),
    }


def build_message_key(item: dict) -> tuple[str, ...] | None:
    chat_id = item.get("chat_id")
    message_id = item.get("message_id")
    if chat_id is not None and message_id is not None:
        return ("telegram", str(chat_id), str(message_id))

    sender_name = str(item.get("sender_name", "")).strip()
    forward_date = item.get("forward_date")
    text = str(item.get("text", "")).strip()
    if sender_name and forward_date is not None and text:
        return ("forward", sender_name, str(forward_date), text)

    sender_label = str(item.get("sender_label", "")).strip()
    date = item.get("date")
    if text or date is not None or sender_label or sender_name:
        return (
            "fallback",
            sender_label,
            sender_name,
            str(date),
            text,
        )

    return ("fallback", repr(sorted(item.items())))


def append_unique_message(messages: list[dict], item: dict) -> bool:
    key = build_message_key(item)
    known_keys = {build_message_key(existing) for existing in messages}
    if key in known_keys:
        return False

    messages.append(item)
    return True


def parse_selected_user(text: str) -> str | None:
    text = text.strip()

    match = re.match(r"^я\s*[-—]?\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None

def extract_sender_name(message: Message) -> str | None:
    if getattr(message, "forward_from", None):
        user = message.forward_from
        if user.username:
            return f"@{user.username}"
        return user.full_name

    if getattr(message, "forward_sender_name", None):
        return message.forward_sender_name

    return None


def extract_text(message: Message) -> str | None:
    if message.text:
        return message.text
    if message.caption:
        return message.caption
    return None


def is_low_signal_transcript(text: str) -> bool:
    normalized = text.lower().replace("ё", "е")
    tokens = re.findall(r"[a-zа-я]+", normalized, flags=re.IGNORECASE)

    if not tokens:
        return True

    if len(tokens) <= 3 and all(token in VOICE_FILLER_WORDS for token in tokens):
        return True

    if all(token in VOICE_NOISE_WORDS for token in tokens):
        return True

    filler_count = sum(token in VOICE_FILLER_WORDS for token in tokens)
    filler_ratio = filler_count / len(tokens)
    unique_tokens = set(tokens)
    meaningful_tokens = [token for token in tokens if token not in VOICE_FILLER_WORDS]

    if len(" ".join(tokens)) < 12:
        return True

    if len(meaningful_tokens) < 2:
        return True

    if len(tokens) <= 8 and filler_ratio >= 0.6:
        return True

    if len(unique_tokens) <= 2 and len(tokens) >= 4:
        return True

    return False


SCORE_VALUE_PATTERNS = (
    r"(?<!\d)(?P<value>10|[0-9])\s*/\s*10(?!\d)",
    r"(?<!\d)(?P<value>10|[0-9])\s+из\s+10(?!\d)",
)


def find_labeled_score(text: str, labels: tuple[str, ...]) -> int | None:
    escaped_labels = "|".join(re.escape(label) for label in labels)
    for pattern in SCORE_VALUE_PATTERNS:
        match = re.search(
            rf"(?:{escaped_labels})[^\n]{{0,40}}?{pattern}",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            value = int(match.group("value"))
            if 0 <= value <= 10:
                return value
    return None


def parse_interest_score(text: str) -> int | None:
    normalized = normalize_score_text(text)
    labeled_groups = (
        ("после", "интерес после", "текущий интерес", "интерес сейчас"),
        ("интерес", "уровень интереса", "оценка интереса"),
        ("до", "интерес до"),
    )

    for labels in labeled_groups:
        score = find_labeled_score(normalized, labels)
        if score is not None:
            return score

    for pattern in SCORE_VALUE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            value = int(match.group("value"))
            if 0 <= value <= 10:
                return value

    return None


def parse_followup_scores(text: str) -> tuple[int | None, int | None]:
    normalized = normalize_score_text(text)
    before = find_labeled_score(normalized, ("до", "интерес до"))
    after = find_labeled_score(normalized, ("после", "интерес после", "текущий интерес", "интерес сейчас"))
    return before, after


def parse_followup_marker(text: str, label: str, allowed_values: tuple[str, ...]) -> str | None:
    normalized = normalize_score_text(text)
    match = re.search(
        rf"{re.escape(label.lower())}\s*:\s*([^\n]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = match.group(1).strip()
    for allowed in allowed_values:
        if value.startswith(allowed):
            return allowed
    return None


def extract_followup_meta(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"<followup_meta>\s*(\{.*?\})\s*</followup_meta>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def strip_followup_meta(text: str) -> str:
    return re.sub(
        r"\s*<followup_meta>\s*\{.*?\}\s*</followup_meta>\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def coerce_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 10 else None
    if isinstance(value, float) and value.is_integer():
        integer_value = int(value)
        return integer_value if 0 <= integer_value <= 10 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            integer_value = int(stripped)
            return integer_value if 0 <= integer_value <= 10 else None
    return None


def coerce_allowed_value(value: Any, allowed_values: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = normalize_score_text(value.strip())
    for allowed in allowed_values:
        if normalized == allowed or normalized.startswith(allowed):
            return allowed
    return None


def build_followup_progress_summary(
    interest_before: int | None,
    interest_after: int | None,
    position_status: str | None,
    warmth_status: str | None,
    advice_effectiveness: str | None,
) -> str | None:
    lines = ["<b>Короткий итог</b>"]

    if interest_before is not None and interest_after is not None:
        if interest_after > interest_before:
            interest_label = "вырос"
        elif interest_after < interest_before:
            interest_label = "просел"
        else:
            interest_label = "не изменился"
        lines.append(
            f"• Интерес: {interest_before}/10 → {interest_after}/10, {interest_label}"
        )
    elif interest_after is not None:
        lines.append(f"• Текущий интерес: {interest_after}/10")

    position_map = {
        "улучшилась": "• Позиция: улучшилась",
        "без изменений": "• Позиция: без заметных изменений",
        "ухудшилась": "• Позиция: ухудшилась",
    }
    warmth_map = {
        "теплее": "• Собеседник: стал теплее",
        "без изменений": "• Собеседник: без заметного потепления",
        "холоднее": "• Собеседник: стал холоднее",
    }
    advice_map = {
        "сработал": "• Прошлый совет: сработал",
        "частично": "• Прошлый совет: сработал частично",
        "не сработал": "• Прошлый совет: не сработал",
    }

    if position_status in position_map:
        lines.append(position_map[position_status])
    if warmth_status in warmth_map:
        lines.append(warmth_map[warmth_status])
    if advice_effectiveness in advice_map:
        lines.append(advice_map[advice_effectiveness])

    if len(lines) == 1:
        return None

    return "\n".join(lines)


def format_messages_for_prompt(messages: list[dict]) -> str:
    return "\n".join(message["text"] for message in messages)


def merge_messages(existing: list[dict], new_messages: list[dict]) -> list[dict]:
    merged = list(existing)
    known_pairs = {build_message_key(item) for item in merged}

    for item in new_messages:
        key = build_message_key(item)
        if key not in known_pairs:
            merged.append(item)
            known_pairs.add(key)

    return sorted(merged, key=lambda item: item["date"])


async def transcribe_voice_message(message: Message) -> str:
    voice = message.voice
    if voice is None:
        raise VoiceProcessingError("Голосовое сообщение не найдено.")

    if voice.duration < MIN_VOICE_DURATION_SECONDS:
        raise VoiceProcessingError(
            "Слишком короткое голосовое. Короткие ответы вроде «да», «угу», "
            "«пон» лучше пересылать текстом."
        )

    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        raise VoiceProcessingError(
            f"Голосовое длиннее {MAX_VOICE_DURATION_SECONDS} сек. "
            "Отправь более короткое голосовое или текст."
        )

    await message.answer(
        "Распознаю голосовое сообщение...\n"
        "Это может занять несколько секунд."
    )

    voice_bytes = io.BytesIO()
    voice_bytes.name = f"voice_{message.message_id}.ogg"

    try:
        await message.bot.download(voice, destination=voice_bytes)
    except Exception as exc:
        logger.exception("Failed to download voice message")
        raise VoiceProcessingError(
            "Не удалось скачать голосовое сообщение. Перешли его ещё раз."
        ) from exc

    try:
        await asyncio.wait_for(
            analysis_semaphore.acquire(),
            timeout=ANALYSIS_QUEUE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise VoiceProcessingError(
            "Сервис распознавания сейчас перегружен. Подожди немного и попробуй снова."
        ) from exc

    try:
        transcript = await asyncio.wait_for(
            transcribe_voice(voice_bytes.name, voice_bytes.getvalue()),
            timeout=ANALYSIS_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise VoiceProcessingError(
            "Распознавание голосового заняло слишком много времени. Попробуй ещё раз."
        ) from exc
    except Exception as exc:
        logger.exception("Voice transcription failed")
        raise VoiceProcessingError(
            "Не удалось распознать голосовое сообщение. Попробуй ещё раз или перешли текст."
        ) from exc
    finally:
        analysis_semaphore.release()

    if is_low_signal_transcript(transcript):
        raise VoiceProcessingError(
            "В голосовом слишком мало смысла для анализа: похоже на короткий ответ, "
            "мусорную фразу или шум. Перешли более содержательный фрагмент или текст."
        )

    return f"[голосовое] {transcript}"


async def extract_message_text(message: Message) -> str | None:
    raw_text = extract_text(message)
    if raw_text:
        return raw_text

    if message.voice:
        return await transcribe_voice_message(message)

    return None


def build_role_keyboard(participants: list[str]) -> ReplyKeyboardMarkup:
    keyboard_rows = [[KeyboardButton(text=f"Я — {name}")] for name in participants]
    keyboard_rows.append([KeyboardButton(text="Отмена")])

    return ReplyKeyboardMarkup(
        keyboard=keyboard_rows,
        resize_keyboard=True
    )

@dp.message(CommandStart())
async def start_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "<b>Разберём переписку вместе</b>\n\n"
        "Я помогаю понять поведение людей в диалоге: кто вкладывается больше, "
        "есть ли реальный интерес, кто в сильной позиции и что на самом деле происходит между вами.\n\n"
        "<b>Как начать:</b>\n"
        "1. Нажми <b>Старт</b>\n"
        "2. Перешли 2–4 сообщения или голосовых, чтобы я определил участников\n"
        "3. Затем отправь остальную переписку и запусти анализ",
        reply_markup=start_keyboard
    )


@dp.message(F.text == BUTTON_FOLLOWUP)
async def followup_button_handler(message: Message, state: FSMContext):
    user = message.from_user
    if user is None:
        await message.answer("Не удалось определить пользователя. Попробуй ещё раз.")
        return

    latest_run = get_latest_analysis_run(user.id)
    if latest_run is None:
        await message.answer(
            "Сначала нужен хотя бы один готовый анализ. Нажми <b>Старт</b> и сделай первый разбор.",
            reply_markup=start_keyboard
        )
        return

    await state.set_state(AnalyzeState.collecting_followup)
    await state.update_data(
        followup_messages=[],
        followup_source_run_id=latest_run["id"],
        followup_last_progress_count=0,
        followup_limit_notified=False,
        followup_limit_exceeded=False,
    )

    await message.answer(
        "<b>Проверим, что изменилось</b>\n\n"
        "Перешли только новые сообщения после твоего ответа или действия.\n"
        f"Лимит на этот этап — до {MAX_DIALOG_MESSAGES} новых сообщений.\n"
        "Когда закончишь, нажми <b>Проверить что изменилось</b>.",
        reply_markup=followup_keyboard
    )


@dp.message(F.text.in_([BUTTON_START, BUTTON_ANALYZE, BUTTON_NEW_ANALYSIS]))
async def analyze_button_handler(message: Message, state: FSMContext):
    await state.set_state(AnalyzeState.collecting_preview)
    await state.update_data(
        preview_waiting_sent=False,
        preview_messages=[],
        participants=[],
        messages=[],
        selected_user=None,
        last_progress_count=0,
        base_messages_count=0,
        limit_notified=False,
        role_prompt_sent=False,
        limit_exceeded=False
    )

    await message.answer(
        "<b>Шаг 1 из 2</b>\n\n"
        "Перешли 2–4 сообщения или голосовых из переписки, чтобы я определил участников.",
        reply_markup=preview_keyboard
    )

@dp.message(AnalyzeState.choosing_role)
async def choose_role_handler(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return

    selected_user = parse_selected_user(message.text or "")

    if not selected_user:
        await message.answer("Выбери участника кнопкой ниже или напиши в формате: Я — имя")
        return

    data = await state.get_data()
    participants = data.get("participants", [])
    preview_messages = data.get("preview_messages", [])

    if selected_user not in participants:
        await message.answer("Такого участника нет в найденных. Выбери кнопкой ниже или напиши в формате: Я — имя")
        return

    converted_messages = []
    for item in preview_messages:
        label = "Ты" if item["sender_name"] == selected_user else "Собеседник"
        converted_item = dict(item)
        converted_item["sender_label"] = label
        converted_item["text"] = f"{label}: {item['text']}"
        converted_messages.append(converted_item)

    await state.update_data(
        selected_user=selected_user,
        messages=converted_messages,
        last_progress_count=0,
        base_messages_count=len(converted_messages),
        limit_notified=False,
        limit_exceeded=False
    )
    await state.set_state(AnalyzeState.collecting_messages)

    await message.answer(
        f"Принято. Ты выбран как: {selected_user}\n\n"
        "Теперь пересылай остальную переписку.\n"
        "Когда закончишь, нажми «Закончить и проанализировать».",
        reply_markup=collect_keyboard
    )

@dp.message(F.text == "Отмена")
async def cancel_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Сбор переписки отменён.",
        reply_markup=start_keyboard
    )


@dp.message(AnalyzeState.collecting_preview)
async def collect_preview_handler(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return

    if message.text == "Закончить и проанализировать":
        return

    sender_name = extract_sender_name(message)
    if not sender_name:
        await message.answer(
            "На этом этапе перешли сообщения или голосовые именно из переписки, "
            "чтобы я увидел участников.",
            reply_markup=preview_keyboard
        )
        return

    try:
        raw_text = await extract_message_text(message)
    except VoiceProcessingError as exc:
        await message.answer(str(exc), reply_markup=preview_keyboard)
        return

    if not raw_text:
        await message.answer(
            "На этом этапе перешли сообщения из переписки: текст, подпись или голосовое.",
            reply_markup=preview_keyboard
        )
        return

    data = await state.get_data()
    preview_messages = data.get("preview_messages", [])
    participants = data.get("participants", [])
    role_prompt_sent = data.get("role_prompt_sent", False)
    preview_waiting_sent = data.get("preview_waiting_sent", False)

    preview_item = build_message_item(
        message,
        raw_text,
        sender_name=sender_name,
    )

    append_unique_message(preview_messages, preview_item)

    is_new_participant = sender_name not in participants
    if is_new_participant:
        participants.append(sender_name)

    await state.update_data(
        preview_messages=preview_messages,
        participants=participants
    )

    if len(participants) >= 2:
        if role_prompt_sent:
            return

        found_participants = participants[:2]

        await state.update_data(
            role_prompt_sent=True,
            preview_waiting_sent=False
        )
        await state.set_state(AnalyzeState.choosing_role)

        await message.answer(
            "Я нашёл участников переписки:\n\n"
            f"• {found_participants[0]}\n"
            f"• {found_participants[1]}\n\n"
            "Кого из них считать тобой?",
            reply_markup=build_role_keyboard(found_participants)
        )
        return

    if len(preview_messages) >= 2 and len(participants) == 1 and not preview_waiting_sent:
        await state.update_data(preview_waiting_sent=True)
        await message.answer(
            "Пока вижу только одного участника. Перешли хотя бы одно сообщение второго человека.",
            reply_markup=preview_keyboard
        )


def calculate_dialog_metrics(messages):
    user_count = 0
    other_count = 0
    user_words = 0
    other_words = 0
    user_questions = 0
    other_questions = 0

    for message in messages:
        text = message["text"]
        _, _, content = text.partition(": ")
        message_text = content or text
        word_count = len(message_text.split())
        has_question = "?" in message_text

        if text.startswith("Ты:"):
            user_count += 1
            user_words += word_count
            user_questions += int(has_question)
        else:
            other_count += 1
            other_words += word_count
            other_questions += int(has_question)

    user_avg_len = user_words // user_count if user_count else 0
    other_avg_len = other_words // other_count if other_count else 0
    starter = "пользователь" if messages[0]["text"].startswith("Ты:") else "собеседник"

    metrics = f"""
Метрики диалога

Сообщений пользователя: {user_count}
Сообщений собеседника: {other_count}

Средняя длина сообщений:
пользователь: {user_avg_len} слов
собеседник: {other_avg_len} слов

Количество вопросов:
пользователь: {user_questions}
собеседник: {other_questions}

Диалог начал: {starter}
"""

    return metrics


def build_dialog_excerpt(messages: list[dict], max_chars: int) -> tuple[str, bool]:
    lines = [message["text"] for message in messages]
    full_dialog = "\n".join(lines)
    if len(full_dialog) <= max_chars:
        return full_dialog, False

    head_limit = min(8, len(lines))
    head_lines = lines[:head_limit]
    head_text = "\n".join(head_lines)
    separator = "\n[... часть переписки сокращена, чтобы не перегружать анализ ...]\n"

    remaining_chars = max_chars - len(head_text) - len(separator)
    if remaining_chars <= 0:
        shortened_head = head_text[:max_chars].rstrip()
        return shortened_head, True

    tail_lines: list[str] = []
    used_chars = 0
    for line in reversed(lines[head_limit:]):
        line_size = len(line) + (1 if tail_lines else 0)
        if used_chars + line_size > remaining_chars:
            break
        tail_lines.append(line)
        used_chars += line_size

    tail_lines.reverse()
    if not tail_lines:
        shortened_head = head_text[:max_chars].rstrip()
        return shortened_head, True

    return head_text + separator + "\n".join(tail_lines), True


def build_followup_prompt(
    selected_user: str | None,
    other_name: str,
    previous_interest: int | None,
    previous_analysis: str,
    previous_messages: list[dict],
    followup_messages: list[dict],
) -> tuple[str, bool]:
    previous_messages_text = format_messages_for_prompt(previous_messages)
    followup_messages_text = format_messages_for_prompt(followup_messages)
    previous_interest_text = (
        f"{previous_interest}/10" if isinstance(previous_interest, int) else "не извлечена"
    )
    max_previous_analysis_chars = 3000
    analysis_was_truncated = len(previous_analysis) > max_previous_analysis_chars
    previous_analysis_excerpt = previous_analysis[:max_previous_analysis_chars].rstrip()
    if analysis_was_truncated:
        previous_analysis_excerpt += "\n[... предыдущий анализ сокращён ...]"

    prompt_header = (
        f"Пользователь бота: {selected_user}\n"
        f"Собеседник: {other_name}\n\n"
        f"Предыдущая оценка интереса: {previous_interest_text}\n\n"
        "Предыдущий анализ:\n"
        f"{previous_analysis_excerpt}\n\n"
        "Диалог ДО:\n"
    )
    prompt_middle = "\n\nНовые сообщения ПОСЛЕ действия пользователя:\n"

    reserved_chars = len(prompt_header) + len(prompt_middle) + 400
    available_chars = max(1000, MAX_ANALYSIS_TEXT_CHARS - reserved_chars)
    before_budget = max(350, int(available_chars * 0.55))
    after_budget = max(350, available_chars - before_budget)

    previous_excerpt, previous_truncated = build_dialog_excerpt(previous_messages, before_budget)
    followup_excerpt, followup_truncated = build_dialog_excerpt(followup_messages, after_budget)

    if not previous_excerpt:
        previous_excerpt = previous_messages_text
    if not followup_excerpt:
        followup_excerpt = followup_messages_text

    prompt_text = prompt_header + previous_excerpt + prompt_middle + followup_excerpt
    was_truncated = previous_truncated or followup_truncated or analysis_was_truncated

    if was_truncated:
        prompt_text += (
            "\n\nПримечание: часть диалога сокращена, чтобы сохранить фокус на ключевых фрагментах "
            "и не перегружать анализ."
        )

    return prompt_text, was_truncated

@dp.message(F.text == "Закончить и проанализировать")
async def finish_handler(message: Message, state: FSMContext):
    current_state = await state.get_state()

    if current_state != AnalyzeState.collecting_messages.state:
        await message.answer(
            "Сейчас нечего анализировать. Нажми <b>Старт</b>.",
            reply_markup=start_keyboard
        )
        return

    data = await state.get_data()
    limit_exceeded = data.get("limit_exceeded", False)

    if limit_exceeded:
        await message.answer(
            "Лимит сообщений был превышен. Этот набор не будет проанализирован.\n"
            "Нажми <b>Старт</b> и начни заново.",
            reply_markup=start_keyboard
        )
        await state.clear()
        return
    
    messages = sorted(data.get("messages", []), key=lambda item: item["date"])
    selected_user = data.get("selected_user")
    participants = data.get("participants", [])

    if not messages:
        await message.answer(
            "Ты ещё не переслал ни одного сообщения.",
            reply_markup=collect_keyboard
        )
        return

    user = message.from_user
    if user is not None and ANALYSIS_COOLDOWN_SECONDS > 0:
        now = time.monotonic()
        available_at = analysis_cooldowns.get(user.id, 0.0)
        remaining_seconds = int(available_at - now)
        if remaining_seconds > 0:
            await message.answer(
                f"Новый анализ можно запустить через {remaining_seconds} сек.",
                reply_markup=collect_keyboard
            )
            return

        analysis_cooldowns[user.id] = now + ANALYSIS_COOLDOWN_SECONDS

    other_name = next((participant for participant in participants if participant != selected_user), "Собеседник")
    dialog_excerpt, was_truncated = build_dialog_excerpt(messages, MAX_ANALYSIS_TEXT_CHARS)

    full_text = (
        f"Пользователь бота: {selected_user}\n"
        f"Собеседник: {other_name}\n\n"
        "Диалог:\n"
        + dialog_excerpt
    )

    if was_truncated:
        full_text += "\n\nПримечание: длинная переписка была сокращена, сохранив начало и последние сообщения."

    await message.answer("🧠 Анализирую переписку...\nЭто может занять несколько секунд.")

    try:
        metrics = calculate_dialog_metrics(messages)
        try:
            await asyncio.wait_for(
                analysis_semaphore.acquire(),
                timeout=ANALYSIS_QUEUE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            await message.answer(
                "Сервис сейчас перегружен. Подожди немного и попробуй снова.",
                reply_markup=result_keyboard
            )
            return

        try:
            analysis = await asyncio.wait_for(
                analyze_chat(
                    "КОНТЕКСТ ДИАЛОГА:\n"
                    + full_text
                    + "\n\nМЕТРИКИ ДИАЛОГА:\n"
                    + metrics
                ),
                timeout=ANALYSIS_TIMEOUT_SECONDS
            )
        finally:
            analysis_semaphore.release()

        if user is not None:
            initial_interest_score = parse_interest_score(analysis)
            save_analysis_run(
                user_id=user.id,
                run_type="initial",
                payload={
                    "selected_user": selected_user,
                    "participants": participants,
                    "messages": messages,
                    "analysis": analysis,
                    "interest_score": initial_interest_score,
                    "interest_score_after": initial_interest_score,
                },
            )

        await message.answer(analysis, reply_markup=result_keyboard)
    except asyncio.TimeoutError:
        await message.answer(
            "Сервис анализа сейчас отвечает слишком долго. Попробуй ещё раз через минуту.",
            reply_markup=result_keyboard
        )
    except Exception as e:
        logger.exception("Analysis failed")
        await message.answer(
            f"Ошибка анализа: {type(e).__name__}: {e}",
            reply_markup=result_keyboard
        )

    await state.clear()


@dp.message(AnalyzeState.collecting_messages)
async def collect_messages_handler(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return

    if message.text == "Закончить и проанализировать":
        await finish_handler(message, state)
        return

    try:
        raw_text = await extract_message_text(message)
    except VoiceProcessingError as exc:
        await message.answer(str(exc), reply_markup=collect_keyboard)
        return

    sender_name = extract_sender_name(message)

    if not raw_text:
        await message.answer(
            "Это сообщение не удалось прочитать. Перешли текст, подпись или голосовое, "
            "либо нажми «Закончить и проанализировать».",
            reply_markup=collect_keyboard
        )
        return

    data = await state.get_data()
    messages = data.get("messages", [])
    selected_user = data.get("selected_user")
    limit_notified = data.get("limit_notified", False)
    limit_exceeded = data.get("limit_exceeded", False)

    if limit_exceeded:
        return

    if len(messages) >= MAX_DIALOG_MESSAGES:
        await state.update_data(limit_exceeded=True)

        if not limit_notified:
            await state.update_data(limit_notified=True)
            await message.answer(
                f"Ты превысил лимит — {MAX_DIALOG_MESSAGES} сообщений за один анализ.\n"
                "Этот анализ отменён. Нажми <b>Старт</b> и отправь переписку заново в пределах лимита.",
                reply_markup=start_keyboard
            )
            await state.clear()
        return

    if sender_name:
        label = "Ты" if sender_name == selected_user else "Собеседник"
    else:
        label = "Собеседник"

    msg_text = f"{label}: {raw_text}".strip()

    msg = build_message_item(
        message,
        msg_text,
        sender_name=sender_name,
        sender_label=label,
    )

    append_unique_message(messages, msg)

    base_messages_count = data.get("base_messages_count", 0)
    current_count = len(messages) - base_messages_count
    last_progress_count = data.get("last_progress_count", 0)

    await state.update_data(messages=messages)

    if current_count > 0 and current_count % 5 == 0 and current_count != last_progress_count:
        await state.update_data(last_progress_count=current_count)
        await message.answer(
            "Можешь переслать ещё или нажать «Закончить и проанализировать».",
            reply_markup=collect_keyboard
        )


@dp.message(F.text == BUTTON_FINISH_FOLLOWUP)
async def finish_followup_handler(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state != AnalyzeState.collecting_followup.state:
        await message.answer(
            "Сейчас нечего сравнивать. Сначала нажми <b>Ответил — проверить что изменилось</b>.",
            reply_markup=result_keyboard
        )
        return

    data = await state.get_data()
    followup_messages = sorted(data.get("followup_messages", []), key=lambda item: item["date"])
    source_run_id = data.get("followup_source_run_id")
    followup_limit_exceeded = data.get("followup_limit_exceeded", False)

    if followup_limit_exceeded:
        await message.answer(
            "Лимит новых сообщений для этого follow-up уже был превышен. Начни новый цикл меньшим куском.",
            reply_markup=result_keyboard
        )
        await state.clear()
        return

    if not followup_messages:
        await message.answer(
            "Ты ещё не переслал новые сообщения после ответа.",
            reply_markup=followup_keyboard
        )
        return

    user = message.from_user
    if user is None:
        await message.answer("Не удалось определить пользователя. Попробуй ещё раз.")
        return

    source_run = None
    if source_run_id is not None:
        source_run = get_analysis_run_by_id(source_run_id)
    if source_run is None:
        source_run = get_latest_analysis_run(user.id)

    if source_run is None:
        await message.answer(
            "Не нашёл предыдущий анализ для сравнения. Нажми <b>Старт</b> и сделай новый разбор.",
            reply_markup=start_keyboard
        )
        await state.clear()
        return

    selected_user = source_run.get("selected_user")
    participants = source_run.get("participants", [])
    previous_messages = source_run.get("messages", [])
    previous_analysis = source_run.get("analysis", "")
    previous_interest = source_run.get("interest_score")
    merged_messages = merge_messages(previous_messages, followup_messages)

    other_name = next((participant for participant in participants if participant != selected_user), "Собеседник")
    prompt_text, _ = build_followup_prompt(
        selected_user=selected_user,
        other_name=other_name,
        previous_interest=previous_interest,
        previous_analysis=previous_analysis,
        previous_messages=previous_messages,
        followup_messages=followup_messages,
    )

    await message.answer(
        "Сравниваю, что изменилось после твоего действия...\n"
        "Это может занять несколько секунд."
    )

    try:
        try:
            await asyncio.wait_for(
                analysis_semaphore.acquire(),
                timeout=ANALYSIS_QUEUE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            await message.answer(
                "Сервис сейчас перегружен. Подожди немного и попробуй снова.",
                reply_markup=result_keyboard
            )
            return

        try:
            followup_meta, analysis = await asyncio.wait_for(
                analyze_followup(prompt_text),
                timeout=ANALYSIS_TIMEOUT_SECONDS * 2
            )
        finally:
            analysis_semaphore.release()

        interest_before = coerce_score(followup_meta.get("interest_before"))
        if interest_before is None:
            interest_before, _ = parse_followup_scores(analysis)

        interest_after = coerce_score(followup_meta.get("interest_after"))
        if interest_after is None:
            _, interest_after = parse_followup_scores(analysis)

        current_interest = interest_after if interest_after is not None else parse_interest_score(analysis)
        position_status = coerce_allowed_value(
            followup_meta.get("position_status"),
            ("улучшилась", "без изменений", "ухудшилась"),
        )
        if position_status is None:
            position_status = parse_followup_marker(
                analysis,
                "ПОЗИЦИЯ",
                ("улучшилась", "без изменений", "ухудшилась"),
            )

        warmth_status = coerce_allowed_value(
            followup_meta.get("warmth_status"),
            ("теплее", "без изменений", "холоднее"),
        )
        if warmth_status is None:
            warmth_status = parse_followup_marker(
                analysis,
                "ТЕПЛОТА",
                ("теплее", "без изменений", "холоднее"),
            )

        advice_effectiveness = coerce_allowed_value(
            followup_meta.get("advice_effectiveness"),
            ("сработал", "частично", "не сработал"),
        )
        if advice_effectiveness is None:
            advice_effectiveness = parse_followup_marker(
                analysis,
                "СОВЕТ ИЗ ПРОШЛОГО АНАЛИЗА",
                ("сработал", "частично", "не сработал"),
            )
        progress_summary = build_followup_progress_summary(
            interest_before=interest_before,
            interest_after=interest_after if interest_after is not None else current_interest,
            position_status=position_status,
            warmth_status=warmth_status,
            advice_effectiveness=advice_effectiveness,
        )

        save_analysis_run(
            user_id=user.id,
            run_type="followup",
            payload={
                "selected_user": selected_user,
                "participants": participants,
                "messages": merged_messages,
                "new_messages": followup_messages,
                "analysis": analysis,
                "followup_meta": followup_meta,
                "interest_score": current_interest,
                "interest_score_before": interest_before,
                "interest_score_after": interest_after if interest_after is not None else current_interest,
                "position_status": position_status,
                "warmth_status": warmth_status,
                "advice_effectiveness": advice_effectiveness,
                "progress_summary": progress_summary,
                "previous_run_id": source_run["id"],
            },
        )

        final_text = analysis
        if progress_summary:
            final_text = f"{progress_summary}\n\n━━━━━━━━━━━━━━\n\n{analysis}"

        await message.answer(final_text, reply_markup=result_keyboard)
    except asyncio.TimeoutError:
        await message.answer(
            "Сервис анализа сейчас отвечает слишком долго. Попробуй ещё раз через минуту.",
            reply_markup=result_keyboard
        )
    except Exception as exc:
        logger.exception("Follow-up analysis failed")
        await message.answer(
            f"Ошибка повторного анализа: {type(exc).__name__}: {exc}",
            reply_markup=result_keyboard
        )

    await state.clear()


@dp.message(AnalyzeState.collecting_followup)
async def collect_followup_handler(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return

    if message.text == BUTTON_FINISH_FOLLOWUP:
        await finish_followup_handler(message, state)
        return

    try:
        raw_text = await extract_message_text(message)
    except VoiceProcessingError as exc:
        await message.answer(str(exc), reply_markup=followup_keyboard)
        return

    if not raw_text:
        await message.answer(
            "Не удалось прочитать сообщение. Перешли текст, подпись или голосовое.",
            reply_markup=followup_keyboard
        )
        return

    user = message.from_user
    if user is None:
        await message.answer("Не удалось определить пользователя. Попробуй ещё раз.")
        return

    source_run_id = (await state.get_data()).get("followup_source_run_id")
    source_run = get_analysis_run_by_id(source_run_id) if source_run_id is not None else None
    if source_run is None:
        source_run = get_latest_analysis_run(user.id)
    if source_run is None:
        await message.answer(
            "Не нашёл базовый анализ для сравнения. Нажми <b>Старт</b> и начни заново.",
            reply_markup=start_keyboard
        )
        await state.clear()
        return

    selected_user = source_run.get("selected_user")
    sender_name = extract_sender_name(message)
    label = "Собеседник"
    if sender_name:
        label = "Ты" if sender_name == selected_user else "Собеседник"

    msg_text = f"{label}: {raw_text}".strip()
    followup_item = build_message_item(
        message,
        msg_text,
        sender_name=sender_name,
        sender_label=label,
    )

    data = await state.get_data()
    followup_messages = data.get("followup_messages", [])
    followup_limit_notified = data.get("followup_limit_notified", False)
    followup_limit_exceeded = data.get("followup_limit_exceeded", False)

    if followup_limit_exceeded:
        return

    if len(followup_messages) >= MAX_DIALOG_MESSAGES:
        await state.update_data(followup_limit_exceeded=True)
        if not followup_limit_notified:
            await state.update_data(followup_limit_notified=True)
            await message.answer(
                f"Ты превысил лимит — {MAX_DIALOG_MESSAGES} новых сообщений за один follow-up.\n"
                "Этот цикл сравнения отменён. Начни новый follow-up меньшим куском.",
                reply_markup=result_keyboard
            )
            await state.clear()
        return

    append_unique_message(followup_messages, followup_item)

    await state.update_data(followup_messages=followup_messages)

    current_count = len(followup_messages)
    last_progress_count = data.get("followup_last_progress_count", 0)
    if current_count > 0 and current_count % 3 == 0 and current_count != last_progress_count:
        await state.update_data(followup_last_progress_count=current_count)
        await message.answer(
            "Можешь переслать ещё новые сообщения или нажать <b>Проверить что изменилось</b>.",
            reply_markup=followup_keyboard
        )


@dp.message()
async def fallback_handler(message: Message):
    await message.answer(
        "Нажми <b>Старт</b>, затем перешли сообщения или голосовые из переписки.",
        reply_markup=start_keyboard
    )


async def main():
    configure_logging()
    init_db()
    logger.info("Starting Telegram bot")
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    try:
        await dp.start_polling(bot)
    except Exception:
        logger.exception("Bot stopped because of an unhandled exception")
        raise
    finally:
        logger.info("Stopping Telegram bot")
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
