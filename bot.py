import asyncio
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
    MESSAGE_RATE_LIMIT_COUNT,
    MESSAGE_RATE_LIMIT_NOTICE_COOLDOWN,
    MESSAGE_RATE_LIMIT_WINDOW_SECONDS,
)
from ai_analyzer import analyze_chat


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Разобрать переписку")]
    ],
    resize_keyboard=True
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
        [KeyboardButton(text="Новый анализ")]
    ],
    resize_keyboard=True
)


class AnalyzeState(StatesGroup):
    collecting_preview = State()
    choosing_role = State()
    collecting_messages = State()

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
        "Привет. Я бот для разбора переписок.\n\n"
        "Нажми кнопку ниже, чтобы начать.",
        reply_markup=start_keyboard
    )


@dp.message(F.text.in_(["Разобрать переписку", "Новый анализ"]))
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
        "Перешли 2–4 сообщения из переписки, чтобы я определил участников.",
        reply_markup=preview_keyboard
    )

@dp.message(AnalyzeState.choosing_role)
async def choose_role_handler(message: Message, state: FSMContext):
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
        converted_messages.append({
            "text": f"{label}: {item['text']}",
            "date": item["date"]
        })

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
    if message.text in ["Закончить и проанализировать", "Отмена"]:
        return

    raw_text = extract_text(message)
    sender_name = extract_sender_name(message)

    if not raw_text or not sender_name:
        await message.answer(
            "На этом этапе перешли текстовые сообщения из переписки, чтобы я увидел участников.",
            reply_markup=preview_keyboard
        )
        return

    data = await state.get_data()
    preview_messages = data.get("preview_messages", [])
    participants = data.get("participants", [])
    role_prompt_sent = data.get("role_prompt_sent", False)
    preview_waiting_sent = data.get("preview_waiting_sent", False)

    preview_item = {
        "sender_name": sender_name,
        "text": raw_text,
        "date": message.date.timestamp()
    }

    if preview_item not in preview_messages:
        preview_messages.append(preview_item)

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

@dp.message(F.text == "Закончить и проанализировать")
async def finish_handler(message: Message, state: FSMContext):
    current_state = await state.get_state()

    if current_state != AnalyzeState.collecting_messages.state:
        await message.answer(
            "Сейчас нечего анализировать. Нажми «Разобрать переписку».",
            reply_markup=start_keyboard
        )
        return

    data = await state.get_data()
    limit_exceeded = data.get("limit_exceeded", False)

    if limit_exceeded:
        await message.answer(
            "Лимит сообщений был превышен. Этот набор не будет проанализирован.\n"
            "Нажми «Разобрать переписку» и начни заново.",
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
    if message.text in ["Закончить и проанализировать", "Отмена"]:
        return

    raw_text = extract_text(message)
    sender_name = extract_sender_name(message)

    if not raw_text:
        await message.answer(
            "Это сообщение не содержит текста. Перешли текстовое сообщение или нажми «Закончить и проанализировать».",
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
                "Этот анализ отменён. Нажми «Разобрать переписку» и отправь переписку заново в пределах лимита.",
                reply_markup=start_keyboard
            )
            await state.clear()
        return

    if sender_name:
        label = "Ты" if sender_name == selected_user else "Собеседник"
    else:
        label = "Собеседник"

    msg_text = f"{label}: {raw_text}".strip()

    msg = {
        "text": msg_text,
        "date": message.date.timestamp()
    }

    existing_texts = {m["text"] for m in messages}

    if msg_text not in existing_texts:
        messages.append(msg)

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


@dp.message()
async def fallback_handler(message: Message):
    await message.answer(
        "Нажми «Разобрать переписку», затем перешли сообщения из переписки.",
        reply_markup=start_keyboard
    )


async def main():
    configure_logging()
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
