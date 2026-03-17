import asyncio
import re
from aiogram import Bot, Dispatcher, F
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

from config import BOT_TOKEN
from ai_analyzer import analyze_chat

dp = Dispatcher()

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

    if not preview_waiting_sent:
        await state.update_data(preview_waiting_sent=True)
        await message.answer(
            f"Пока вижу участников: {', '.join(participants)}\n"
            "Перешли ещё сообщение второго участника.",
            reply_markup=preview_keyboard
        )


def calculate_dialog_metrics(messages, selected_user):
    user_messages = []
    other_messages = []

    for m in messages:
        text = m["text"]
        if m["text"].startswith("Ты:"):
            user_messages.append(text)
        else:
            other_messages.append(text)

    user_count = len(user_messages)
    other_count = len(other_messages)

    user_words = sum(len(m.split()) for m in user_messages)
    other_words = sum(len(m.split()) for m in other_messages)

    user_avg_len = user_words // user_count if user_count else 0
    other_avg_len = other_words // other_count if other_count else 0

    user_questions = sum("?" in m for m in user_messages)
    other_questions = sum("?" in m for m in other_messages)

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
    
    messages = data.get("messages", [])
    selected_user = data.get("selected_user")
    participants = data.get("participants", [])

    if not messages:
        await message.answer(
            "Ты ещё не переслал ни одного сообщения.",
            reply_markup=collect_keyboard
        )
        return

    messages.sort(key=lambda x: x["date"])

    other_participants = [p for p in participants if p != selected_user]
    other_name = other_participants[0] if other_participants else "Собеседник"

    full_text = (
        f"Пользователь бота: {selected_user}\n"
        f"Собеседник: {other_name}\n\n"
        "Диалог:\n"
        + "\n".join(m["text"] for m in messages)
    )

    await message.answer("🧠 Анализирую переписку...\nЭто может занять несколько секунд.")

    try:
        metrics = calculate_dialog_metrics(messages, selected_user)
        analysis = analyze_chat(
            "КОНТЕКСТ ДИАЛОГА:\n"
            + full_text
            + "\n\nМЕТРИКИ ДИАЛОГА:\n"
            + metrics
        )
        await message.answer(analysis, reply_markup=result_keyboard)
    except Exception:
        await message.answer(
            "Сейчас не могу выполнить анализ. "
            "Проверь лимит и попробуй снова.",
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

    if len(messages) >= 60:
        await state.update_data(limit_exceeded=True)

        if not limit_notified:
            await state.update_data(limit_notified=True)
            await message.answer(
                "Ты превысил лимит — 60 сообщений за один анализ.\n"
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
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())