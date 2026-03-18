import logging
import json
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI
from config import (
    ANALYSIS_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TRANSCRIPTION_MODEL,
)

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=ANALYSIS_TIMEOUT_SECONDS,
)
logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = (
    "Ты аналитик переписок. Твоя задача — не поддерживать пользователя, а честно разбирать диалог.\n"
    "Нельзя додумывать факты, которых нет в сообщениях.\n"
    "Опирайся только на текст диалога и метрики.\n\n"

    "Пиши жёстко, ясно и по делу.\n"
    "Не используй расплывчатые формулировки вроде: 'возможно', 'может быть', 'стоит подумать'.\n"
    "Не пиши как психолог из блога. Пиши как человек, который холодно оценивает ситуацию.\n\n"

    "Главная цель ответа:\n"
    "1. Объяснить, что реально происходит в диалоге\n"
    "2. Показать, кто в сильной, а кто в слабой позиции\n"
    "3. Сказать, есть ли интерес у собеседника\n"
    "4. Дать конкретное действие\n"
    "5. Дать готовый текст ответа\n\n"

    "Формат ответа для Telegram:\n"
    "Используй короткие блоки, без длинных абзацев.\n"
    "Используй разделители: ━━━━━━━━━━━━━━\n\n"

    "Структура ответа:\n\n"

    "📊 Разбор переписки\n\n"

    "━━━━━━━━━━━━━━\n"
    "🧠 Что происходит\n"
    "Коротко и прямо объясни суть диалога.\n\n"

    "━━━━━━━━━━━━━━\n"
    "⚖️ Позиции в разговоре\n"
    "- кто вкладывается больше\n"
    "- кто ведёт разговор\n"
    "- кто в более слабой позиции\n\n"

    "━━━━━━━━━━━━━━\n"
    "📉 Уровень интереса\n"
    "Оцени интерес собеседника по шкале от 0 до 10.\n"
    "Обязательно объясни, на каких фактах из диалога основана оценка.\n\n"

    "━━━━━━━━━━━━━━\n"
    "🚩 Тревожные сигналы\n"
    "Назови конкретные признаки: холодность, игнор, формальность, уход от темы, перекос инициативы.\n\n"

    "━━━━━━━━━━━━━━\n"
    "📌 Честный вывод\n"
    "Скажи прямо: есть ли смысл продолжать разговор или пользователь тратит время.\n\n"

    "━━━━━━━━━━━━━━\n"
    "🎯 Что делать сейчас\n"
    "Дай одно конкретное действие. Без вариантов и без воды.\n\n"

    "━━━━━━━━━━━━━━\n"
    "💬 Что можно ответить\n"
    "Напиши одно готовое сообщение, которое пользователь может отправить сразу.\n\n"

    "Жёсткие правила:\n"
    "- Не пиши нейтральную ерунду\n"
    "- Не повторяй одно и то же разными словами\n"
    "- Не придумывай эмоции без оснований\n"
    "- Не раздувай ответ\n"
    "- Если интерес слабый, скажи это прямо\n"
    "- Если пользователь сам ведёт себя слабо или навязчиво, скажи это прямо\n\n"
    "Диалог:\n"
)

ANALYSIS_DECISION_SUFFIX = (
    "\nОСОБО ВАЖНО:\n"
    "- Не просто анализируй диалог, а принимай решение за пользователя.\n"
    "- Если разговор бесперспективный, скажи это прямо.\n"
    "- Дай один лучший следующий шаг, а не список вариантов.\n"
    "- Сообщение для ответа давай только если реально стоит отвечать.\n"
)

FOLLOWUP_META_PROMPT = (
    "Верни только JSON-объект без пояснений и без markdown.\n"
    "Тебе нужно оценить follow-up в диалоге после действия пользователя.\n"
    "Формат JSON:\n"
    "{"
    "\"interest_before\": 0,"
    "\"interest_after\": 0,"
    "\"position_status\": \"без изменений\","
    "\"warmth_status\": \"без изменений\","
    "\"advice_effectiveness\": \"частично\""
    "}\n"
    "Допустимые значения:\n"
    "- interest_before: целое число от 0 до 10\n"
    "- interest_after: целое число от 0 до 10\n"
    "- position_status: улучшилась | без изменений | ухудшилась\n"
    "- warmth_status: теплее | без изменений | холоднее\n"
    "- advice_effectiveness: сработал | частично | не сработал\n"
    "Нельзя добавлять никакой другой текст."
)

FOLLOWUP_ANALYSIS_PROMPT = (
    "Ты анализируешь, что изменилось после действия пользователя в диалоге.\n"
    "Тебе нужно сравнить ситуацию ДО и ПОСЛЕ и принять решение за пользователя.\n"
    "Не размазывай выводы. Пиши коротко, жёстко и по делу.\n\n"
    "Что обязательно сделать:\n"
    "1. Сказать, сработало ли действие пользователя\n"
    "2. Показать, что именно изменилось в поведении собеседника\n"
    "3. Дать оценку интереса ДО и ПОСЛЕ по шкале от 0 до 10\n"
    "4. Явно сказать, улучшилась ли позиция пользователя\n"
    "5. Явно сказать, стал ли собеседник теплее или холоднее\n"
    "6. Указать, сработал ли совет из прошлого анализа\n"
    "7. Дать одно конкретное решение: продолжать, дожимать или прекращать\n"
    "8. Если нужен ответ пользователю, дать один лучший текст ответа\n\n"
    "Формат ответа:\n"
    "📉 ДО: X/10\n"
    "📈 ПОСЛЕ: Y/10\n"
    "ПОЗИЦИЯ: улучшилась / без изменений / ухудшилась\n"
    "ТЕПЛОТА: теплее / без изменений / холоднее\n"
    "СОВЕТ ИЗ ПРОШЛОГО АНАЛИЗА: сработал / частично / не сработал\n"
    "Что изменилось\n"
    "Сработало ли\n"
    "Что делать сейчас\n"
    "Что написать сейчас\n"
)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced_match = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates = [stripped, fenced_match]

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                return payload

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(stripped[start:end + 1])
        if isinstance(payload, dict):
            return payload

    raise ValueError("OpenAI returned invalid JSON for follow-up meta")


async def analyze_chat(text: str) -> str:
    logger.info("Starting OpenAI analysis with model=%s", OPENAI_MODEL)
    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=ANALYSIS_PROMPT + ANALYSIS_DECISION_SUFFIX + text,
    )

    analysis = response.output_text.strip()
    if not analysis:
        logger.error("OpenAI returned an empty analysis")
        raise RuntimeError("OpenAI returned an empty analysis")

    logger.info("OpenAI analysis completed successfully")
    return analysis


async def transcribe_voice(filename: str, audio_bytes: bytes) -> str:
    logger.info("Starting voice transcription with model=%s", OPENAI_TRANSCRIPTION_MODEL)

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    response = await client.audio.transcriptions.create(
        model=OPENAI_TRANSCRIPTION_MODEL,
        file=audio_file,
    )

    transcript = (getattr(response, "text", "") or "").strip()
    if not transcript:
        logger.error("OpenAI returned an empty transcription")
        raise RuntimeError("OpenAI returned an empty transcription")

    logger.info("Voice transcription completed successfully")
    return transcript


async def analyze_followup(text: str) -> tuple[dict[str, Any], str]:
    logger.info("Starting follow-up meta extraction with model=%s", OPENAI_MODEL)
    meta_response = await client.responses.create(
        model=OPENAI_MODEL,
        input=FOLLOWUP_META_PROMPT + "\n\n" + text,
    )

    meta_text = meta_response.output_text.strip()
    if not meta_text:
        logger.error("OpenAI returned empty follow-up meta")
        raise RuntimeError("OpenAI returned empty follow-up meta")

    followup_meta = extract_json_object(meta_text)

    logger.info("Starting follow-up analysis with model=%s", OPENAI_MODEL)
    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=FOLLOWUP_ANALYSIS_PROMPT + "\n\n" + text,
    )

    analysis = response.output_text.strip()
    if not analysis:
        logger.error("OpenAI returned an empty follow-up analysis")
        raise RuntimeError("OpenAI returned an empty follow-up analysis")

    logger.info("Follow-up analysis completed successfully")
    return followup_meta, analysis
