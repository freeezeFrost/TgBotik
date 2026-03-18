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

REANALYZE_FOOTER = "Если ситуация изменится или придёт новый ответ — проанализируй заново."
FOLLOWUP_CTA_FOOTER = "Хочешь проверить, как изменится ситуация после твоего ответа — скинь новый диалог."

ANALYSIS_PROMPT = (
    "Твоя задача — показать пользователю реальную картину так, чтобы он узнал себя в этом и понял, что происходит на самом деле."
    "Ты аналитик переписок. Твоя задача — не поддерживать пользователя, а давать честные и жёсткие выводы.\n"
    "Ты не поддерживаешь пользователя и не утешаешь его. Ты разбираешь ситуацию и говоришь правду, даже если она неприятная.\n"
    "Не придумывай факты, которых нет в тексте, но анализируй поведенческие сигналы в диалоге.\n\n"

    "ЖЁСТКИЕ ПРАВИЛА:\n"
    "- Не давай мягкие советы\n"
    "- Не оставляй выбор пользователю\n"
    "- Всегда давай конкретное решение\n"
    "- Если ситуация плохая — говори прямо\n"
    "- Если пользователь косячит — говори прямо\n\n"

    "Главная цель:\n"
    "1. Понять, что реально происходит\n"
    "2. Дать чёткий вердикт\n"
    "3. Объяснить ошибки пользователя\n"
    "4. Дать конкретное действие\n"
    "5. Показать последствия\n"
    "6. Дать готовый ответ\n\n"

    "Формат ответа (строго соблюдать):\n\n"

    "━━━━━━━━━━━━━━\n"
    "🧠 Что происходит\n"
    "Коротко: суть диалога без воды.\n\n"

    "━━━━━━━━━━━━━━\n"
    "❗ Вердикт\n"
    "Выбери ОДИН вариант:\n"
    "- продолжать общение\n"
    "- снизить активность\n"
    "- прекратить общение\n"
    "Опирайся только на сигналы из диалога. Анализируй поведенческие сигналы: инициатива, скорость ответов, длина сообщений, вовлечённость. Не придумывай факты, которых нет в тексте.\n"
    "Без объяснений в стиле 'смотря как'. Только чётко.\n\n"

    "━━━━━━━━━━━━━━\n"
    "⚠️ Где ты косячишь\n"
    "Если пользователь делает ошибки — укажи их прямо.\n"
    "Покажи, как это выглядит со стороны и почему это снижает интерес к пользователю.\n"
    "Конкретно: навязывается, тянет диалог, пишет лишнее, оправдывается, давит и т.д.\n\n"

    "━━━━━━━━━━━━━━\n"
    "🎯 Что делать сейчас\n"
    "Дай одно чёткое действие:\n"
    "- что именно сделать\n"
    "- на какой срок\n"
    "- зачем\n\n"

    "━━━━━━━━━━━━━━\n"
    "🔮 Что будет дальше\n"
    "1. Если ничего не менять\n"
    "2. Если сделать как ты сказал\n"
    "Коротко и по факту.\n\n"

    "━━━━━━━━━━━━━━\n"
    "💬 Что написать\n"
    "Дай ОДИН лучший вариант сообщения, который максимально усиливает позицию пользователя.\n"
    "Если писать не нужно — прямо скажи НЕ ПИСАТЬ.\n\n"

    "ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА:\n"
    "- Не повторяй одно и то же\n"
    "- Не раздувай текст\n"
    "- Пиши коротко\n"
    "- Делай вывод строго по фактам диалога, а не по шаблону\n"
    "- Если пользователь ведёт себя слабо — укажи это\n\n"

    "Диалог:\n"
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
    "8. Если нужен ответ пользователю, дать один рабочий текст ответа\n\n"
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

REPLY_VARIANTS_PROMPT = (
    "Ты помогаешь подготовить 3 разных варианта ответа на основе уже сделанного разбора переписки.\n"
    "Нельзя повторять варианты почти дословно.\n"
    "Каждый вариант должен отличаться по тону или тактике:\n"
    "1. спокойный\n"
    "2. увереннее и короче\n"
    "3. более холодный и дистанционный\n\n"
    "Если по ситуации писать вообще не нужно, вместо вариантов напиши только:\n"
    "НЕ ПИСАТЬ\n"
    "И одной короткой строкой почему.\n\n"
    "Если варианты нужны, формат строго такой:\n"
    "Вариант 1:\n"
    "<текст>\n\n"
    "Вариант 2:\n"
    "<текст>\n\n"
    "Вариант 3:\n"
    "<текст>\n"
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
        input=ANALYSIS_PROMPT + text,
    )

    analysis = response.output_text.strip()
    if not analysis:
        logger.error("OpenAI returned an empty analysis")
        raise RuntimeError("OpenAI returned an empty analysis")

    footer_parts: list[str] = []
    if REANALYZE_FOOTER not in analysis:
        footer_parts.append(REANALYZE_FOOTER)
    if FOLLOWUP_CTA_FOOTER not in analysis:
        footer_parts.append(FOLLOWUP_CTA_FOOTER)
    if footer_parts:
        analysis = f"{analysis}\n\n" + "\n".join(footer_parts)

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

    footer_parts: list[str] = []
    if REANALYZE_FOOTER not in analysis:
        footer_parts.append(REANALYZE_FOOTER)
    if FOLLOWUP_CTA_FOOTER not in analysis:
        footer_parts.append(FOLLOWUP_CTA_FOOTER)
    if footer_parts:
        analysis = f"{analysis}\n\n" + "\n".join(footer_parts)

    logger.info("Follow-up analysis completed successfully")
    return followup_meta, analysis


async def generate_reply_variants(text: str) -> str:
    logger.info("Starting reply variants generation with model=%s", OPENAI_MODEL)
    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=REPLY_VARIANTS_PROMPT + "\n\n" + text,
    )

    variants = response.output_text.strip()
    if not variants:
        logger.error("OpenAI returned empty reply variants")
        raise RuntimeError("OpenAI returned empty reply variants")

    logger.info("Reply variants generated successfully")
    return variants
