import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()

    if required and not value:
        raise ValueError(f"Не найден {name} в .env")

    if value:
        return value

    if default is not None:
        return default

    return ""


def get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом") from exc

    if value <= 0:
        raise ValueError(f"{name} должен быть больше 0")

    return value


def get_non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом") from exc

    if value < 0:
        raise ValueError(f"{name} не может быть отрицательным")

    return value


BOT_TOKEN = get_env("BOT_TOKEN", required=True)
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
OPENAI_MODEL = get_env("OPENAI_MODEL", default="gpt-4.1")
ANALYSIS_TIMEOUT_SECONDS = get_positive_int_env("ANALYSIS_TIMEOUT_SECONDS", default=45)
ANALYSIS_QUEUE_TIMEOUT_SECONDS = get_positive_int_env("ANALYSIS_QUEUE_TIMEOUT_SECONDS", default=3)
LOG_LEVEL = get_env("LOG_LEVEL", default="INFO").upper()
MAX_DIALOG_MESSAGES = get_positive_int_env("MAX_DIALOG_MESSAGES", default=150)
MAX_ANALYSIS_TEXT_CHARS = get_positive_int_env("MAX_ANALYSIS_TEXT_CHARS", default=14000)
MESSAGE_RATE_LIMIT_WINDOW_SECONDS = get_positive_int_env("MESSAGE_RATE_LIMIT_WINDOW_SECONDS", default=10)
MESSAGE_RATE_LIMIT_COUNT = get_positive_int_env("MESSAGE_RATE_LIMIT_COUNT", default=12)
MESSAGE_RATE_LIMIT_NOTICE_COOLDOWN = get_positive_int_env("MESSAGE_RATE_LIMIT_NOTICE_COOLDOWN", default=5)
ANALYSIS_COOLDOWN_SECONDS = get_non_negative_int_env("ANALYSIS_COOLDOWN_SECONDS", default=15)
MAX_CONCURRENT_ANALYSES = get_positive_int_env("MAX_CONCURRENT_ANALYSES", default=2)
