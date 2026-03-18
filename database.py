import json
import sqlite3
import time
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).with_name("bot_data.sqlite3")


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                run_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                payload TEXT NOT NULL,
                previous_run_id INTEGER,
                interest_score INTEGER,
                interest_score_before INTEGER,
                interest_score_after INTEGER,
                position_status TEXT,
                warmth_status TEXT,
                advice_effectiveness TEXT,
                progress_summary TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_run_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                scope TEXT NOT NULL,
                position INTEGER NOT NULL,
                sender_label TEXT,
                sender_name TEXT,
                message_text TEXT NOT NULL,
                message_date REAL,
                message_key TEXT,
                FOREIGN KEY(run_id) REFERENCES analysis_runs(id)
            )
            """
        )
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(analysis_runs)")
        }
        required_columns = {
            "previous_run_id": "INTEGER",
            "interest_score": "INTEGER",
            "interest_score_before": "INTEGER",
            "interest_score_after": "INTEGER",
            "position_status": "TEXT",
            "warmth_status": "TEXT",
            "advice_effectiveness": "TEXT",
            "progress_summary": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE analysis_runs ADD COLUMN {column_name} {column_type}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_user_created
            ON analysis_runs (user_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_user_type_created
            ON analysis_runs (user_id, run_type, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_previous_run
            ON analysis_runs (previous_run_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_run_messages_run_scope_position
            ON analysis_run_messages (run_id, scope, position)
            """
        )


def serialize_message_key(message: dict[str, Any]) -> str | None:
    chat_id = message.get("chat_id")
    message_id = message.get("message_id")
    if chat_id is not None and message_id is not None:
        return f"telegram:{chat_id}:{message_id}"

    sender_name = message.get("sender_name")
    forward_date = message.get("forward_date")
    message_text = message.get("text")
    if sender_name and forward_date is not None and message_text:
        return f"forward:{sender_name}:{forward_date}:{message_text}"

    return None


def build_payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    stored_payload = dict(payload)
    stored_payload.pop("messages", None)
    stored_payload.pop("new_messages", None)
    return stored_payload


def save_run_messages(connection: sqlite3.Connection, run_id: int, payload: dict[str, Any]) -> None:
    message_groups = (
        ("messages", payload.get("messages", [])),
        ("new_messages", payload.get("new_messages", [])),
    )

    for scope, messages in message_groups:
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                continue

            connection.execute(
                """
                INSERT INTO analysis_run_messages (
                    run_id,
                    scope,
                    position,
                    sender_label,
                    sender_name,
                    message_text,
                    message_date,
                    message_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    scope,
                    position,
                    message.get("sender_label"),
                    message.get("sender_name"),
                    message.get("text", ""),
                    message.get("date"),
                    serialize_message_key(message),
                ),
            )


def load_run_messages(connection: sqlite3.Connection, run_id: int) -> dict[str, list[dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT scope, sender_label, sender_name, message_text, message_date, message_key
        FROM analysis_run_messages
        WHERE run_id = ?
        ORDER BY scope, position
        """,
        (run_id,),
    ).fetchall()

    grouped_messages: dict[str, list[dict[str, Any]]] = {
        "messages": [],
        "new_messages": [],
    }

    for row in rows:
        grouped_messages.setdefault(row["scope"], []).append(
            {
                "text": row["message_text"],
                "date": row["message_date"],
                "sender_label": row["sender_label"],
                "sender_name": row["sender_name"],
                "message_key": row["message_key"],
            }
        )

    return grouped_messages


def hydrate_payload_messages(
    connection: sqlite3.Connection,
    run_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    stored_messages = load_run_messages(connection, run_id)
    for scope in ("messages", "new_messages"):
        if stored_messages.get(scope):
            payload[scope] = stored_messages[scope]
        else:
            payload.setdefault(scope, [])
    return payload


def save_analysis_run(user_id: int, run_type: str, payload: dict[str, Any]) -> int:
    serialized_payload = json.dumps(build_payload_for_storage(payload), ensure_ascii=False)
    previous_run_id = payload.get("previous_run_id")
    interest_score = payload.get("interest_score")
    interest_score_before = payload.get("interest_score_before")
    interest_score_after = payload.get("interest_score_after")
    position_status = payload.get("position_status")
    warmth_status = payload.get("warmth_status")
    advice_effectiveness = payload.get("advice_effectiveness")
    progress_summary = payload.get("progress_summary")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_runs (
                user_id,
                run_type,
                created_at,
                payload,
                previous_run_id,
                interest_score,
                interest_score_before,
                interest_score_after,
                position_status,
                warmth_status,
                advice_effectiveness,
                progress_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                run_type,
                int(time.time()),
                serialized_payload,
                previous_run_id,
                interest_score,
                interest_score_before,
                interest_score_after,
                position_status,
                warmth_status,
                advice_effectiveness,
                progress_summary,
            ),
        )
        run_id = int(cursor.lastrowid)
        save_run_messages(connection, run_id, payload)
        return run_id


def get_latest_analysis_run(user_id: int, run_type: str | None = None) -> dict[str, Any] | None:
    query = """
        SELECT
            id,
            user_id,
            run_type,
            created_at,
            payload,
            previous_run_id,
            interest_score,
            interest_score_before,
            interest_score_after,
            position_status,
            warmth_status,
            advice_effectiveness,
            progress_summary
        FROM analysis_runs
        WHERE user_id = ?
    """
    params: list[Any] = [user_id]

    if run_type is not None:
        query += " AND run_type = ?"
        params.append(run_type)

    query += " ORDER BY created_at DESC, id DESC LIMIT 1"

    with get_connection() as connection:
        row = connection.execute(query, params).fetchone()

        if row is None:
            return None

        payload = json.loads(row["payload"])
        payload["id"] = row["id"]
        payload["user_id"] = row["user_id"]
        payload["run_type"] = row["run_type"]
        payload["created_at"] = row["created_at"]
        payload.setdefault("previous_run_id", row["previous_run_id"])
        payload.setdefault("interest_score", row["interest_score"])
        payload.setdefault("interest_score_before", row["interest_score_before"])
        payload.setdefault("interest_score_after", row["interest_score_after"])
        payload.setdefault("position_status", row["position_status"])
        payload.setdefault("warmth_status", row["warmth_status"])
        payload.setdefault("advice_effectiveness", row["advice_effectiveness"])
        payload.setdefault("progress_summary", row["progress_summary"])
        return hydrate_payload_messages(connection, row["id"], payload)


def get_analysis_run_by_id(run_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                user_id,
                run_type,
                created_at,
                payload,
                previous_run_id,
                interest_score,
                interest_score_before,
                interest_score_after,
                position_status,
                warmth_status,
                advice_effectiveness,
                progress_summary
            FROM analysis_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        payload = json.loads(row["payload"])
        payload["id"] = row["id"]
        payload["user_id"] = row["user_id"]
        payload["run_type"] = row["run_type"]
        payload["created_at"] = row["created_at"]
        payload.setdefault("previous_run_id", row["previous_run_id"])
        payload.setdefault("interest_score", row["interest_score"])
        payload.setdefault("interest_score_before", row["interest_score_before"])
        payload.setdefault("interest_score_after", row["interest_score_after"])
        payload.setdefault("position_status", row["position_status"])
        payload.setdefault("warmth_status", row["warmth_status"])
        payload.setdefault("advice_effectiveness", row["advice_effectiveness"])
        payload.setdefault("progress_summary", row["progress_summary"])
        return hydrate_payload_messages(connection, row["id"], payload)
