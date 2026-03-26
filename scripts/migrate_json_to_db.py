"""One-time migration script from data/app_state.json to database tables."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from utils.db import init_db, get_db_session
from utils.models import PredictionHistory, SessionToken, User


BASE_DIR = Path(__file__).resolve().parent.parent
STATE_PATH = BASE_DIR / "data" / "app_state.json"


def _parse_ts(value: str | None):
    if not value:
        return None
    # datetime.fromisoformat handles timezone offsets.
    from datetime import datetime

    return datetime.fromisoformat(value)


def main() -> None:
    init_db()
    if not STATE_PATH.exists():
        print(f"No JSON state file found at {STATE_PATH}. Nothing to migrate.")
        return

    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    users = payload.get("users", [])
    sessions = payload.get("sessions", {})

    with get_db_session() as db:
        existing = db.execute(select(User.id).limit(1)).first()
        if existing:
            print("Database already has data; skipping migration.")
            return

        email_to_user_id: dict[str, int] = {}

        for user in users:
            row = User(
                email=user["email"],
                salt=user["salt"],
                password_hash=user["password_hash"],
                joined_at=_parse_ts(user.get("joined_at")),
                last_login=_parse_ts(user.get("last_login")),
                last_prediction_at=_parse_ts(user.get("last_prediction_at")),
            )
            db.add(row)
            db.flush()
            email_to_user_id[row.email] = row.id

            for item in user.get("history", []):
                db.add(
                    PredictionHistory(
                        user_id=row.id,
                        timestamp=_parse_ts(item.get("timestamp")),
                        preview=item.get("preview", ""),
                        label=item.get("label", ""),
                        score=float(item.get("score", 0.0)),
                        source=item.get("source", "unknown"),
                        input_length=int(item.get("input_length", 0)),
                    )
                )

        for token, session in sessions.items():
            email = session.get("email")
            user_id = email_to_user_id.get(email)
            if not user_id:
                continue
            db.add(
                SessionToken(
                    token=token,
                    user_id=user_id,
                    created_at=_parse_ts(session.get("created_at")),
                )
            )

    print(f"Migrated {len(users)} users from {STATE_PATH} into database.")


if __name__ == "__main__":
    main()
