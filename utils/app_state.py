"""Database-backed app state for auth and prediction history."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone

from sqlalchemy import delete, desc, func, select

from .db import get_db_session
from .models import PredictionHistory, SessionToken, User

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
UNKNOWN_USER_EMAIL = "unknown@guest.local"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()


def _issue_token(user_id: int) -> SessionToken:
    return SessionToken(
        token=secrets.token_urlsafe(32),
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
    )


def _validate_credentials(email: str, password: str) -> tuple[str, str]:
    normalized_email = email.strip().lower()
    if not EMAIL_PATTERN.match(normalized_email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return normalized_email, password


def _public_user(user: User, prediction_count: int) -> dict:
    return {
        "email": user.email,
        "joined_at": user.joined_at.isoformat() if user.joined_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "last_prediction_at": user.last_prediction_at.isoformat() if user.last_prediction_at else None,
        "prediction_count": int(prediction_count),
    }


def _ensure_unknown_user(db) -> User:
    user = db.execute(select(User).where(User.email == UNKNOWN_USER_EMAIL)).scalar_one_or_none()
    if user is not None:
        return user

    # Placeholder credentials for anonymous usage tracking only.
    salt = secrets.token_hex(16)
    now = datetime.now(timezone.utc)
    user = User(
        email=UNKNOWN_USER_EMAIL,
        salt=salt,
        password_hash=_hash_password(secrets.token_urlsafe(24), salt),
        joined_at=now,
        last_login=None,
        last_prediction_at=None,
    )
    db.add(user)
    db.flush()
    return user


def create_user(email: str, password: str) -> tuple[str, dict]:
    normalized_email, password = _validate_credentials(email, password)
    with get_db_session() as db:
        existing = db.execute(select(User).where(User.email == normalized_email)).scalar_one_or_none()
        if existing is not None:
            raise ValueError("An account with that email already exists.")

        salt = secrets.token_hex(16)
        now = datetime.now(timezone.utc)
        user = User(
            email=normalized_email,
            salt=salt,
            password_hash=_hash_password(password, salt),
            joined_at=now,
            last_login=now,
            last_prediction_at=None,
        )
        db.add(user)
        db.flush()

        session = _issue_token(user.id)
        db.add(session)
        db.flush()
        return session.token, _public_user(user, prediction_count=0)


def authenticate_user(email: str, password: str) -> tuple[str, dict]:
    normalized_email, password = _validate_credentials(email, password)
    with get_db_session() as db:
        user = db.execute(select(User).where(User.email == normalized_email)).scalar_one_or_none()
        if user is None:
            raise ValueError("Invalid email or password.")

        expected_hash = _hash_password(password, user.salt)
        if not hmac.compare_digest(expected_hash, user.password_hash):
            raise ValueError("Invalid email or password.")

        session = _issue_token(user.id)
        user.last_login = datetime.now(timezone.utc)
        db.add(session)
        count_stmt = select(func.count(PredictionHistory.id)).where(PredictionHistory.user_id == user.id)
        prediction_count = db.execute(count_stmt).scalar_one() or 0
        return session.token, _public_user(user, prediction_count=prediction_count)


def get_user_by_token(token: str) -> dict | None:
    with get_db_session() as db:
        session = db.execute(select(SessionToken).where(SessionToken.token == token)).scalar_one_or_none()
        if session is None:
            return None

        user = db.execute(select(User).where(User.id == session.user_id)).scalar_one_or_none()
        if user is None:
            return None
        count_stmt = select(func.count(PredictionHistory.id)).where(PredictionHistory.user_id == user.id)
        prediction_count = db.execute(count_stmt).scalar_one() or 0
        return _public_user(user, prediction_count=prediction_count)


def record_prediction(email: str | None, text: str, label: str, score: float, source: str) -> dict:
    with get_db_session() as db:
        if email:
            user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if user is None:
                raise ValueError("User account not found.")
        else:
            user = _ensure_unknown_user(db)

        timestamp = datetime.now(timezone.utc)
        entry = {
            "timestamp": timestamp.isoformat(),
            "preview": " ".join(text.strip().split())[:180],
            "label": label,
            "score": round(float(score), 4),
            "source": source,
            "input_length": len(text),
        }

        record = PredictionHistory(
            user_id=user.id,
            timestamp=timestamp,
            preview=entry["preview"],
            label=label,
            score=entry["score"],
            source=source,
            input_length=entry["input_length"],
        )
        db.add(record)
        user.last_prediction_at = timestamp
        db.flush()

        # Keep the latest 100 entries per user, mirroring previous behavior.
        stale_ids = db.execute(
            select(PredictionHistory.id)
            .where(PredictionHistory.user_id == user.id)
            .order_by(desc(PredictionHistory.timestamp), desc(PredictionHistory.id))
            .offset(100)
        ).scalars().all()
        if stale_ids:
            db.execute(delete(PredictionHistory).where(PredictionHistory.id.in_(stale_ids)))

        return entry


def get_user_history(email: str) -> list[dict]:
    with get_db_session() as db:
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            raise ValueError("User account not found.")

        rows = db.execute(
            select(PredictionHistory)
            .where(PredictionHistory.user_id == user.id)
            .order_by(desc(PredictionHistory.timestamp), desc(PredictionHistory.id))
            .limit(100)
        ).scalars().all()

        return [
            {
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "preview": row.preview,
                "label": row.label,
                "score": float(row.score),
                "source": row.source,
                "input_length": int(row.input_length),
            }
            for row in rows
        ]


def get_public_user_summaries() -> list[dict]:
    with get_db_session() as db:
        prediction_count_expr = func.count(PredictionHistory.id).label("prediction_count")
        rows = db.execute(
            select(User, prediction_count_expr)
            .outerjoin(PredictionHistory, PredictionHistory.user_id == User.id)
            .group_by(User.id)
            .order_by(prediction_count_expr.desc(), User.joined_at.desc())
        ).all()
        return [_public_user(user, int(prediction_count or 0)) for user, prediction_count in rows]