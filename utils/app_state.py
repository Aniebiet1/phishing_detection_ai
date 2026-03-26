"""Simple JSON-backed app state for auth and prediction history."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


BASE_DIR = Path(__file__).resolve().parent.parent
STATE_PATH = BASE_DIR / "data" / "app_state.json"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
STATE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict:
    return {"users": [], "sessions": {}}


def _ensure_state_file() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        STATE_PATH.write_text(json.dumps(_default_state(), indent=2), encoding="utf-8")


def _load_state() -> dict:
    _ensure_state_file()
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _find_user(state: dict, email: str) -> dict | None:
    for user in state["users"]:
        if user["email"] == email:
            return user
    return None


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()


def _issue_token(state: dict, email: str) -> str:
    token = secrets.token_urlsafe(32)
    state["sessions"][token] = {"email": email, "created_at": _utc_now()}
    return token


def _validate_credentials(email: str, password: str) -> tuple[str, str]:
    normalized_email = email.strip().lower()
    if not EMAIL_PATTERN.match(normalized_email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return normalized_email, password


def _public_user(user: dict) -> dict:
    history = user.get("history", [])
    return {
        "email": user["email"],
        "joined_at": user["joined_at"],
        "last_login": user.get("last_login"),
        "last_prediction_at": user.get("last_prediction_at"),
        "prediction_count": len(history),
    }


def create_user(email: str, password: str) -> tuple[str, dict]:
    normalized_email, password = _validate_credentials(email, password)
    with STATE_LOCK:
        state = _load_state()
        if _find_user(state, normalized_email) is not None:
            raise ValueError("An account with that email already exists.")

        salt = secrets.token_hex(16)
        user = {
            "email": normalized_email,
            "salt": salt,
            "password_hash": _hash_password(password, salt),
            "joined_at": _utc_now(),
            "last_login": None,
            "last_prediction_at": None,
            "history": [],
        }
        state["users"].append(user)
        token = _issue_token(state, normalized_email)
        user["last_login"] = _utc_now()
        _save_state(state)
        return token, _public_user(user)


def authenticate_user(email: str, password: str) -> tuple[str, dict]:
    normalized_email, password = _validate_credentials(email, password)
    with STATE_LOCK:
        state = _load_state()
        user = _find_user(state, normalized_email)
        if user is None:
            raise ValueError("Invalid email or password.")

        expected_hash = _hash_password(password, user["salt"])
        if not hmac.compare_digest(expected_hash, user["password_hash"]):
            raise ValueError("Invalid email or password.")

        token = _issue_token(state, normalized_email)
        user["last_login"] = _utc_now()
        _save_state(state)
        return token, _public_user(user)


def get_user_by_token(token: str) -> dict | None:
    with STATE_LOCK:
        state = _load_state()
        session = state["sessions"].get(token)
        if session is None:
            return None
        user = _find_user(state, session["email"])
        return _public_user(user) if user is not None else None


def record_prediction(email: str, text: str, label: str, score: float, source: str) -> dict:
    with STATE_LOCK:
        state = _load_state()
        user = _find_user(state, email)
        if user is None:
            raise ValueError("User account not found.")

        entry = {
            "timestamp": _utc_now(),
            "preview": " ".join(text.strip().split())[:180],
            "label": label,
            "score": round(float(score), 4),
            "source": source,
            "input_length": len(text),
        }
        history = user.setdefault("history", [])
        history.insert(0, entry)
        del history[100:]
        user["last_prediction_at"] = entry["timestamp"]
        _save_state(state)
        return entry


def get_user_history(email: str) -> list[dict]:
    with STATE_LOCK:
        state = _load_state()
        user = _find_user(state, email)
        if user is None:
            raise ValueError("User account not found.")
        return user.get("history", [])


def get_public_user_summaries() -> list[dict]:
    with STATE_LOCK:
        state = _load_state()
        users = [_public_user(user) for user in state["users"]]
        return sorted(users, key=lambda item: item["prediction_count"], reverse=True)