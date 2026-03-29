from __future__ import annotations

from typing import Any


def truncate_for_char_features(texts: Any, max_chars: int) -> list[str]:
    """Truncate raw text inputs before char-level hashing to limit memory usage."""

    return [str(item)[:max_chars] for item in texts]