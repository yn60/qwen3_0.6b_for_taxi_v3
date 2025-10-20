"""Action coercion utilities shared between backend and notebooks.

This module contains logic to robustly coerce an arbitrary LLM output
into a valid Taxi-v3 discrete action code (0-5).
"""

from __future__ import annotations

import re
from typing import Optional


ACTION_LABELS = {
    0: "Move South",
    1: "Move North",
    2: "Move East",
    3: "Move West",
    4: "Pick up passenger",
    5: "Drop off passenger",
}


_WORD_NUMBER_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


_DIRECTION_KEYWORDS = {
    0: {"south", "down"},
    1: {"north", "up"},
    2: {"east", "right"},
    3: {"west", "left"},
}


_PICK_KEYWORDS = {"pick", "pickup", "pick-up", "grab", "collect"}
_DROP_KEYWORDS = {"drop", "dropoff", "drop-off", "release", "deliver"}


def coerce_action(candidate) -> Optional[int]:
    """Coerce an arbitrary candidate into a valid action code (0..5) or None.

    Handles ints, floats, strings with numbers or keywords, lists/tuples, and dicts
    with common fields like "action", "value", "direction", etc.
    """
    if candidate is None or isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return candidate if candidate in ACTION_LABELS else None
    if isinstance(candidate, dict):
        for key in (
            "code",
            "id",
            "index",
            "value",
            "action",
            "action_code",
            "action_id",
            "choice",
            "selection",
        ):
            if key in candidate:
                coerced = coerce_action(candidate[key])
                if coerced is not None:
                    return coerced
        for key in ("direction", "target", "name", "label"):
            if key in candidate:
                coerced = coerce_action(candidate[key])
                if coerced is not None:
                    return coerced
        return None
    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            coerced = coerce_action(item)
            if coerced is not None:
                return coerced
        return None
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"null", "none"}:
            return None

        digit_match = re.search(r"\b([0-5])\b", lowered)
        if digit_match:
            return int(digit_match.group(1))

        for word, value in _WORD_NUMBER_TO_INT.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                if value in ACTION_LABELS:
                    return value

        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        tokens = [tok for tok in cleaned.split() if tok]
        token_set = set(tokens)

        if token_set & _PICK_KEYWORDS or (
            "pick" in token_set and ("up" in token_set or "passenger" in token_set)
        ):
            return 4
        if token_set & _DROP_KEYWORDS or (
            "drop" in token_set and ("off" in token_set or "passenger" in token_set)
        ):
            return 5

        for code, keywords in _DIRECTION_KEYWORDS.items():
            if token_set & keywords:
                return code

        if {"move", "s"}.issubset(token_set) or {"action", "s"}.issubset(token_set):
            return 0
        if {"move", "n"}.issubset(token_set) or {"action", "n"}.issubset(token_set):
            return 1
        if {"move", "e"}.issubset(token_set) or {"action", "e"}.issubset(token_set):
            return 2
        if {"move", "w"}.issubset(token_set) or {"action", "w"}.issubset(token_set):
            return 3

        for code, keywords in _DIRECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    return code

        if any(keyword in lowered for keyword in _PICK_KEYWORDS):
            return 4
        if any(keyword in lowered for keyword in _DROP_KEYWORDS):
            return 5

        return None
    if isinstance(candidate, float) and candidate.is_integer():
        candidate_int = int(candidate)
        return candidate_int if candidate_int in ACTION_LABELS else None
    try:
        candidate_int = int(candidate)
    except (TypeError, ValueError):
        return None
    return candidate_int if candidate_int in ACTION_LABELS else None


__all__ = ["ACTION_LABELS", "coerce_action"]
