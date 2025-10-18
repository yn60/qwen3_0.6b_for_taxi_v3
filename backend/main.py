import re
import threading
from typing import Dict, Optional

from pathlib import Path

import json

from flask import Flask, Response, jsonify, render_template, stream_with_context

from taxi.environment import TaxiEnvironment
from taxi.state_utils import decode_state, describe_state_for_llm, get_prompt
from llm.client import stream_qwen_action


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR.parent / "frontend" / "templates"
STATIC_DIR = BASE_DIR.parent / "frontend" / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

app.logger.info("Templates directory: %s", TEMPLATE_DIR)
app.logger.info("Static directory: %s", STATIC_DIR)


class GameState:
    def __init__(self) -> None:
        self.env = TaxiEnvironment()
        self.state: int = self.env.observation
        self.terminated: bool = False
        self.total_reward: float = 0.0
        self.steps: int = 0
        self.last_llm_response: Dict[str, object] = {}
        self.last_render: str = self.env.render()
        self.last_llm_error: Optional[str] = None
        self.lock = threading.Lock()


game = GameState()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset_game():
    with game.lock:
        game.state = game.env.reset()
        game.terminated = False
        game.total_reward = 0.0
        game.steps = 0
        game.last_llm_response = {}
        game.last_render = game.env.render()
        game.last_llm_error = None
    return jsonify(get_current_game_state())


@app.route("/step", methods=["POST"])
def take_step():
    def encode_event(payload: Dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    def stream_response():
        with game.lock:
            if game.terminated:
                yield encode_event({"type": "error", "error": "Game is over. Please reset."})
                return

            state_description = describe_state_for_llm(decode_state(game.state))
            prompt = get_prompt(state_description)

        for event in stream_qwen_action(prompt):
            kind = event.get("type")

            if kind == "token":
                yield encode_event({"type": "token", "token": event.get("token", "")})
                continue

            if kind == "error":
                yield encode_event({"type": "error", "error": event.get("error", "Model error")})
                return

            if kind == "result":
                llm_response = event.get("data", {}) or {}

                with game.lock:
                    game.last_llm_response = llm_response
                    game.last_llm_error = None
                    action = _coerce_action(llm_response.get("action"))
                    if action is None:
                        game.last_llm_error = (
                            "Invalid action from LLM output; please try again."
                        )
                        final_payload = get_current_game_state()
                        final_payload["llm_raw_action"] = llm_response.get("action")
                        final_payload["llm_raw_response"] = llm_response.get("raw_response")
                    else:
                        llm_response["action"] = action
                        new_state, reward, terminated = game.env.step(action)
                        game.state = new_state
                        game.total_reward += reward
                        game.steps += 1
                        game.terminated = terminated
                        game.last_render = game.env.render()
                        final_payload = get_current_game_state()
                        final_payload["llm_raw_action"] = action
                        final_payload["llm_raw_response"] = llm_response.get("raw_response")

                yield encode_event({"type": "final", "payload": final_payload})
                return

        yield encode_event({"type": "error", "error": "Model did not produce a result."})

    response = Response(
        stream_with_context(stream_response()),
        mimetype="application/x-ndjson",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/state", methods=["GET"])
def get_state():
    with game.lock:
        return jsonify(get_current_game_state())


def get_current_game_state():
    decoded = decode_state(game.state)
    state_description = describe_state_for_llm(decoded)
    llm_thinking = game.last_llm_response.get("thinking_process", "")
    llm_action = game.last_llm_response.get("action")
    raw_response = game.last_llm_response.get("raw_response")

    return {
        "steps": game.steps,
        "total_reward": game.total_reward,
        "is_over": game.terminated,
        "state": decoded,
        "state_description": state_description,
        "render": game.last_render,
        "llm_thinking": llm_thinking,
        "llm_action_code": llm_action,
        "llm_action_text": ACTION_LABELS.get(int(llm_action)) if isinstance(llm_action, int) else None,
        "llm_raw_action": llm_action,
        "llm_raw_response": raw_response,
        "grid": game.env.grid_state(decoded),
        "llm_error": game.last_llm_error,
    }


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


def _coerce_action(candidate):
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
                coerced = _coerce_action(candidate[key])
                if coerced is not None:
                    return coerced
        for key in ("direction", "target", "name", "label"):
            if key in candidate:
                coerced = _coerce_action(candidate[key])
                if coerced is not None:
                    return coerced
        return None
    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            coerced = _coerce_action(item)
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


if __name__ == "__main__":
    app.run(debug=True, port=5001)