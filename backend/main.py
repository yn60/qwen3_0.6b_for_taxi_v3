import re
import threading
from typing import Dict, Optional

from pathlib import Path

import json

from flask import Flask, Response, jsonify, render_template, stream_with_context

from taxi.environment import TaxiEnvironment
from taxi.state_utils import decode_state, describe_state_for_llm, get_prompt
from taxi.action_utils import coerce_action, ACTION_LABELS
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


ACTION_LABELS = ACTION_LABELS


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
    # Back-compat wrapper so existing imports keep working
    return coerce_action(candidate)


if __name__ == "__main__":
    app.run(debug=True, port=5001)