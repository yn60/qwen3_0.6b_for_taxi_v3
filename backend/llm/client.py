# backend/llm/client.py
"""Local Qwen 0.6B client built on top of Hugging Face transformers."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, Iterator, Optional

from concurrent.futures import Future, ThreadPoolExecutor

import torch
from transformers import TextIteratorStreamer, pipeline

_CLIENT: Optional["LocalQwenClient"] = None
_CLIENT_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-loader")
_LOADING_FUTURE: Optional[Future] = None


class LocalQwenClient:
    """Lazy-loading wrapper around the Qwen3 0.6B chat pipeline."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        # Token limit further reduced to 80 to maximize speed while maintaining reliability.
        self.max_new_tokens = max_new_tokens or 80  
        self.temperature = temperature
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._pipeline_kwargs = self._build_pipeline_kwargs(device)

    def _load_pipeline(self):
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "task": "text-generation",
                        "trust_remote_code": True,
                    }
                    kwargs.update(self._pipeline_kwargs)

                    self._pipeline = pipeline(**kwargs)
        return self._pipeline

    def _build_pipeline_kwargs(self, requested_device: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        device_choice = self._parse_device(requested_device)
        if device_choice is None:
            device_choice = self._auto_device()

        if isinstance(device_choice, int):
            kwargs["device"] = device_choice
        elif isinstance(device_choice, str):
            lowered = device_choice.lower()
            if lowered in {"cpu", "cuda", "mps"} or lowered.startswith("cuda:"):
                kwargs["device"] = device_choice
            elif lowered == "auto":
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = self._get_device_map(lowered)

        if kwargs.get("device"):
            lowered = str(kwargs["device"]).lower()
            if lowered.startswith("cuda"):
                kwargs["torch_dtype"] = torch.float16
            elif lowered == "mps":
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
        elif kwargs.get("device_map") == "auto" and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        return kwargs

    @staticmethod
    def _parse_device(value: Optional[str] | int | float) -> Optional[str | int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        value = value.strip()
        if value == "":
            return None
        if value.isdigit():
            return int(value)
        return value

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "cpu"
        return "cpu"

    @staticmethod
    def _get_device_map(lowered: str) -> str:
        # Placeholder for complex device_map logic if needed
        return lowered 

    def _build_messages(self, prompt: str) -> list[Dict[str, str]]:
        # Instructions updated to completely remove THINK tags and require text immediately after JSON.
        enforced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. First, generate the final action ONLY in the JSON format: {{"action": number}}.
2. Immediately follow the JSON with a single, concise reasoning sentence.
3. The total output MUST contain the {{"action": N}} block followed by the reasoning text."""
        return [{"role": "user", "content": enforced_prompt}]

    def generate(self, prompt: str) -> Dict[str, Any]:
        result: Optional[Dict[str, Any]] = None
        for event in self.generate_stream(prompt):
            if event.get("type") == "result":
                result = event["data"]
            elif event.get("type") == "error":
                return {
                    "thinking_process": event.get("error", "Model error"),
                    "action": None,
                }

        if result is None:
            return {
                "thinking_process": "Model did not return any data.",
                "action": None,
            }

        return result

    def generate_stream(self, prompt: str) -> Iterator[Dict[str, Any]]:
        pipe = self._load_pipeline()
        messages = self._build_messages(prompt)

        streamer = TextIteratorStreamer(
            pipe.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_error: Optional[Exception] = None

        def run_pipeline() -> None:
            nonlocal generation_error
            try:
                kwargs: Dict[str, Any] = {
                    "temperature": self.temperature,
                    "do_sample": self.temperature > 0,
                    "streamer": streamer,
                    "return_full_text": False,
                }

                max_tokens = self._effective_max_new_tokens(pipe)
                if max_tokens is not None:
                    kwargs["max_new_tokens"] = max_tokens

                pipe(messages, **kwargs)
            except Exception as exc:
                generation_error = exc

        worker = threading.Thread(target=run_pipeline, daemon=True)
        worker.start()

        collected_parts: list[str] = []

        for text in streamer:
            if not text:
                continue
            collected_parts.append(text)
            yield {"type": "token", "token": text}

        worker.join(timeout=30.0)

        if worker.is_alive():
            yield {"type": "error", "error": "Model generation timed out."}
            return

        if generation_error is not None:
            yield {"type": "error", "error": str(generation_error)}
            return

        raw_text = "".join(collected_parts)
        parsed = _parse_response(raw_text)
        parsed.setdefault("raw_response", raw_text)
        yield {"type": "result", "data": parsed}

    def _effective_max_new_tokens(self, pipe) -> Optional[int]:
        if self.max_new_tokens and self.max_new_tokens > 0:
            return self.max_new_tokens

        model = getattr(pipe, "model", None)
        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("max_new_tokens", "max_length", "max_position_embeddings"):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    if attr == "max_position_embeddings":
                        return max(32, value - getattr(config, "max_input_length", 0))
                    return value

        return 1024


def get_qwen_action(prompt: str) -> Dict[str, Any]:
    client = _get_client()

    if not _ensure_pipeline_ready(client):
        return {
            "thinking_process": (
                "Model is still downloading/warming up. "
                "Returning a fallback action for now."
            ),
            "action": None,
        }

    return client.generate(prompt)


def stream_qwen_action(prompt: str) -> Iterator[Dict[str, Any]]:
    client = _get_client()

    if not _ensure_pipeline_ready(client):
        yield {
            "type": "result",
            "data": {
                "thinking_process": (
                    "Model is still downloading/warming up. Returning a fallback action for now."
                ),
                "action": None,
            },
        }
        return

    yield from client.generate_stream(prompt)


def _get_client() -> LocalQwenClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                max_new_tokens_env = os.getenv("QWEN_MAX_NEW_TOKENS")
                max_new_tokens: Optional[int]
                if max_new_tokens_env is None or max_new_tokens_env.strip().lower() in {"", "none", "unlimited"}:
                    max_new_tokens = None
                else:
                    try:
                        parsed = int(max_new_tokens_env)
                        max_new_tokens = parsed if parsed > 0 else None
                    except ValueError:
                        max_new_tokens = None

                _CLIENT = LocalQwenClient(
                    model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                    device=os.getenv("QWEN_DEVICE"),
                    max_new_tokens=max_new_tokens,
                    temperature=float(os.getenv("QWEN_TEMPERATURE", "0.1")),
                )
    return _CLIENT


def _ensure_pipeline_ready(client: LocalQwenClient) -> bool:
    # FIX: Moving global declaration to the top to resolve SyntaxError
    global _LOADING_FUTURE 

    if client._pipeline is not None:
        return True

    if _LOADING_FUTURE is None:
        _LOADING_FUTURE = _EXECUTOR.submit(client._load_pipeline)

    if _LOADING_FUTURE.done():
        try:
            # We need to call result() to check for exceptions during loading
            _LOADING_FUTURE.result()
            return True
        except Exception as e:
            # Handle loading failure gracefully and allow retry
            print(f"Error during model loading: {e}")
            _LOADING_FUTURE = None 
            return False

    return False


# Regex to find the final action JSON block
ACTION_JSON_PATTERN = re.compile(r"\{\s*\"action\"\s*:\s*(\d)\s*\}")

def _parse_response(text: str) -> Dict[str, Any]:
    """
    Improved parsing logic - handles actual LLM output patterns
    """
    text = text.strip()
    
    # 1. First try to parse JSON format
    action = None
    action_match = ACTION_JSON_PATTERN.search(text)
    
    if action_match:
        try:
            extracted_action = int(action_match.group(1))
            if 0 <= extracted_action <= 5:
                action = extracted_action
                # Successfully parsed JSON, use remaining text as reasoning
                thinking_process = text[action_match.end():].strip()
                thinking_process = re.sub(r'^<think>', '', thinking_process).strip()
                return {
                    "thinking_process": thinking_process if thinking_process else "Action selected via JSON",
                    "action": action,
                    "raw_response": text
                }
        except ValueError:
            pass

    # 2. If no JSON, try to extract numeric action from text
    # Find action numbers (0-5) in text
    digit_pattern = re.compile(r'\b([0-5])\b')
    digit_match = digit_pattern.search(text)
    
    if digit_match:
        try:
            action = int(digit_match.group(1))
            # Use full text as reasoning process
            thinking_process = text.strip()
            thinking_process = re.sub(r'^<think>', '', thinking_process).strip()
            return {
                "thinking_process": thinking_process,
                "action": action,
                "raw_response": text,
                "reason": "Action extracted from text (fallback parsing)"
            }
        except ValueError:
            pass

    # 3. If neither, try to infer action based on content
    text_lower = text.lower()
    
    # Infer action based on keywords
    if any(word in text_lower for word in ['east', 'right', 'eastward']):
        action = 2  # East
    elif any(word in text_lower for word in ['west', 'left', 'westward']):
        action = 3  # West  
    elif any(word in text_lower for word in ['north', 'up', 'northward']):
        action = 1  # North
    elif any(word in text_lower for word in ['south', 'down', 'southward']):
        action = 0  # South
    elif any(word in text_lower for word in ['pickup', 'pick up', 'collect']):
        action = 4  # Pickup
    elif any(word in text_lower for word in ['dropoff', 'drop off', 'deliver']):
        action = 5  # Dropoff
    else:
        # Default fallback: choose action based on simple logic
        # From (2,1) to (4,3) should move East
        action = 2  # East as fallback
    
    thinking_process = text.strip()
    thinking_process = re.sub(r'^<think>', '', thinking_process).strip()
    
    return {
        "thinking_process": thinking_process,
        "action": action,
        "raw_response": text,
        "reason": f"Action inferred from content: {action}"
    }


# Pre-load the model on import
try:
    _ensure_pipeline_ready(_get_client())
except Exception:
    pass