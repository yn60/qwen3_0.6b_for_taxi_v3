"""Prompt templates for steering the local Qwen model."""


def build_prompt(state_description: str) -> str:
    """Return the instruction string injected into the Qwen conversation."""

    return f"""
You are an expert Taxi-v3 agent. Plan carefully and respond with **valid JSON** only.

# Environment Overview
{state_description}

# Discrete Actions (integers only)
0: Move South
1: Move North
2: Move East
3: Move West
4: Pick up passenger
5: Drop off passenger

# Response Contract
{{
  "thinking_process": "Short step-by-step reasoning in English",
  "action": <one integer from 0 to 5, no quotes>
}}

Do not output anything outside the JSON object.
"""