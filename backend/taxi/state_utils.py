# backend/taxi/state_utils.py
"""Utility functions for handling Taxi-v3 environment states"""

from typing import Dict, Mapping, Tuple
from . import prompt_builder

# Location information mapping
LOCATION_LOOKUP: Mapping[int, Dict[str, str]] = {
    0: {"name": "Red", "coords": "(0,0)", "short": "R"},
    1: {"name": "Green", "coords": "(0,4)", "short": "G"},
    2: {"name": "Yellow", "coords": "(4,0)", "short": "Y"},
    3: {"name": "Blue", "coords": "(4,3)", "short": "B"},
}

def decode_state(state: int) -> Dict[str, int]:
    """Decode integer state from Taxi-v3 into component parts"""
    dest_idx = state % 4
    state //= 4
    passenger_location = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state

    return {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_location": passenger_location,
        "destination_index": dest_idx,
    }

def describe_state_for_llm(decoded_state: Dict[str, int]) -> str:
    """Generate concise natural language state description"""
    taxi_pos = f"({decoded_state['taxi_row']},{decoded_state['taxi_col']})"
    destination = LOCATION_LOOKUP[decoded_state["destination_index"]]

    if decoded_state["passenger_location"] == 4:
        return f"Taxi{taxi_pos},P in taxi→{destination['short']}{destination['coords']}"
    else:
        pickup = LOCATION_LOOKUP[decoded_state["passenger_location"]]
        return f"Taxi{taxi_pos},P at {pickup['short']}{pickup['coords']}→{destination['short']}{destination['coords']}"

def get_prompt(state_description: str) -> str:
    """Build instruction prompt for Qwen model"""
    return prompt_builder.build_prompt(state_description)

def get_optimized_prompt(decoded_state: Dict[str, int]) -> str:
    """Get most suitable prompt based on decoded state"""
    return prompt_builder.build_optimized_prompt_by_state(decoded_state)

def get_prompt_with_history(state_description: str, previous_actions: list) -> str:
    """Build prompt with action history context"""
    return prompt_builder.build_prompt_with_history(state_description, previous_actions)

def get_context_aware_prompt(decoded_state: Dict[str, int]) -> str:
    """Get context-aware prompt with strategic hints"""
    from . import prompt_builder
    state_desc = describe_state_for_llm(decoded_state)
    taxi_pos = (decoded_state["taxi_row"], decoded_state["taxi_col"])
    dest_pos = get_destination_coords(decoded_state["destination_index"])
    return prompt_builder.build_context_aware_prompt(state_desc, taxi_pos, dest_pos)

def get_destination_coords(dest_index: int) -> Tuple[int, int]:
    """Get coordinates from location index"""
    loc = LOCATION_LOOKUP[dest_index]
    coords = loc["coords"].strip("()").split(",")
    return int(coords[0]), int(coords[1])

__all__ = [
    "decode_state",
    "describe_state_for_llm",
    "get_prompt",
    "get_optimized_prompt",
    "get_prompt_with_history",
    "get_context_aware_prompt"
]