"""Utility helpers for working with the Taxi-v3 environment state."""

from typing import Dict, Mapping

from . import prompt_builder


LOCATION_LOOKUP: Mapping[int, Dict[str, str]] = {
    0: {"name": "Red", "coords": "(0, 0)"},
    1: {"name": "Green", "coords": "(0, 4)"},
    2: {"name": "Yellow", "coords": "(4, 0)"},
    3: {"name": "Blue", "coords": "(4, 3)"},
}


def decode_state(state: int) -> Dict[str, int]:
    """Decode the integer state returned by Taxi-v3 into its components."""
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
    """Generate a natural-language description of the decoded state."""

    taxi_pos = f"({decoded_state['taxi_row']}, {decoded_state['taxi_col']})"
    destination = LOCATION_LOOKUP[decoded_state["destination_index"]]

    if decoded_state["passenger_location"] == 4:
        passenger_sentence = (
            "The passenger is already in the taxi. "
            f"They need to be dropped off at {destination['name']} located at {destination['coords']}."
        )
    else:
        pickup = LOCATION_LOOKUP[decoded_state["passenger_location"]]
        passenger_sentence = (
            f"The passenger is waiting at {pickup['name']} located at {pickup['coords']}. "
            f"The final destination is {destination['name']} at {destination['coords']}."
        )

    return f"The taxi is currently at position {taxi_pos}. {passenger_sentence}"


def get_prompt(state_description: str) -> str:
    """Build the instruction prompt presented to Qwen."""

    return prompt_builder.build_prompt(state_description)


__all__ = ["decode_state", "describe_state_for_llm", "get_prompt"]