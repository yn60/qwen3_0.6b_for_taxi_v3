# backend/taxi/prompt_builder.py
"""Efficient prompt templates designed for Taxi-v3 environment"""

def build_prompt(state_description: str) -> str:
    """Build concise instruction string for rapid decision-making"""
    return f"""{state_description}

CRITICAL: Output ONLY JSON with 'action' key (0-5). No thinking, no explanations.

Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff

Output format: {{"action": number}}"""

def build_prompt_with_history(state_description: str, previous_actions: list) -> str:
    """Enhanced prompt with action history to prevent repetitive moves"""
    history_context = ""
    if previous_actions:
        recent_actions = previous_actions[-2:]  # Keep only last 2 actions
        action_names = ["S", "N", "E", "W", "P", "D"]  # Single letter abbreviations
        history_list = [f"{action_names[i]}" for i in recent_actions if i < len(action_names)]
        if history_list:
            history_context = f"\nRecent actions: {''.join(history_list)}"
    
    return f"""{state_description}{history_context}

CRITICAL: Output ONLY JSON with 'action' key (0-5). No thinking, no explanations.

Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff

Output format: {{"action": number}}"""

def build_context_aware_prompt(state_description: str, taxi_pos: tuple, destination: tuple) -> str:
    """Context-aware prompt with strategic hints based on current position and destination"""
    taxi_row, taxi_col = taxi_pos
    dest_row, dest_col = destination
    
    # Provide movement suggestions based on relative position
    if taxi_row < dest_row:
        vertical_hint = "move South"
    else:
        vertical_hint = "move North"
    
    if taxi_col < dest_col:
        horizontal_hint = "move East" 
    else:
        horizontal_hint = "move West"
    
    return f"""{state_description}
Strategy: {vertical_hint} and {horizontal_hint}

CRITICAL: Output ONLY JSON with 'action' key (0-5). No thinking, no explanations.

Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff

Output format: {{"action": number}}"""

# Predefined prompts for common states to enable faster responses
COMMON_STATE_PROMPTS = {
    # (taxi_row, taxi_col, passenger_loc, destination)
    (2, 1, 4, 3): '{"action":2}',  # From (2,1) to Blue (4,3) → East
    (0, 0, 0, 1): '{"action":2}',  # From Red to Green → East
    (4, 0, 2, 3): '{"action":2}',  # From Yellow to Blue → East
    (0, 4, 1, 0): '{"action":3}',  # From Green to Red → West
}

def build_optimized_prompt_by_state(decoded_state: dict) -> str:
    """Return the most suitable prompt based on decoded state"""
    key = (
        decoded_state["taxi_row"],
        decoded_state["taxi_col"], 
        decoded_state["passenger_location"],
        decoded_state["destination_index"]
    )
    
    # Use predefined prompt for common states
    if key in COMMON_STATE_PROMPTS:
        return COMMON_STATE_PROMPTS[key]
    
    # Otherwise use general prompt
    from .state_utils import describe_state_for_llm
    state_desc = describe_state_for_llm(decoded_state)
    return build_prompt(state_desc)