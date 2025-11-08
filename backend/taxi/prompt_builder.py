# backend/taxi/prompt_builder.py
"""Clean prompt templates for Taxi-v3 environment - NO CHEATING VERSION"""

SYSTEM_PROMPT = (
    "You are an AI agent controlling a taxi in a 5x5 grid environment. "
    "Your primary goal is to complete the passenger trip in the minimum number of steps. "
    "CRITICAL: You MUST output the JSON action FIRST, immediately followed by "
    "a SINGLE, CONCISE sentence for your reasoning."
)

ACTION_MAPPINGS = "Actions: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff"

def build_prompt(state_description: str) -> str:
    """Basic prompt with STRONGER enforcement"""
    return f"""{SYSTEM_PROMPT}

{ACTION_MAPPINGS}

Current State: {state_description}

CRITICAL OUTPUT FORMAT - MUST FOLLOW EXACTLY:
1. First line: {{"action": number}} 
2. Second line: [Single reasoning sentence]

EXAMPLE:
{{"action": 2}}
[I need to move East to reach the destination at (4,3)]

DO NOT:
- Use <think> tags
- Put JSON in the middle
- Write long explanations
- Output anything else"""

def build_strict_prompt(state_description: str) -> str:
    """更严格的prompt版本"""
    return f"""{SYSTEM_PROMPT}

{ACTION_MAPPINGS}

Current State: {state_description}

STRICT OUTPUT FORMAT - ZERO TOLERANCE:
{{"action": N}}
[Your one-sentence reasoning here]

FAILURE TO FOLLOW FORMAT WILL RESULT IN SYSTEM ERROR.

VALID EXAMPLE:
{{"action": 2}}
[Moving East toward Blue destination at (4,3)]

INVALID (WILL FAIL):
<think>I should move east because...</think>
{{"action": 2}}
Or any other format variation.

REMEMBER: Only TWO lines total!"""

def build_prompt_with_history(state_description: str, previous_actions: list) -> str:
    """Prompt with action history to avoid repetition"""
    history_context = ""
    if previous_actions:
        recent_actions = previous_actions[-2:]
        action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
        history_list = [f"Action {i} ({action_names[i]})" for i in recent_actions if 0 <= i <= 5]
        if history_list:
            history_context = f"| History (Recent Actions): {', '.join(history_list)}"
    
    return f"""{SYSTEM_PROMPT}

{ACTION_MAPPINGS}

Current State: {state_description} {history_context}

CRITICAL OUTPUT FORMAT - MUST FOLLOW EXACTLY:
1. First line: {{"action": number}} 
2. Second line: [Single reasoning sentence]

EXAMPLE:
{{"action": 2}}
[I need to move East to reach the destination at (4,3)]

DO NOT:
- Use <think> tags
- Put JSON in the middle
- Write long explanations
- Output anything else"""

def build_optimized_prompt_by_state(decoded_state: dict) -> str:
    """Universal prompt selector - now truly clean"""
    from .state_utils import describe_state_for_llm
    state_desc = describe_state_for_llm(decoded_state)
    return build_prompt(state_desc)