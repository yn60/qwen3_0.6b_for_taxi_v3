import openai
import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_openai_explanation(prompt):
    """Generate explanation using new OpenAI API"""
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not openai_api_key:
        return "OpenAI API key not set, cannot generate explanation"
    
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI planning expert skilled in analyzing RL agent decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return f"Unable to generate explanation: {e}"

def generate_explanation_for_rl_steps(successful_episodes):
    explanations = []
    for episode_data in successful_episodes:
        simplified_steps = []
        # Fix: step is a dictionary, not a tuple
        for step in episode_data["steps"]:
            state_info = step["state"]      # Access dictionary
            action = step["action"]
            reward = step["reward"]
            
            simplified_steps.append(f"State: {state_info}, Action: {action}, Reward: {reward}")
        
        steps_str = "\n".join(simplified_steps)
        
        prompt = (
            "The following is a sequence of steps taken by a reinforcement learning agent to solve the Taxi-v3 problem. "
            "The agent's goal is to pick up a passenger and drop them off at the correct location. "
            "Explain why this sequence of actions is optimal for achieving the goal.\n\n"
            f"Episode Total Reward: {episode_data['total_reward']}\n"
            "Steps:\n"
            f"{steps_str}\n\n"
            "Explanation:"
        )
        
        explanation = get_openai_explanation(prompt)
        explanations.append({
            "episode": int(episode_data["episode"]),  # Convert to Python int
            "total_reward": float(episode_data["total_reward"]),  # Convert to Python float
            "steps": episode_data["steps"],
            "explanation": explanation
        })
        
    return explanations

# New function for LLM reasoning agent
def generate_teacher_data(state_description):
    """Generate thinking and action for training data using local Qwen model"""
    try:
        # Initialize model (could be moved to class initialization)
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        prompt = f"""You are playing Taxi-v3 game. Given the state, think and choose action.

State:
{state_description}

Actions:
0: South
1: North  
2: East
3: West
4: Pickup
5: Dropoff

Respond with:
<THINK>
your reasoning
</THINK>
<ACT>number</ACT>

Now respond:
<THINK>
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_response = response[len(prompt):]
        
        # Parse response
        think_match = re.search(r'<THINK>(.*?)</THINK>', full_response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else full_response[:150]
        
        act_match = re.search(r'<ACT>(\d+)</ACT>', full_response)
        if act_match:
            action = int(act_match.group(1))
            if action < 0 or action > 5:
                action = 0
        else:
            action = 0
            
        return {
            'state_description': state_description,
            'thinking': thinking,
            'action': action,
            'full_response': full_response
        }
        
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return {
            'state_description': state_description,
            'thinking': "Reasoning error",
            'action': 0,
            'full_response': ""
        }