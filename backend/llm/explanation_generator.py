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