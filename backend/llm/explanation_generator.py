import openai
import json
from config.settings import get_settings

def get_openai_explanation(prompt):
    settings = get_settings()
    openai.api_key = settings.openai_api_key
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()

def generate_explanation_for_rl_steps(successful_episodes):
    explanations = []
    for episode_data in successful_episodes:
        # Create a simplified representation of the steps
        simplified_steps = []
        for state, action, reward in episode_data["steps"]:
            simplified_steps.append(f"State: {state}, Action: {action}, Reward: {reward}")
        
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
            "episode": episode_data["episode"],
            "total_reward": episode_data["total_reward"],
            "steps": episode_data["steps"],
            "explanation": explanation
        })
        
    return explanations

if __name__ == "__main__":
    # This is an example of how you might use this module.
    # You would first run the rl_agent.py to get the successful_episodes data.
    
    # Example successful_episodes data (replace with actual data from rl_agent)
    successful_episodes_example = [
        {
            "episode": 0,
            "total_reward": 13,
            "steps": [
                (328, 1, -1), (428, 1, -1), (448, 4, -1), (348, 1, -1), 
                (448, 5, 20)
            ]
        }
    ]
    
    explained_data = generate_explanation_for_rl_steps(successful_episodes_example)
    
    # Save the explained data to a file
    with open("explained_rl_data.json", "w") as f:
        json.dump(explained_data, f, indent=4)
        
    print("Generated explanations for RL data and saved to explained_rl_data.json")
    if explained_data:
        print("\nExample of explained data:")
        print(explained_data[0])
