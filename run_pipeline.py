import json
import os
from backend.taxi.rl_agent import solve_taxi_v3_and_collect_data
from backend.llm.explanation_generator import generate_explanation_for_rl_steps

def run_data_collection_and_explanation_pipeline():
    """
    Runs the full pipeline to:
    1. Solve the Taxi-v3 environment using a Q-learning agent.
    2. Collect data from successful episodes.
    3. Generate explanations for the collected steps using an LLM.
    4. Save the final data to a JSON file.
    """
    print("Starting the reinforcement learning process to solve Taxi-v3...")
    successful_episodes = solve_taxi_v3_and_collect_data()
    print(f"RL process completed. Collected {len(successful_episodes)} successful episodes.")

    if not successful_episodes:
        print("No successful episodes were collected. Exiting pipeline.")
        return

    print("\nGenerating explanations for the collected RL data...")
    explained_data = generate_explanation_for_rl_steps(successful_episodes)
    print("Explanation generation completed.")

    # Define the output path
    output_dir = "finetuning_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filepath = os.path.join(output_dir, "explained_rl_data.json")

    # Save the explained data
    with open(output_filepath, "w") as f:
        json.dump(explained_data, f, indent=4)

    print(f"\nSuccessfully saved the explained data to {output_filepath}")
    if explained_data:
        print("\nExample of the first explained episode:")
        print(json.dumps(explained_data[0], indent=4))

if __name__ == "__main__":
    # Make sure to have your OPENAI_API_KEY set in your environment or .env file
    run_data_collection_and_explanation_pipeline()
