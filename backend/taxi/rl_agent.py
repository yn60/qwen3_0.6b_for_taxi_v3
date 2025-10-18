import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def train(self, episodes):
        successful_episodes = []
        for i in range(episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_data = []
            total_reward = 0

            while not done and not truncated:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.learn(state, action, reward, next_state)
                episode_data.append((state, action, reward))
                state = next_state
                total_reward += reward

            if reward == 20:  # Successfully dropped off the passenger
                successful_episodes.append({
                    "episode": i,
                    "total_reward": total_reward,
                    "steps": episode_data
                })

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (i + 1) % 100 == 0:
                print(f"Episode {i + 1}/{episodes} completed.")

        return successful_episodes

def solve_taxi_v3_and_collect_data():
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(env)
    successful_episodes = agent.train(episodes=20000)
    env.close()
    return successful_episodes

if __name__ == "__main__":
    collected_data = solve_taxi_v3_and_collect_data()
    print(f"Collected {len(collected_data)} successful episodes.")
    # You can now process or save the collected_data
    # For example, printing the first successful episode
    if collected_data:
        print("First successful episode data:")
        print(collected_data[0])
