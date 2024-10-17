import numpy as np
import joblib
from grid_environment import GridEnv

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((*env.size, len(env.actions)))  # Q-table initialization
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.num_episodes = 1000  # Number of episodes for training

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                # Choose action using epsilon-greedy policy
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(len(self.env.actions))  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit

                next_state, reward, done = self.env.step(action)

                # Update Q-value
                self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])
                
                state = next_state

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print("Training completed.")
        joblib.dump(self.q_table, 'q_learning_model.pkl')  # Save the Q-table

if __name__ == "__main__":
    env = GridEnv(size=(5, 5))  # Initialize the environment
    agent = QLearningAgent(env)  # Create agent
    agent.train()  # Train the agent
