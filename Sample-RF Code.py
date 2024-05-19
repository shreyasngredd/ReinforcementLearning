import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.95, exploration_prob=1.0, exploration_decay=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

        # Modify the state space to handle possible dictionaries
        self.state_space_size = self.calculate_state_space_size(env.observation_space)
        self.action_space_size = env.action_space.n
        self.q_values = {}

    def calculate_state_space_size(self, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            return sum(self.calculate_state_space_size(subspace) for subspace in observation_space.spaces.values())
        elif isinstance(observation_space, gym.spaces.Discrete):
            return observation_space.n
        elif isinstance(observation_space, gym.spaces.Box):
            return observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.Tuple):
            return sum(self.calculate_state_space_size(subspace) for subspace in observation_space.spaces)
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

    def discretize_state(self, state):
        return str(state)

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            # Exploration: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploitation: choose the action with the highest Q-value
            if state not in self.q_values:
                # Initialize Q-values for the state if not present
                self.q_values[state] = np.zeros(self.action_space_size)
            return np.argmax(self.q_values[state])

    def update_q_values(self, state, action, reward, next_state, done):
        # Q-value update using Q-learning equation
        if state not in self.q_values:
            # Initialize Q-values for the state if not present
            self.q_values[state] = np.zeros(self.action_space_size)

        if next_state not in self.q_values:
            # Initialize Q-values for the next state if not present
            self.q_values[next_state] = np.zeros(self.action_space_size)

        best_next_action = np.argmax(self.q_values[next_state])
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_values[next_state][best_next_action]
        self.q_values[state][action] += self.learning_rate * (target - self.q_values[state][action])

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.discretize_state(self.env.reset())
            total_reward = 0

            while True:
                action = self.select_action(state)
                obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.discretize_state(obs)

                self.update_q_values(state, action, reward, next_state, terminated)

                state = next_state
                total_reward += reward

                if terminated:
                    break

            # Decay exploration probability
            self.exploration_prob *= self.exploration_decay

            # Print total reward for each episode
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent for a certain number of episodes
agent.train(num_episodes=10)

# Test the agent's performance
state = agent.discretize_state(env.reset())
total_reward = 0

while True:
    env.render()
    action = agent.select_action(state)
    obs, reward, terminated, truncated, info = env.step(action)
    next_state = agent.discretize_state(obs)

    total_reward += reward
    state = next_state

    if terminated:
        break

print(f"Test Complete. Total Reward: {total_reward}")

# Close the environment after testing
env.close()