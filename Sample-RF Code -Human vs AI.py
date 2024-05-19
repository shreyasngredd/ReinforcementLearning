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
agent.train(num_episodes=20)

# Play the game: Human vs Machine
total_reward_human = 0
total_reward_machine = 0

state = agent.discretize_state(env.reset())

while True:
    # Human's turn
    env.render()
    action_human = int(input("Enter 0 for left or 1 for right: "))
    obs, reward_human, terminated, truncated, info = env.step(action_human)
    state = agent.discretize_state(obs)
    total_reward_human += reward_human

    # Check for termination after human's turn
    if terminated:
        print(f"Human's Turn Complete. Total Reward (Human): {total_reward_human}, Total Reward (Machine): {total_reward_machine}")
        break

    # Machine's turn
    action_machine = agent.select_action(state)
    obs, reward_machine, terminated, truncated, info = env.step(action_machine)
    state = agent.discretize_state(obs)
    total_reward_machine += reward_machine

    # Check for termination after machine's turn
    if terminated:
        print(f"Machine's Turn Complete. Total Reward (Human): {total_reward_human}, Total Reward (Machine): {total_reward_machine}")
        break