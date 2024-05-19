import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DeepQAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95, exploration_prob=1.0, exploration_decay=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

        self.state_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n

        # Neural network model for Q-learning
        self.q_network = self.build_q_network()
        self.q_target_network = self.build_q_network()
        self.update_target_network()

    def build_q_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_space_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            # Convert the state tuple to a numpy array before predicting
            state_array = np.array(state)
            state_array = state_array.reshape(1, -1)
            q_values = self.q_network.predict(state_array)
            return np.argmax(q_values)


        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            q_values = self.q_network.predict(state.reshape(1, -1))
            return np.argmax(q_values)

    def update_q_values(self, state, action, reward, next_state, done):
        q_values = self.q_network.predict(state.reshape(1, -1))
        target = q_values.copy()

        if done:
            target[0][action] = reward
        else:
            next_q_values = self.q_target_network.predict(next_state.reshape(1, -1))
            target[0][action] = reward + self.discount_factor * np.max(next_q_values)

        self.q_network.fit(state.reshape(1, -1), target, epochs=1, verbose=0)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update_q_values(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            self.exploration_prob *= self.exploration_decay
            self.update_target_network()

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize the Deep Q-learning agent
deep_q_agent = DeepQAgent(env)

# Train the agent for a certain number of episodes
deep_q_agent.train(num_episodes=20)

# Play the game: Human vs Machine
total_reward_human = 0
total_reward_machine = 0

state = env.reset()

while True:
    # Human's turn
    env.render()
    action_human = int(input("Enter 0 for left or 1 for right: "))
    obs, reward_human, terminated, _, _ = env.step(action_human)
    state = obs
    total_reward_human += reward_human

    # Check for termination after human's turn
    if terminated:
        print(f"Human's Turn Complete. Total Reward (Human): {total_reward_human}, Total Reward (Machine): {total_reward_machine}")
        break

    # Machine's turn
    action_machine = deep_q_agent.select_action(state)
    obs, reward_machine, terminated, _, _ = env.step(action_machine)
    state = obs
    total_reward_machine += reward_machine

    # Check for termination after machine's turn
    if terminated:
        print(f"Machine's Turn Complete. Total Reward (Human): {total_reward_human}, Total Reward (Machine): {total_reward_machine}")
        break

# Close the environment after playing
env.close()
