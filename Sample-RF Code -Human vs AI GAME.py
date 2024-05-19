import pygame
from pygame.locals import QUIT
import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.95, exploration_prob=1.0, exploration_decay=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
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
        if isinstance(state, str):
            return state
        elif isinstance(state, (list, np.ndarray)):
            return '_'.join(map(lambda x: str(float(x)), state))
        else:
            return str(state)

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            self.q_values.setdefault(state, np.zeros(self.action_space_size))
            return np.argmax(self.q_values[state])

    def update_q_values(self, state, action, reward, next_state, done):
        self.q_values.setdefault(state, np.zeros(self.action_space_size))
        self.q_values.setdefault(next_state, np.zeros(self.action_space_size))

        best_next_action = np.argmax(self.q_values[next_state])
        target = reward if done else reward + self.discount_factor * self.q_values[next_state][best_next_action]
        self.q_values[state][action] += self.learning_rate * (target - self.q_values[state][action])

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.discretize_state(self.env.reset())
            total_reward = 0

            while True:
                action = self.select_action(state)
                obs, reward, terminated, _, _ = self.env.step(action)
                next_state = self.discretize_state(obs)
                self.update_q_values(state, action, reward, next_state, terminated)

                state = next_state
                total_reward += reward

                if terminated:
                    break

            self.exploration_prob *= self.exploration_decay
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Custom game environment with Pygame visualization
class CustomGame:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.total_reward_human = 0
        self.total_reward_machine = 0
        self.state = agent.discretize_state(env.reset())

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("CartPole Game")
        self.clock = pygame.time.Clock()

    def human_turn(self, action_human):
        obs, reward_human, terminated, _, _ = self.env.step(action_human)
        self.state = self.agent.discretize_state(obs)
        self.total_reward_human += reward_human
        return terminated

    def machine_turn(self):
        action_machine = self.agent.select_action(self.state)
        obs, reward_machine, terminated, _, _ = self.env.step(action_machine)
        self.state = self.agent.discretize_state(obs)
        self.total_reward_machine += reward_machine
        return terminated
    
    def render(self):
        self.screen.fill((255, 255, 255))
    
        # Render CartPole
        pole_length = 100
        pole_thickness = 5
        cart_width = 50
        cart_height = 30
    
        try:
            cart_x = int(float(self.state[0]))  # Convert state to numeric value
            cart_x = int(cart_x * 100 + 200)
        except (ValueError, IndexError) as e:
            print(f"Error: Unable to get cart_x value from state - {e}")
            cart_x = 0

        cart_y = 200

        try:
            angle = float(self.state[2])
            pole_bottom_x = cart_x + np.sin(angle) * pole_length
            pole_bottom_y = 200 - np.cos(angle) * pole_length
        except (ValueError, IndexError) as e:
            print(f"Error: Unable to get angle value from state - {e}")
            pole_bottom_x = cart_x
            pole_bottom_y = 200

        pygame.draw.rect(
            self.screen, (0, 0, 0), (cart_x - cart_width // 2, 200 - cart_height // 2, cart_width, cart_height)
        )
        pygame.draw.line(self.screen, (0, 0, 0), (cart_x, 200), (pole_bottom_x, pole_bottom_y), pole_thickness)

        pygame.display.flip()
        self.clock.tick(60)  # Control the frame rate (e.g., 60 frames per second)

# Create the CartPole environment for training the agent
env = gym.make('CartPole-v1')

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent for a certain number of episodes
agent.train(num_episodes=100)

# Play the game with Pygame visualization
custom_game = CustomGame(agent, env)

while True:
    # Human's turn
    custom_game.render()
    while True:
        action_human_str = input("Enter 0 for left or 1 for right: ")
        if action_human_str.isdigit() and int(action_human_str) in [0, 1]:
            action_human = int(action_human_str)
            break
        else:
            print("Invalid input. Please enter 0 or 1.")

    terminated_human = custom_game.human_turn(action_human)
    print(f"Human's Turn Complete. Total Reward: {custom_game.total_reward_human}")

    if terminated_human:
        break

    # Machine's turn
    terminated_machine = custom_game.machine_turn()
    print(f"Machine's Turn Complete. Total Reward: {custom_game.total_reward_machine}")

    if terminated_machine:
        break

# Keep the Pygame window open for a while (e.g., 5 seconds)
pygame.time.delay(5000)

# Close the environment after playing
env.close()

