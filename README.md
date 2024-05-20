# Experiments with Reinforcement Learning 

Ideally, Machine Learning models are of 3 categories: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. While Unsupervised and Supervised are frequently 
discussed, Reinforcement Learning is rarely discussed. Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize some notion of cumulative 
reward. Much like Supervised Learning, Reinforcement Learning requires labeled data for training. 

The project aimed to explore Q-learning and compare the decision-making processes of human participants and AI. We utilized Q-learning and Deep Q-Network (DQN) techniques 
to achieve this understanding. Q-learning is a model-free reinforcement learning algorithm that enables an agent to evaluate the value of taking specific actions in given states 
within an environment. Its goal is to discover an optimal action-selection policy that maximizes the agent's cumulative reward over time, with the "Q" representing the quality 
of an action in a particular state. DQN extends Q-learning by integrating it with deep neural networks, allowing for more complex and scalable learning.

![Reinforcement-Learning-Agent-and-Environment](https://github.com/shreyasngredd/ReinforcementLearning/assets/15787360/0fab33cc-ebc2-4c27-9b4f-1f4313159914)

This code sets up and trains a Deep Q-Network (DQN) agent to play the CartPole-v1 game. The goal is to balance a pole on a cart by moving the cart left or right. The neural network approximates the Q-values, representing 
the expected future rewards for taking different actions in a given state. Experience replay stabilizes training, breaking the correlation between consecutive 
experiences. This technique helps the agent learn from diverse experiences, improving sample efficiency and overall performance. The epsilon-greedy strategy
is also implemented to balance exploration and exploitation during training. This means the agent occasionally takes random actions (exploration) to discover new strategies while mostly relying on its learned policy (exploitation) to maximize rewards. The environment is designed to be straightforward, with a clear goal for the human player and the AI agent: to prevent the pole from falling by making appropriate cart movements. The code provides a simple framework for comparing the actions and performance of a human player with those of an AI agent trained using Q-learning. By interacting with the environment, humans and AI aim to reach the terminal state while maximizing their 
rewards.
