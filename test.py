from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import ale_py

episodes = 10000
max_steps = 10000
batch_size = 64
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995

hidden_layer = 128
learning_rate = 0.0001
step_repeat = 4
gamma = 0.99

env = gym.make("ALE/Pong-v5", render_mode="human")

# Simplify data for agent
env = ResizeObservation(env, (64,64))
env = GrayscaleObservation(env, keep_dim=True)

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, 
              step_repeat=step_repeat,
              gamma=gamma)

agent.test()
