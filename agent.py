from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2

class Agent():
    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        self.env = env
        self.step_repeat = step_repeat
        self.gamma = gamma
        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
        print(f"Loaded model on device {self.device}")

        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)
        self.model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        
        self.target_model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
    
    def process_observation(self, obs):
        # Convert the observation to a tensor and move it to the appropriate device
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) 
        return obs
    
    def test(self):
        self.model.load_the_model()

        obs, info = self.env.reset()

        # Initialize variables for training
        total_steps = 0

        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        done = False

        episode_reward = 0
        
        while not done:

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                action = torch.argmax(q_values, dim=-1).item()

            reward = 0

            for i in range(self.step_repeat):
                reward_temp = 0

                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                reward += reward_temp

                if done:
                    break
            
            obs = self.process_observation(next_obs)
            episode_reward += reward


    
    def train(self, episodes, max_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, epsilon_min):
        
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(log_dir=summary_writer_name)

        # Create a directory for TensorBoard logs
        if not os.path.exists('models'):
            os.makedirs('models')

        # Initialize variables for training
        total_steps = 0

        for episode in range(episodes):
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            done = False
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()

            while not done and episode_steps < max_steps:

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()

                reward = 0

                for _ in range(self.step_repeat):
                    reward_temp = 0

                    next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                    reward += reward_temp

                    if done:
                        break

                next_obs = self.process_observation(next_obs)

                # Store the transition in memory
                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1

                if self.memory.can_sample(batch_size):
                    # Sample a batch of transitions from memory
                    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.memory.sample_buffer(batch_size)

                    done_batch = done_batch.unsqueeze(1).float()

                    # Compute Q-values and target Q-values
                    q_values = self.model(obs_batch)
                    action_batch = action_batch.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, action_batch)

                    next_action_batch = torch.argmax(self.model(next_obs_batch), dim=1, keepdim=True)

                    next_q_values = self.target_model(next_obs_batch).gather(1, next_action_batch)

                    target_batch = reward_batch.unsqueeze(1) + (1 - done_batch) * self.gamma * next_q_values

                    # Compute loss and update model
                    loss = F.mse_loss(qsa_batch, target_batch.detach())
                    
                    writer.add_scalar('Loss/model', loss.item(), total_steps)

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)
            
            self.model.save_the_model()
            
            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            episode_end_time = time.time() - episode_start_time
            print(f"Completed episode {episode} - Reward: {episode_reward} - Steps: {episode_steps} - Time: {episode_end_time:.2f}s")
