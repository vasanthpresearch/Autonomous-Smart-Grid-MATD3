import torch
import numpy as np
import os
from src.opendss_env import OpenDSSEnv
from src.matd3_agent import Actor, Critic  # Assuming these are in your MATD3 class
from src.utils import ReplayBuffer, GridLogger

# --- Hyperparameters from your WJAETS Paper ---

# ACTION_DIM and other hyperparameters remain fixed
ACTION_DIM = 4
MAX_ACTION = 1.0
BATCH_SIZE = 100
MAX_EPISODES = 1000
EXPLORE_STEPS = 5000  # Initial random actions to fill buffer
GAMMA = 0.99          # Discount factor for long-term stability
TAU = 0.005           # Soft update rate for target networks
POLICY_NOISE = 0.2    # Noise for target policy smoothing
NOISE_CLIP = 0.5
POLICY_FREQ = 2       # Delayed policy updates

def train():

    # 1. Initialize Environment and Buffer
    env = OpenDSSEnv("dss_files/master.dss")
    STATE_DIM = env.state_dim
    replay_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)
    logger = GridLogger(filename="data/simulation_results/training_log.csv")

    # 2. Initialize MATD3 Networks (Simplification of the full class)
    # In a real scenario, you'd wrap these in a 'MATD3' class
    actor = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION)
    actor_target = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic = Critic(STATE_DIM, ACTION_DIM)
    critic_target = Critic(STATE_DIM, ACTION_DIM)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    total_steps = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        for t in range(24):  # 24-hour simulation window
            total_steps += 1

            # A. Select Action (Exploration vs Exploitation)
            if total_steps < EXPLORE_STEPS:
                action = np.random.uniform(-MAX_ACTION, MAX_ACTION, size=ACTION_DIM)
            else:
                action = (
                    actor(torch.FloatTensor(state.reshape(1, -1))).cpu().data.numpy().flatten()
                    + np.random.normal(0, 0.1, size=ACTION_DIM) # Exploration noise
                ).clip(-MAX_ACTION, MAX_ACTION)

            # B. Execute step in OpenDSS
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            
            state = next_state
            episode_reward += reward

            # C. Training Step (Off-Policy)
            if total_steps > EXPLORE_STEPS:
                # Sample a batch from memory
                s, a, ns, r, d = replay_buffer.sample(BATCH_SIZE)
                
                # --- Twin Critic Update (Clipped Double Q-Learning) ---
                with torch.no_grad():
                    # Select next action with noise (Target Policy Smoothing)
                    noise = (torch.randn_like(torch.FloatTensor(a)) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                    next_action = (actor_target(torch.FloatTensor(ns)) + noise).clamp(-MAX_ACTION, MAX_ACTION)

                    # Compute target Q-value
                    target_Q1, target_Q2 = critic_target(torch.FloatTensor(ns), next_action)
                    target_Q = torch.FloatTensor(r) + (1 - torch.FloatTensor(d)) * GAMMA * torch.min(target_Q1, target_Q2)

                # Update Critics
                current_Q1, current_Q2 = critic(torch.FloatTensor(s), torch.FloatTensor(a))
                critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + torch.nn.functional.mse_loss(current_Q2, target_Q)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # --- Delayed Policy Update ---
                if total_steps % POLICY_FREQ == 0:
                    # Use the first output of critic as Q1
                    actor_loss = -critic(torch.FloatTensor(s), actor(torch.FloatTensor(s)))[0].mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft update target networks
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        if episode % 10 == 0:
            print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Total Steps: {total_steps}")
            # Ensure models directory exists before saving
            os.makedirs("models", exist_ok=True)
            torch.save(actor.state_dict(), f"models/trained_matd3_actor.pth")

    logger.save()

if __name__ == "__main__":
    train()