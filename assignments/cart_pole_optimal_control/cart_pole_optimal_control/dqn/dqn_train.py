#!/usr/bin/env python3
"""
DQN Training for Cart-Pole with Earthquake Disturbances
Trains using gymnasium CartPole-v1 for speed, then evaluates in ROS2/Gazebo
"""

from dqn_agent import DQNAgent
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

# ── Earthquake parameters (match ROS2 earthquake_force_generator.py) ──────────
NUM_WAVES       = 5
FREQ_RANGE      = [0.5, 4.0]   # Hz
BASE_AMPLITUDE  = 15.0          # N
ENV_TIMESTEP    = 0.02          # s (CartPole default)
NOISE_STD       = BASE_AMPLITUDE * 0.1

def make_earthquake():
    """Create a new random earthquake profile."""
    freqs  = np.random.uniform(FREQ_RANGE[0], FREQ_RANGE[1], NUM_WAVES)
    phases = np.random.uniform(0, 2 * np.pi, NUM_WAVES)
    amps   = np.random.uniform(0.8, 1.2, NUM_WAVES) * BASE_AMPLITUDE / NUM_WAVES
    return freqs, phases, amps

def generate_earthquake_force(t, freqs, phases, amps):
    """Superposition of sine waves + Gaussian noise."""
    force = sum(a * np.sin(2 * np.pi * f * t + p)
                for a, f, p in zip(amps, freqs, phases))
    force += np.random.normal(0, NOISE_STD)
    return float(force)

# ── Reward function ────────────────────────────────────────────────────────────
def compute_reward(state, done, cart_limit=2.4):
    """
    Reward shaping that matches the assignment objectives:
    - Stay alive
    - Keep pole upright
    - Keep cart near center
    - Penalize termination
    """
    cart_pos, cart_vel, pole_angle, pole_vel = state

    if done:
        return -10.0  # Heavy penalty for falling

    # Survival reward
    r_alive = 1.0

    # Pole upright reward (cosine shape - smooth near 0)
    r_pole = np.cos(pole_angle) * 2.0

    # Cart centering reward (gaussian shape)
    r_cart = np.exp(-0.5 * (cart_pos / cart_limit) ** 2)

    # Velocity penalties (discourage fast motion)
    r_vel = -0.01 * abs(cart_vel) - 0.005 * abs(pole_vel)

    return r_alive + r_pole + r_cart + r_vel

# ── Training setup ─────────────────────────────────────────────────────────────
env = gym.make("CartPole-v1")

# State: [cart_pos, cart_vel, pole_angle, pole_vel, earthquake_force] = 5 dims
state_dim  = env.observation_space.shape[0] + 1
action_dim = env.action_space.n  # 2 discrete actions: push left or right

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    lr=1e-3,
    epsilon=1.0,
    min_epsilon=0.05,
    decay=0.9995,
)

# ── Training loop ──────────────────────────────────────────────────────────────
NUM_EPISODES        = 2000
TARGET_UPDATE_FREQ  = 10    # Update target network every N episodes
PRINT_FREQ          = 50    # Print stats every N episodes
SOLVE_THRESHOLD     = 450   # Consider solved if avg reward > this

total_rewards    = []
steps_per_ep     = []
epsilon_values   = []
best_avg_reward  = -np.inf
recent_rewards   = deque(maxlen=100)

print("=" * 60)
print("DQN Training - Cart-Pole with Earthquake Disturbances")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"Earthquake amplitude: {BASE_AMPLITUDE}N, "
      f"Frequency: {FREQ_RANGE[0]}-{FREQ_RANGE[1]}Hz")
print("=" * 60)

for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    
    # New earthquake profile each episode
    freqs, phases, amps = make_earthquake()
    
    total_reward = 0
    steps = 0
    t = 0.0

    for _ in range(500):  # Max steps per episode
        # Generate earthquake force
        eq_force = generate_earthquake_force(t, freqs, phases, amps)

        # Augment state with earthquake force
        state_aug = np.append(state, eq_force / BASE_AMPLITUDE)  # normalize

        # Select action
        action = agent.select_action(state_aug)

        # Step environment
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Apply earthquake effect to next state cart position
        # (approximates external force effect)
        next_state_mod = next_state.copy()
        next_state_mod[0] += eq_force * ENV_TIMESTEP * 0.1  # scaled effect
        next_state_mod[1] += eq_force * ENV_TIMESTEP        # velocity effect

        # Clip to reasonable bounds
        next_state_mod[0] = np.clip(next_state_mod[0], -2.4, 2.4)
        next_state_mod[1] = np.clip(next_state_mod[1], -4.0, 4.0)

        # Check if modified state is terminal
        if abs(next_state_mod[0]) >= 2.4 or abs(next_state_mod[2]) >= 0.209:
            done = True

        # Compute shaped reward
        reward = compute_reward(next_state_mod, done)

        # Next augmented state
        next_eq_force = generate_earthquake_force(t + ENV_TIMESTEP, freqs, phases, amps)
        next_state_aug = np.append(next_state_mod, next_eq_force / BASE_AMPLITUDE)

        # Store and train
        agent.store_transition(state_aug, action, reward, next_state_aug, done)
        agent.train()

        state = next_state_mod
        total_reward += reward
        steps += 1
        t += ENV_TIMESTEP

        if done:
            break

    # Update target network periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    total_rewards.append(total_reward)
    steps_per_ep.append(steps)
    epsilon_values.append(agent.epsilon)
    recent_rewards.append(total_reward)

    # Print progress
    if episode % PRINT_FREQ == 0:
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {episode:4d} | "
              f"Avg Reward (100ep): {avg_reward:7.2f} | "
              f"Steps: {steps:3d} | "
              f"ε: {agent.epsilon:.4f}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model("dqn_cartpole_earthquake_best.pth")
            print(f"  → New best model saved! (avg={avg_reward:.2f})")

        if avg_reward >= SOLVE_THRESHOLD:
            print(f"\nSolved at episode {episode}!")
            break

# Save final model
agent.save_model("dqn_cartpole_earthquake_final.pth")
env.close()

print("\n" + "=" * 60)
print(f"Training complete!")
print(f"Best average reward: {best_avg_reward:.2f}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print("=" * 60)

# ── Plot training results ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('DQN Training Progress - Cart-Pole with Earthquake Disturbances', 
             fontsize=13)

# Smooth rewards for plotting
def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

axes[0].plot(total_rewards, alpha=0.3, color='blue', label='Raw reward')
if len(total_rewards) >= 50:
    axes[0].plot(range(49, len(total_rewards)), 
                 smooth(total_rewards), 
                 color='blue', linewidth=2, label='Smoothed (50ep)')
axes[0].axhline(SOLVE_THRESHOLD, color='green', linestyle='--', 
                label=f'Solve threshold ({SOLVE_THRESHOLD})')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Training Rewards')
axes[0].legend()

axes[1].plot(steps_per_ep, alpha=0.3, color='orange', label='Steps')
if len(steps_per_ep) >= 50:
    axes[1].plot(range(49, len(steps_per_ep)),
                 smooth(steps_per_ep),
                 color='orange', linewidth=2, label='Smoothed')
axes[1].set_ylabel('Steps per Episode')
axes[1].set_title('Episode Length')
axes[1].legend()

axes[2].plot(epsilon_values, color='red', label='Epsilon (ε)')
axes[2].set_ylabel('Exploration Rate')
axes[2].set_xlabel('Episode')
axes[2].set_title('Exploration vs Exploitation')
axes[2].legend()

plt.tight_layout()
plt.savefig('dqn_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Training plot saved as 'dqn_training_results.png'")
