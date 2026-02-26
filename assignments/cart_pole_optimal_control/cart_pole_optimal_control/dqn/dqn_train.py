from dqn_agent import DQNAgent
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# Initialize environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0] + 1  # 5 states (4 original + earthquake force)
action_dim = env.action_space.n

# Scale action force to match earthquake magnitude
# Earthquakes peak at ~75N, DQN needs comparable authority
env.unwrapped.force_mag = 40.0  # Fixed at start, never modified during training

# Initialize DQN Agent
agent = DQNAgent(state_dim, action_dim)

# Earthquake Force Parameters (matches earthquake_force_generator.py)
num_waves = 5
freq_range = [0.5, 4.0]
base_amplitude = 15.0
env_timestep = 0.02

def new_earthquake_profile():
    """Generate a new random earthquake profile for each episode."""
    freqs = np.random.uniform(freq_range[0], freq_range[1], num_waves)
    phases = np.random.uniform(0, 2 * np.pi, num_waves)
    return freqs, phases

def generate_earthquake_force(time, freqs, phases):
    """Generate earthquake-like force using superposition of sine waves."""
    force = 0.0
    for freq, phase in zip(freqs, phases):
        amplitude = base_amplitude * np.random.uniform(0.8, 1.2)
        force += amplitude * np.sin(2 * np.pi * freq * time + phase)
    force += np.random.normal(0, base_amplitude * 0.1)
    return float(force)

def apply_earthquake_to_state(state, eq_force, dt=0.02):
    """
    Apply earthquake as an external disturbance to the state.
    Modifies cart velocity and position based on force impulse.
    F = ma -> a = F/m, m=1.0 for CartPole
    """
    state = state.copy()
    state[1] += eq_force * dt          # cart velocity += a*dt
    state[0] += 0.5 * eq_force * dt**2  # cart position += 0.5*a*dt^2
    return np.clip(state, -10, 10)

def compute_reward(state, done):
    """
    Improved reward function with smooth gradients.
    Cosine pole reward + Gaussian cart reward + velocity penalty.
    """
    if done:
        return -10.0  # Heavy termination penalty

    cart_pos   = state[0]
    cart_vel   = state[1]
    pole_angle = state[2]
    pole_vel   = state[3]

    r_alive = 1.0
    r_pole  = 2.0 * np.cos(pole_angle)                        # max 2.0 at theta=0
    r_cart  = np.exp(-0.5 * (cart_pos / 2.4) ** 2)            # Gaussian, max 1.0 at center
    r_vel   = -0.01 * abs(cart_vel) - 0.005 * abs(pole_vel)   # small velocity penalty

    return r_alive + r_pole + r_cart + r_vel

# Training Loop
num_episodes = 1000
best_avg_reward = -np.inf
solve_threshold = 450.0

total_rewards = []
steps_per_episode = []
epsilon_values = []

print("=" * 60)
print("DQN Training - Improved Version")
print(f"Action force magnitude: {env.unwrapped.force_mag}N")
print(f"Earthquake amplitude: {base_amplitude}N base (peaks ~{base_amplitude*num_waves:.0f}N)")
print("=" * 60)

for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    freqs, phases = new_earthquake_profile()  # New earthquake per episode
    total_reward = 0
    steps = 0
    episode_time = 0.0

    for t in range(1000):
        # Generate earthquake force
        eq_force = generate_earthquake_force(episode_time, freqs, phases)
        episode_time += env_timestep

        # Apply earthquake as state perturbation (NOT modifying force_mag)
        state_disturbed = apply_earthquake_to_state(state, eq_force)

        # Normalize earthquake force before appending to state
        state_with_force = np.append(state_disturbed, eq_force / (base_amplitude * num_waves))

        # Select and apply action
        action = agent.select_action(state_with_force, evaluate=False)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Apply earthquake to next state too
        next_state_disturbed = apply_earthquake_to_state(next_state, eq_force)
        next_state_with_force = np.append(next_state_disturbed, eq_force / (base_amplitude * num_waves))

        # Compute improved reward
        reward = compute_reward(next_state_disturbed, done)

        agent.store_transition(state_with_force, action, reward, next_state_with_force, done)
        agent.train()

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    agent.update_target_model()
    total_rewards.append(total_reward)
    steps_per_episode.append(steps)
    epsilon_values.append(agent.epsilon)

    # Logging every 10 episodes
    if episode % 10 == 0:
        avg_reward = np.mean(total_rewards[-100:])
        print(f"Episode {episode:4d} | Avg Reward (100ep): {avg_reward:8.2f} | "
              f"Steps: {steps:4d} | ε: {agent.epsilon:.4f}")

        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model('dqn_cartpole_earthquake_best.pth')
            print(f"  → New best model saved! (avg={avg_reward:.2f})")

        # Check if solved
        if avg_reward >= solve_threshold and episode >= 100:
            print(f"\nSolved at episode {episode}!")
            break

agent.save_model('dqn_cartpole_earthquake_final.pth')
env.close()

print("\n" + "=" * 60)
print(f"Training complete!")
print(f"Best average reward: {best_avg_reward:.2f}")
print(f"Final epsilon: {agent.epsilon:.4f}")
print("=" * 60)

# Plot Training Performance
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('DQN Training Performance', fontsize=14)

axes[0].plot(total_rewards, alpha=0.4, color='blue', label='Episode Reward')
if len(total_rewards) >= 100:
    rolling = np.convolve(total_rewards, np.ones(100)/100, mode='valid')
    axes[0].plot(range(99, len(total_rewards)), rolling, color='blue', linewidth=2, label='100-ep avg')
axes[0].set_ylabel('Total Reward')
axes[0].legend()

axes[1].plot(steps_per_episode, alpha=0.4, color='green', label='Steps')
if len(steps_per_episode) >= 100:
    rolling = np.convolve(steps_per_episode, np.ones(100)/100, mode='valid')
    axes[1].plot(range(99, len(steps_per_episode)), rolling, color='green', linewidth=2, label='100-ep avg')
axes[1].set_ylabel('Steps per Episode')
axes[1].legend()

axes[2].plot(epsilon_values, color='red', label='Epsilon')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Exploration Rate (ε)')
axes[2].legend()

plt.tight_layout()
plt.savefig('dqn_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as 'dqn_training_results.png'")
