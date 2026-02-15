#!/usr/bin/env python3
"""
DQN Evaluation in ROS2/Gazebo simulation
Compares DQN performance against LQR baseline
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from collections import deque

# Add dqn directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import DQNAgent

# ── Earthquake parameters (match earthquake_force_generator.py) ────────────────
NUM_WAVES      = 5
FREQ_RANGE     = [0.5, 4.0]
BASE_AMPLITUDE = 15.0
NOISE_STD      = BASE_AMPLITUDE * 0.1

def generate_earthquake_force(t, freqs, phases, amps):
    force = sum(a * np.sin(2 * np.pi * f * t + p)
                for a, f, p in zip(amps, freqs, phases))
    force += np.random.normal(0, NOISE_STD)
    return float(force)

class DQNController(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        # Load trained DQN agent
        self.agent = DQNAgent(state_dim=5, action_dim=2)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'dqn_cartpole_earthquake_best.pth')
        self.agent.q_network.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        self.agent.q_network.eval()
        self.get_logger().info(f'Loaded DQN model from {model_path}')

        # Discrete action force magnitude (matches CartPole-v1)
        self.FORCE_MAG = 15.0  # N - scaled up from gym's 10N for our system

        # Earthquake profile
        self.freqs  = np.random.uniform(FREQ_RANGE[0], FREQ_RANGE[1], NUM_WAVES)
        self.phases = np.random.uniform(0, 2 * np.pi, NUM_WAVES)
        self.amps   = np.random.uniform(0.8, 1.2, NUM_WAVES) * BASE_AMPLITUDE / NUM_WAVES

        # State
        self.x = None
        self.state_initialized = False
        self.start_time = None
        self.t = 0.0

        # Data logging
        self.time_steps    = deque()
        self.cart_positions = deque()
        self.pole_angles   = deque()
        self.control_forces = deque()
        self.eq_forces     = deque()

        # Publishers/Subscribers
        self.force_pub = self.create_publisher(
            Float64,
            '/model/cart_pole/joint/cart_to_base/cmd_force',
            10)

        self.joint_sub = self.create_subscription(
            JointState,
            '/world/empty/model/cart_pole/joint_state',
            self.joint_state_callback,
            10)

        self.eq_sub = self.create_subscription(
            Float64,
            '/earthquake_force',
            self.earthquake_callback,
            10)

        self.last_eq_force = 0.0
        self.timer = self.create_timer(0.02, self.control_loop)  # 50Hz
        self.MAX_TIME = 120.0

        self.get_logger().info('DQN Controller initialized')

    def joint_state_callback(self, msg):
        try:
            cart_idx = msg.name.index('cart_to_base')
            pole_idx = msg.name.index('pole_joint')

            self.x = np.array([
                msg.position[cart_idx],
                msg.velocity[cart_idx],
                msg.position[pole_idx],
                msg.velocity[pole_idx]
            ])

            if not self.state_initialized:
                self.state_initialized = True
                self.start_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info(
                    f'Initial state: cart={self.x[0]:.3f}m, '
                    f'pole={np.degrees(self.x[2]):.3f}°')

        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Joint state error: {e}')

    def earthquake_callback(self, msg):
        self.last_eq_force = msg.data

    def control_loop(self):
        if not self.state_initialized:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

        # Build augmented state [cart_pos, cart_vel, pole_angle, pole_vel, eq_force]
        state_aug = np.append(self.x, self.last_eq_force / BASE_AMPLITUDE)

        # Select action (greedy - no exploration)
        action = self.agent.select_action(state_aug, evaluate=True)

        # Convert discrete action to force
        # Action 0 = push left, Action 1 = push right
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        # Publish force
        msg = Float64()
        msg.data = float(force)
        self.force_pub.publish(msg)

        # Log data
        self.time_steps.append(current_time)
        self.cart_positions.append(self.x[0])
        self.pole_angles.append(np.degrees(self.x[2]))
        self.control_forces.append(force)
        self.eq_forces.append(self.last_eq_force)

        self.t += 0.02

        # Termination conditions
        if (abs(self.x[0]) > 2.5 or
                abs(self.x[2]) > np.radians(45) or
                current_time >= self.MAX_TIME):

            self.get_logger().warn(
                f'Simulation ended: cart={self.x[0]:.2f}m, '
                f'pole={np.degrees(self.x[2]):.2f}°, '
                f'duration={current_time:.2f}s')

            self.print_metrics(current_time)
            self.plot_results()
            rclpy.shutdown()

    def print_metrics(self, duration):
        max_cart = max(abs(p) for p in self.cart_positions)
        max_angle = max(abs(a) for a in self.pole_angles)
        avg_force = np.mean(np.abs(self.control_forces))
        rms_cart  = np.sqrt(np.mean(np.array(self.cart_positions)**2))

        self.get_logger().info("=" * 50)
        self.get_logger().info("DQN PERFORMANCE METRICS")
        self.get_logger().info(f"Duration of stable operation: {duration:.2f}s")
        self.get_logger().info(f"Maximum cart displacement:    {max_cart:.3f}m")
        self.get_logger().info(f"RMS cart position:            {rms_cart:.3f}m")
        self.get_logger().info(f"Maximum pole angle deviation: {max_angle:.3f}°")
        self.get_logger().info(f"Average control effort:       {avg_force:.3f}N")
        self.get_logger().info("=" * 50)
        self.get_logger().info("LQR BASELINE (for comparison):")
        self.get_logger().info("Duration: 7.56s")
        self.get_logger().info("Max cart displacement: 0.242m")
        self.get_logger().info("Max pole angle: ~97° (at failure)")
        self.get_logger().info("=" * 50)

    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DQN Controller Performance vs LQR Baseline', fontsize=13)

        axes[0, 0].plot(self.time_steps, self.cart_positions,
                        color='blue', label='DQN Cart Position')
        axes[0, 0].axhline(2.5, color='red', linestyle='--', label='Limit (+2.5m)')
        axes[0, 0].axhline(-2.5, color='red', linestyle='--', label='Limit (-2.5m)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Cart Position (m)')
        axes[0, 0].set_title('Cart Position')
        axes[0, 0].legend()

        axes[0, 1].plot(self.time_steps, self.pole_angles,
                        color='red', label='DQN Pole Angle')
        axes[0, 1].axhline(45, color='orange', linestyle='--', label='Limit (45°)')
        axes[0, 1].axhline(-45, color='orange', linestyle='--')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Pole Angle (°)')
        axes[0, 1].set_title('Pole Angle')
        axes[0, 1].legend()

        axes[1, 0].plot(self.time_steps, self.eq_forces,
                        color='green', label='Earthquake Force')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Force (N)')
        axes[1, 0].set_title('Earthquake Disturbance')
        axes[1, 0].legend()

        axes[1, 1].plot(self.time_steps, self.control_forces,
                        color='magenta', label='DQN Control Force')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Force (N)')
        axes[1, 1].set_title('Control Force')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('dqn_evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Evaluation plot saved as 'dqn_evaluation_results.png'")


def main(args=None):
    rclpy.init(args=args)
    controller = DQNController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
