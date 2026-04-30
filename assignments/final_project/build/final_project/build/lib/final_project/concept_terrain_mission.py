#!/usr/bin/env python3
"""
concept_terrain_mission_v2.py - Simplified version with better ROS2 integration
"""

import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import pathlib
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleCommand, TrajectorySetpoint
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=384, dict_size=512, topk=32):
        super().__init__()
        self.topk = topk
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        h = F.relu(self.encoder(x))
        if self.topk < h.shape[-1]:
            topk_vals, _ = torch.topk(h, self.topk, dim=-1)
            threshold = topk_vals[:, -1:]
            h = h * (h >= threshold).float()
        return h

    def forward(self, x):
        h = self.encode(x)
        return self.decoder(h), h


class ConceptTerrainMission(Node):

    GRID_X_MIN, GRID_X_MAX = -20.0, 20.0
    GRID_Y_MIN, GRID_Y_MAX = -20.0, 20.0
    GRID_STEP = 10.0
    ALTITUDE = -15.0

    def __init__(self):
        super().__init__('concept_terrain_mission')

        self.get_logger().info('=== ConceptTerrainMission v2 Initializing ===')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # Subscribers
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, qos)
        
        # Image subscription with explicit QoS
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/drone/front_rgb', self._image_cb, img_qos)

        self.get_logger().info('Subscriptions created')

        # State
        self.position = [0.0, 0.0, 0.0]
        self.bridge = CvBridge()
        self.latest_rgb = None
        self._img_lock = threading.Lock()

        self.offboard_counter = 0
        self.phase = 'PRE_ARM'
        self.pre_arm_counter = 0
        self.engage_counter = 0
        self.PRE_ARM_CYCLES = 50
        self.phase1_waypoints = self._gen_waypoints()
        self.phase1_idx = 0
        self.phase1_embeddings = []
        self.phase1_positions = []

        # SAE + Phase 2
        self.sae = None
        self.selective_idx = None
        self.ucb_grid = None
        self.phase2_idx = None
        self.landmark_map = []

        #logging
        self._log_path = pathlib.Path('/tmp/mission_log.json')
        self._mission_log = {
            'phase1_waypoints': [],
            'phase1_embeddings_shape': None,
            'sae_concepts': 0,
            'concept_activations': [],
            'phase2_visits': [],
            'landmarks': [],
        }

        # DINOv2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Device: {self.device}')
        self.dinov2 = None
        self._load_dinov2()

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Viz
        self._init_viz()

        # Timer
        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info('=== ConceptTerrainMission Ready ===')

    def _gen_waypoints(self):
        wps = []
        cols = list(np.arange(self.GRID_X_MIN, self.GRID_X_MAX + self.GRID_STEP, self.GRID_STEP))
        for i, x in enumerate(cols):
            ys = list(np.arange(self.GRID_Y_MIN, self.GRID_Y_MAX + self.GRID_STEP, self.GRID_STEP))
            if i % 2 == 1:
                ys = ys[::-1]
            for y in ys:
                wps.append((float(x), float(y), float(self.ALTITUDE)))
        return wps

    def _load_dinov2(self):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.get_logger().info('Loading DINOv2...')
        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(self.device).eval()
            self.dinov2.eval()
            self.dinov2 = self.dinov2.to(torch.device('cpu'))
            self.get_logger().info('DINOv2 loaded ✓')
        except Exception as e:
            self.get_logger().error(f'DINOv2 load failed: {e}')

    def _embed(self, bgr_image):
        if self.dinov2 is None:
            return np.zeros(384, dtype=np.float32)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        t = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.dinov2(t).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)

    def _odom_cb(self, msg):
        self.position = list(msg.position[:3])

    def _image_cb(self, msg):
        with self._img_lock:
            try:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.get_logger().debug(f'Image received: {self.latest_rgb.shape}')
            except Exception as e:
                self.get_logger().warn(f'Image conversion failed: {e}')

    def _init_viz(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax in self.axes:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.pause(0.01)
        self.get_logger().info('Visualization initialized')

    def _viz_update(self, title_suffix=''):
        for ax in self.axes:
            ax.cla()
        self.axes[0].set_title(f'Phase: {self.phase}')
        self.axes[1].set_title(f'Landmarks: {len(self.landmark_map)}')
        self.axes[2].set_title(f'Patches: {len(self.phase1_embeddings)}{title_suffix}')
        plt.pause(0.001)

    def _timer_cb(self):
        self._pub_offboard()
        self.offboard_counter +=1

        if self.phase == 'PRE_ARM':
            self._pub_traj(0.0, 0.0, 0.0)
            self.pre_arm_counter += 1
            if self.pre_arm_counter >= self.PRE_ARM_CYCLES:
                self.get_logger().info('PRE_ARM -> engaging offboard + arm')
                self._engage_offboard()
                self.phase = 'ENGAGING_OFFBOARD'
                self.engage_counter = 0
        elif self.phase == 'ENGAGING_OFFBOARD':
            self._pub_traj(0.0, 0.0, 0.0)
            self._engage_offboard()
            self.engage_counter += 1
            if self.engage_counter >= 15:
                self.get_logger().info('ofboard confirmed -> arming')
                self._arm()
                self.phase = 'ARM_TAKEOFF'
    

        # === STATE MACHINE ===

        elif self.phase == 'ARM_TAKEOFF':
            target = (0.0, 0.0, self.ALTITUDE)
            self._pub_traj(*target)
            if self._dist(*target) < 1.5:
                self.get_logger().info('Takeoff complete → PHASE1_FLY')
                self.phase = 'PHASE1_FLY'
                self.phase1_idx = 0

        elif self.phase == 'PHASE1_FLY':
            if self.phase1_idx >= len(self.phase1_waypoints):
                self.get_logger().info(f'Phase 1 done: {len(self.phase1_embeddings)} patches. Training SAE...')
                self.phase = 'TRAIN_SAE'
                threading.Thread(target=self._train_sae, daemon=True).start()
                return

            wp = self.phase1_waypoints[self.phase1_idx]
            self._pub_traj(*wp)

            if self._dist(*wp) < 1.5:
                with self._img_lock:
                    frame = self.latest_rgb.copy() if self.latest_rgb is not None else None

                if frame is not None:
                    emb = self._embed(frame)
                    self.phase1_embeddings.append(emb)
                    self.phase1_positions.append(wp[:2])
                    self.get_logger().info(f'Phase1[{len(self.phase1_embeddings)}] at ({wp[0]:.0f}, {wp[1]:.0f})')
                self._mission_log['phase1_waypoints'].append({
                    'idx': self.phase1_idx,
                    'x': float(wp[0]),
                    'y': float(wp[1]),
                    'z': float(wp[2]),
                })
                self._save_log()
                self.phase1_idx += 1
                self._viz_update(f' [Phase1: {len(self.phase1_embeddings)}]')

        elif self.phase == 'TRAIN_SAE':
            wp = self.phase1_waypoints[-1]
            self._pub_traj(*wp)
            self._viz_update(' [Training SAE...]')

        elif self.phase == 'PHASE2_FLY':
            if self.phase2_idx is None:
                return
            wp = self.ucb_grid[self.phase2_idx]
            self._pub_traj(*wp)
            if self._dist(*wp) < 1.5:
                with self._img_lock:
                    frame = self.latest_rgb.copy() if self.latest_rgb is not None else None
                reward = 0.0
                if frame is not None:
                    emb = self._embed(frame)
                    reward, h = self._sae_reward(emb)
                    reward = float(reward)
                    if reward > 0.5:
                        self.landmark_map.append({'pos': wp[:2], 'reward': reward})
                        self.get_logger().info(f'Phase2[{self.phase2_idx}] R={reward:.3f} → landmark')
                dominant_concept = int(np.argmax(h[self.selective_idx])) if self.selective_idx is not None and len(self.selective_idx) > 0 else 0
                is_lm = reward > 0.5
                self._mission_log['phase2_visits'].append({
                    'idx': self.phase2_idx,
                    'x': float(wp[0]),
                    'y': float(wp[1]),
                    'reward': float(reward),
                    'dominant_concept': dominant_concept,
                    'is_landmark': is_lm,
                })
                if is_lm:
                    self._mission_log['landmarks'].append({
                        'x': float(wp[0]),
                        'y': float(wp[1]),
                        'reward': float(reward),
                        'dominant_concept': dominant_concept,
                    })
                self._save_log()
                self.phase2_idx = (self.phase2_idx + 1) % len(self.ucb_grid)
                if self.phase2_idx == 0:
                    self.get_logger().info(f'Phase 2 complete: {len(self.landmark_map)} landmarks')
                    self.phase = 'DONE'

        elif self.phase == 'DONE':
            self._pub_traj(0, 0, self.ALTITUDE)

        self._viz_update()

    def _train_sae(self):
        self.get_logger().info(f'Training SAE on {len(self.phase1_embeddings)} patches...')
        embs = np.stack(self.phase1_embeddings)
        sae = SparseAutoencoder(input_dim=384, dict_size=512, topk=32).to(self.device)
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
        ds = torch.utils.data.TensorDataset(torch.tensor(embs, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        for epoch in range(1, 31):
            for (batch,) in dl:
                batch = batch.to(self.device)
                x_hat, h = sae(batch)
                loss = F.mse_loss(x_hat, batch) + 1e-3 * h.abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % 10 == 0:
                self.get_logger().info(f'SAE epoch {epoch}/30')

        sae.eval()
        self.sae = sae

        with torch.no_grad():
            t = torch.tensor(embs, dtype=torch.float32).to(self.device)
            h = sae.encode(t).cpu().numpy()
        firing = (h > 0).mean(axis=0)
        self.selective_idx = np.where((firing >= 0.05) & (firing <= 0.30))[0]
        self.get_logger().info(f'Found {len(self.selective_idx)} selective concepts')

        self.ucb_grid = self.phase1_waypoints
        self.phase2_idx = 0
        self.phase = 'PHASE2_FLY'
        self._mission_log['phase1_embeddings_shape'] = list(embs.shape)
        self._mission_log['sae_concepts'] = len(self.selective_idx)
        acts = []
        for emb in embs:
            t = torch.tensor(emb, dtype=torch.float32)
            with torch.no_grad():
                _, hidden = self.sae(t.unsqueeze(0))
            acts.append(hidden.squeeze(0).tolist())
        self._mission_log['concept_activations'] = acts
        self._save_log()
        self.get_logger().info('→ PHASE2_FLY')

    def _sae_reward(self, emb):
        if self.sae is None or self.selective_idx is None or len(self.selective_idx) == 0:
            return 0.0
        t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h = self.sae.encode(t).squeeze(0).cpu().numpy()
        return float(h[self.selective_idx].sum()), h.copy()

    def _dist(self, tx, ty, tz):
        dx, dy, dz = self.position[0] - tx, self.position[1] - ty, self.position[2] - tz
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _pub_traj(self, x=0.0, y=0.0, z=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command, msg.param1, msg.param2 = cmd, float(p1), float(p2)
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0)
        self.get_logger().info('Arm command sent (force flag)')

    def _engage_offboard(self):
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info('Offboard mode command sent')
    

    def _save_log(self):
        with open(self._log_path, 'w') as f:
                json.dump(self._mission_log, f, indent=2)
def main(args=None):
    rclpy.init(args=args)
    node = ConceptTerrainMission()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
