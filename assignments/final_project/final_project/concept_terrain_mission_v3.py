#!/usr/bin/env python3
"""
concept_terrain_mission_v3.py

Changes from v2:
- Phase 2 waypoints are now ranked by Phase 1 SAE concept activation scores
  (highest reward first), not replayed in grid order.
- UCB variable name replaced with ranked_grid to reflect actual behavior.
- Early stopping: Phase 2 halts once reward drops below REWARD_THRESHOLD
  for EARLY_STOP_PATIENCE consecutive waypoints.
- Frames saved to /tmp/mission_frames/ for post-hoc GradCAM audit.
- SAE checkpoint saved to /tmp/sae_checkpoint.pth.
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

#import matplotlib.pyplot as plt


# ── SAE ───────────────────────────────────────────────────────────────────────

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


# ── Mission node ──────────────────────────────────────────────────────────────

class ConceptTerrainMission(Node):

    GRID_X_MIN, GRID_X_MAX = -12.0, 11.0
    GRID_Y_MIN, GRID_Y_MAX = -10.0, 10.0
    GRID_STEP   = 5.0
    ALTITUDE    = -5.0

    # Phase 2 landmark guidance
    REWARD_THRESHOLD   = 0.5   # below this is not a landmark
    EARLY_STOP_PATIENCE = 8    # stop after this many consecutive sub-threshold visits

    def __init__(self):
        super().__init__('concept_terrain_mission')
        self.get_logger().info('=== ConceptTerrainMission v3 Initializing ===')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub     = self.create_publisher(TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  qos)
        self.cmd_pub      = self.create_publisher(VehicleCommand,      '/fmu/in/vehicle_command',      qos)

        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, qos)

        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/drone/front_rgb', self._image_cb, img_qos)

        # State
        self.position   = [0.0, 0.0, 0.0]
        self.bridge     = CvBridge()
        self.latest_rgb = None
        self._img_lock  = threading.Lock()

        self.offboard_counter = 0
        self.phase            = 'PRE_ARM'
        self.pre_arm_counter  = 0
        self.engage_counter   = 0
        self.PRE_ARM_CYCLES   = 50

        self.phase1_waypoints  = self._gen_waypoints()
        self.phase1_idx        = 0
        self.phase1_embeddings = []
        self.phase1_positions  = []
        self.phase1_scores     = []   # SAE reward score per Phase 1 waypoint

        # SAE + Phase 2
        self.sae            = None
        self.selective_idx  = None
        self.ranked_grid    = None    # Phase 1 waypoints sorted by descending reward
        self.phase2_idx     = None
        self.landmark_map   = []
        self._consec_below  = 0       # consecutive sub-threshold counter for early stop

        # Frame saving (for GradCAM audit)
        self._frames_dir   = pathlib.Path('/tmp/mission_frames')
        self._frames_dir.mkdir(exist_ok=True)
        self._frame_counter = 0

        # Logging
        self._log_path    = pathlib.Path('/tmp/mission_log.json')
        self._mission_log = {
            'phase1_waypoints':         [],
            'phase1_embeddings_shape':  None,
            'phase1_scores':            [],   # reward score at each Phase 1 waypoint
            'sae_concepts':             0,
            'concept_activations':      [],
            'phase2_visits':            [],
            'landmarks':                [],
            'ranked_grid':              [],   # order Phase 2 actually flew
        }

        # DINOv2
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dinov2  = None
        self._load_dinov2()

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._init_viz()
        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info('=== ConceptTerrainMission v3 Ready ===')

    # ── Waypoint generation ───────────────────────────────────────────────────

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

    # ── DINOv2 ────────────────────────────────────────────────────────────────

    def _load_dinov2(self):
        self.get_logger().info('Loading DINOv2...')
        try:
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', verbose=False
            ).to(torch.device('cpu')).eval()
            self.get_logger().info('DINOv2 loaded ✓')
        except Exception as e:
            self.get_logger().error(f'DINOv2 load failed: {e}')

    def _embed(self, bgr_image):
        if self.dinov2 is None:
            return np.zeros(384, dtype=np.float32)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        t   = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.dinov2(t).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.position = list(msg.position[:3])

    def _image_cb(self, msg):
        with self._img_lock:
            try:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().warn(f'Image conversion failed: {e}')

    # ── Viz ───────────────────────────────────────────────────────────────────

    def _init_viz(self):
        #plt.ion()
        #self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        #for ax in self.axes:
         #   ax.text(0.5, 0.5, 'Waiting...', ha='center', va='center', transform=ax.transAxes)
        #plt.tight_layout()
        #plt.pause(0.01)
        pass

    def _viz_update(self, suffix=''):
        #for ax in self.axes:
           # ax.cla()
        #self.axes[0].set_title(f'Phase: {self.phase}')
        #self.axes[1].set_title(f'Landmarks: {len(self.landmark_map)}')
        #self.axes[2].set_title(f'Patches: {len(self.phase1_embeddings)}{suffix}')

        # Plot landmark positions if any
        #if self.landmark_map:
        #    xs = [l['pos'][0] for l in self.landmark_map]
         #   ys = [l['pos'][1] for l in self.landmark_map]
         #   rs = [l['reward']  for l in self.landmark_map]
         #   sc = self.axes[1].scatter(xs, ys, c=rs, cmap='hot', vmin=0, vmax=1, s=80)
          #  self.axes[1].set_xlim(self.GRID_X_MIN - 5, self.GRID_X_MAX + 5)
          #  self.axes[1].set_ylim(self.GRID_Y_MIN - 5, self.GRID_Y_MAX + 5)
          #  self.fig.colorbar(sc, ax=self.axes[1], label='reward')

       # plt.pause(0.001)
       pass

    # ── Main timer callback ───────────────────────────────────────────────────

    def _timer_cb(self):
        self._pub_offboard()
        self.offboard_counter += 1

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
                self.get_logger().info('Offboard confirmed -> arming')
                self._arm()
                self.phase = 'ARM_TAKEOFF'

        elif self.phase == 'ARM_TAKEOFF':
            target = (0.0, 0.0, self.ALTITUDE)
            self._pub_traj(*target)
            if self._dist(*target) < 1.5:
                self.get_logger().info('Takeoff complete → PHASE1_FLY')
                self.phase     = 'PHASE1_FLY'
                self.phase1_idx = 0

        elif self.phase == 'PHASE1_FLY':
            if self.phase1_idx >= len(self.phase1_waypoints):
                self.get_logger().info(
                    f'Phase 1 done: {len(self.phase1_embeddings)} patches. Training SAE...'
                )
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

                    # Save frame for GradCAM audit
                    frame_path = self._frames_dir / f'phase1_{self._frame_counter:04d}.jpg'
                    cv2.imwrite(str(frame_path), frame)
                    self._frame_counter += 1

                    self.get_logger().info(
                        f'Phase1[{len(self.phase1_embeddings)}] at ({wp[0]:.0f},{wp[1]:.0f})'
                    )

                self._mission_log['phase1_waypoints'].append(
                    {'idx': self.phase1_idx, 'x': float(wp[0]), 'y': float(wp[1]), 'z': float(wp[2])}
                )
                self._save_log()
                self.phase1_idx += 1
                self._viz_update(f' [Phase1: {len(self.phase1_embeddings)}]')

        elif self.phase == 'TRAIN_SAE':
            wp = self.phase1_waypoints[-1]
            self._pub_traj(*wp)
            self._viz_update(' [Training SAE...]')

        elif self.phase == 'PHASE2_FLY':
            if self.phase2_idx is None or self.ranked_grid is None:
                return

            # Early stop: all waypoints visited or patience exhausted
            if self.phase2_idx >= len(self.ranked_grid):
                self.get_logger().info(
                    f'Phase 2 complete: {len(self.landmark_map)} landmarks'
                )
                self.phase = 'DONE'
                return

            wp     = self.ranked_grid[self.phase2_idx]
            self._pub_traj(*wp)

            if self._dist(*wp) < 1.5:
                with self._img_lock:
                    frame = self.latest_rgb.copy() if self.latest_rgb is not None else None

                reward = 0.0
                h_full = np.zeros(512)

                if frame is not None:
                    # Save frame
                    frame_path = self._frames_dir / f'phase2_{self.phase2_idx:04d}.jpg'
                    cv2.imwrite(str(frame_path), frame)

                    emb = self._embed(frame)
                    reward, h_full = self._sae_reward(emb)
                    reward = float(reward)

                    if reward > self.REWARD_THRESHOLD:
                        self.landmark_map.append({'pos': wp[:2], 'reward': reward})
                        self._consec_below = 0
                        self.get_logger().info(
                            f'Phase2[{self.phase2_idx}] R={reward:.3f} → LANDMARK ★'
                        )
                    else:
                        self._consec_below += 1
                        self.get_logger().info(
                            f'Phase2[{self.phase2_idx}] R={reward:.3f} '
                            f'(below threshold {self._consec_below}/{self.EARLY_STOP_PATIENCE})'
                        )
                        if self._consec_below >= self.EARLY_STOP_PATIENCE:
                            self.get_logger().info(
                                f'Early stop: {self.EARLY_STOP_PATIENCE} consecutive '
                                f'sub-threshold visits. Phase 2 done.'
                            )
                            self.phase = 'DONE'
                            self._save_log()
                            return

                dominant_concept = (
                    int(self.selective_idx[np.argmax(h_full[self.selective_idx])])
                    if self.selective_idx is not None and len(self.selective_idx) > 0
                    else 0
                )
                is_lm = reward > self.REWARD_THRESHOLD

                self._mission_log['phase2_visits'].append({
                    'idx':              self.phase2_idx,
                    'x':                float(wp[0]),
                    'y':                float(wp[1]),
                    'reward':           reward,
                    'dominant_concept': dominant_concept,
                    'is_landmark':      is_lm,
                })
                if is_lm:
                    self._mission_log['landmarks'].append({
                        'x': float(wp[0]), 'y': float(wp[1]),
                        'reward': reward, 'dominant_concept': dominant_concept,
                    })

                self._save_log()
                self.phase2_idx += 1
                self._viz_update()

        elif self.phase == 'DONE':
            self._pub_traj(0.0, 0.0, self.ALTITUDE)


    # ── SAE training + Phase 2 ranking ────────────────────────────────────────

    def _train_sae(self):
        self.get_logger().info(f'Training SAE on {len(self.phase1_embeddings)} patches...')
        embs = np.stack(self.phase1_embeddings)          # (N, 384)
        sae  = SparseAutoencoder(input_dim=384, dict_size=512, topk=32).to(self.device)
        opt  = torch.optim.Adam(sae.parameters(), lr=1e-3)
        ds   = torch.utils.data.TensorDataset(torch.tensor(embs, dtype=torch.float32))
        dl   = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        for epoch in range(1, 31):
            for (batch,) in dl:
                batch = batch.to(self.device)
                x_hat, h = sae(batch)
                loss = F.mse_loss(x_hat, batch) + 1e-3 * h.abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if epoch % 10 == 0:
                self.get_logger().info(f'SAE epoch {epoch}/30')

        sae.eval()
        self.sae = sae

        # Save checkpoint for GradCAM audit
        torch.save(sae.state_dict(), '/tmp/sae_checkpoint.pth')
        self.get_logger().info('SAE checkpoint saved → /tmp/sae_checkpoint.pth')

        # Selective concepts: firing rate 5–30%
        with torch.no_grad():
            t_all = torch.tensor(embs, dtype=torch.float32).to(self.device)
            h_all = sae.encode(t_all).cpu().numpy()     # (N, 512)

        firing = (h_all > 0).mean(axis=0)
        self.selective_idx = np.where((firing >= 0.05) & (firing <= 0.30))[0]
        self.get_logger().info(f'Selective concepts: {len(self.selective_idx)}')

        # Score every Phase 1 waypoint by selective concept activation sum
        scores = h_all[:, self.selective_idx].sum(axis=1)   # (N,)
        self.phase1_scores = scores.tolist()
        self._mission_log['phase1_scores'] = self.phase1_scores

        # Rank Phase 1 waypoints descending by score → this is Phase 2's flight order
        ranked_order = np.argsort(scores)[::-1]
        self.ranked_grid = [self.phase1_waypoints[i] for i in ranked_order]

        self.get_logger().info(
            f'Top-3 Phase 1 scores: {scores[ranked_order[:3]].tolist()}'
        )
        self.get_logger().info(
            f'Top-3 waypoints: {[self.phase1_waypoints[i] for i in ranked_order[:3]]}'
        )

        # Log ranked grid
        self._mission_log['ranked_grid'] = [
            {'rank': int(r), 'original_idx': int(i),
             'x': float(self.phase1_waypoints[i][0]),
             'y': float(self.phase1_waypoints[i][1]),
             'score': float(scores[i])}
            for r, i in enumerate(ranked_order)
        ]

        # Log full concept activations
        self._mission_log['phase1_embeddings_shape'] = list(embs.shape)
        self._mission_log['sae_concepts']            = int(len(self.selective_idx))
        self._mission_log['concept_activations']     = h_all.tolist()
        self._save_log()

        self.phase2_idx    = 0
        self._consec_below = 0
        self.phase         = 'PHASE2_FLY'
        self.get_logger().info('→ PHASE2_FLY (landmark-guided, score-ranked)')

    # ── SAE reward ────────────────────────────────────────────────────────────

    def _sae_reward(self, emb):
        if self.sae is None or self.selective_idx is None or len(self.selective_idx) == 0:
            return 0.0, np.zeros(512)
        t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h = self.sae.encode(t).squeeze(0).cpu().numpy()
        return float(h[self.selective_idx].mean()), h.copy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dist(self, tx, ty, tz):
        dx = self.position[0] - tx
        dy = self.position[1] - ty
        dz = self.position[2] - tz
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position  = True
        msg.velocity  = False
        msg.acceleration = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _pub_traj(self, x=0.0, y=0.0, z=0.0):
        msg = TrajectorySetpoint()
        msg.position  = [float(x), float(y), float(z)]
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command          = cmd
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system    = msg.target_component = 1
        msg.source_system    = msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0)
        self.get_logger().info('Arm command sent')

    def _engage_offboard(self):
        self._pub_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info('Offboard mode command sent')

    def _save_log(self):
        with open(self._log_path, 'w') as f:
            json.dump(self._mission_log, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ConceptTerrainMission()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
