#!/usr/bin/env python3
"""
concept_terrain_mission_v5.py
=============================
Three-phase sequential mission demonstrating both policies in a single flight.

Phase 1: Boustrophedon survey — collect DINOv2 embeddings, train SAE online.

Phase 2a (GREEDY): Visit top-15 waypoints ranked by Phase 1 SAE score.
    Pure exploitation of the Phase 1 prior.

Phase 2b (ENTROPY): Visit the bottom-15 waypoints that greedy rejected,
    using entropy-seeking active learning to order them.
    If landmarks are found here, greedy's prior was incomplete.

Scientific claim: entropy-seeking recovers value in terrain greedy dismissed.
The transition between 2a and 2b is logged and visible in the dashboard.

Logs to /tmp/mission_log_v5.json
Frames to /tmp/mission_frames_v5/
SAE checkpoint to /tmp/sae_checkpoint_v5.pth
"""

import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import pathlib
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleCommand, TrajectorySetpoint
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


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


# ── Entropy utilities ─────────────────────────────────────────────────────────
def activation_entropy(h_vec):
    h = np.clip(np.array(h_vec, dtype=np.float64), 0, None)
    total = h.sum()
    if total < 1e-9:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def predicted_entropy(mean_val, std_val):
    var = max(float(std_val) ** 2, 1e-9)
    return float(0.5 * np.log(2 * np.pi * np.e * var))


# ── Mission node ──────────────────────────────────────────────────────────────
class ConceptTerrainMissionV5(Node):
    GRID_X_MIN, GRID_X_MAX = -12.0, 11.0
    GRID_Y_MIN, GRID_Y_MAX = -10.0, 10.0
    GRID_STEP   = 5.0
    ALTITUDE    = -5.0
    REWARD_THRESHOLD = 0.5
    GREEDY_N    = 15   # top N for greedy phase
    # remaining (N_total - GREEDY_N) go to entropy phase

    def __init__(self):
        super().__init__('concept_terrain_mission')
        self.get_logger().info(
            '=== ConceptTerrainMission v5 (Greedy → Entropy Sequential) ===')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, qos)

        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Image, '/drone/front_rgb', self._image_cb, img_qos)

        # State
        self.position = [0.0, 0.0, 0.0]
        self.bridge   = CvBridge()
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

        # SAE
        self.sae           = None
        self.selective_idx = None

        # Phase 2a — greedy
        self.greedy_waypoints = None   # top GREEDY_N waypoints by score
        self.greedy_idx       = 0

        # Phase 2b — entropy on rejected waypoints
        self.entropy_waypoints = None  # bottom (N - GREEDY_N) waypoints
        self.belief_mean       = None
        self.belief_std        = None
        self.visited_mask      = None
        self.entropy_step      = 0

        # Shared
        self.landmark_map    = []
        self.greedy_landmarks = []
        self.entropy_landmarks = []

        self._frames_dir = pathlib.Path('/tmp/mission_frames_v5')
        self._frames_dir.mkdir(exist_ok=True)
        self._frame_counter = 0

        self._log_path = pathlib.Path('/tmp/mission_log_v5.json')
        self._mission_log = {
            'policy':            'greedy_then_entropy',
            'greedy_n':          self.GREEDY_N,
            'phase1_waypoints':  [],
            'phase1_scores':     [],
            'sae_concepts':      0,
            'concept_activations': [],
            'phase2a_visits':    [],   # greedy visits
            'phase2b_visits':    [],   # entropy visits
            'greedy_landmarks':  [],
            'entropy_landmarks': [],
            'landmarks':         [],   # all combined
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dinov2 = None
        self._load_dinov2()
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info('=== v5 Ready ===')

    # ── Waypoint generation ───────────────────────────────────────────────────
    def _gen_waypoints(self):
        wps = []
        cols = list(np.arange(
            self.GRID_X_MIN, self.GRID_X_MAX + self.GRID_STEP, self.GRID_STEP))
        for i, x in enumerate(cols):
            ys = list(np.arange(
                self.GRID_Y_MIN, self.GRID_Y_MAX + self.GRID_STEP, self.GRID_STEP))
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
                'facebookresearch/dinov2', 'dinov2_vits14',
                verbose=False).to(torch.device('cpu')).eval()
            self.get_logger().info('DINOv2 loaded ✓')
        except Exception as e:
            self.get_logger().error(f'DINOv2 load failed: {e}')

    def _embed(self, bgr):
        if self.dinov2 is None:
            return np.zeros(384, dtype=np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.dinov2(t).squeeze(0).cpu().numpy().astype(np.float32)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.position = list(msg.position[:3])

    def _image_cb(self, msg):
        with self._img_lock:
            try:
                img = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.latest_rgb = img
            except Exception as e:
                self.get_logger().warn(f'Image conversion failed: {e}')

    # ── Main timer ────────────────────────────────────────────────────────────
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
                self.phase = 'PHASE1_FLY'
                self.phase1_idx = 0

        elif self.phase == 'PHASE1_FLY':
            self._phase1_step()

        elif self.phase == 'TRAIN_SAE':
            self._pub_traj(*self.phase1_waypoints[-1])

        elif self.phase == 'PHASE2A_GREEDY':
            self._phase2a_step()

        elif self.phase == 'PHASE2B_ENTROPY':
            self._phase2b_step()

        elif self.phase == 'DONE':
            self._pub_traj(0.0, 0.0, self.ALTITUDE)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    def _phase1_step(self):
        if self.phase1_idx >= len(self.phase1_waypoints):
            self.get_logger().info(
                f'Phase 1 done: {len(self.phase1_embeddings)} patches. '
                f'Training SAE...')
            self.phase = 'TRAIN_SAE'
            threading.Thread(target=self._train_sae, daemon=True).start()
            return

        wp = self.phase1_waypoints[self.phase1_idx]
        self._pub_traj(*wp)
        if self._dist(*wp) < 1.5:
            with self._img_lock:
                frame = self.latest_rgb.copy() \
                    if self.latest_rgb is not None else None
            if frame is not None:
                emb = self._embed(frame)
                self.phase1_embeddings.append(emb)
                self.phase1_positions.append(wp[:2])
                fp = self._frames_dir / f'phase1_{self._frame_counter:04d}.jpg'
                cv2.imwrite(str(fp), frame)
                self._frame_counter += 1
                self.get_logger().info(
                    f'Phase1[{len(self.phase1_embeddings)}] '
                    f'at ({wp[0]:.0f},{wp[1]:.0f})')
            self._mission_log['phase1_waypoints'].append(
                {'idx': self.phase1_idx,
                 'x': float(wp[0]), 'y': float(wp[1])})
            self._save_log()
            self.phase1_idx += 1

    # ── Phase 2a: Greedy ──────────────────────────────────────────────────────
    def _phase2a_step(self):
        if self.greedy_idx >= len(self.greedy_waypoints):
            n = len(self.greedy_landmarks)
            self.get_logger().info(
                f'Phase 2a (Greedy) complete: {n} landmarks. '
                f'→ PHASE2B_ENTROPY')
            self.phase = 'PHASE2B_ENTROPY'
            return

        wp = self.greedy_waypoints[self.greedy_idx]
        self._pub_traj(*wp)
        if self._dist(*wp) < 1.5:
            reward, h_full, frame = self._observe(
                f'phase2a_{self.greedy_idx:04d}')
            is_lm = reward > self.REWARD_THRESHOLD
            dominant = self._dominant_concept(h_full)

            if is_lm:
                self.greedy_landmarks.append(
                    {'pos': wp[:2], 'reward': reward,
                     'dominant_concept': dominant})
                self.landmark_map.append(
                    {'pos': wp[:2], 'reward': reward,
                     'policy': 'greedy'})
                self.get_logger().info(
                    f'Phase2a[{self.greedy_idx}] R={reward:.3f} '
                    f'→ LANDMARK ★ (greedy)')
            else:
                self.get_logger().info(
                    f'Phase2a[{self.greedy_idx}] R={reward:.3f} '
                    f'(below threshold)')

            self._mission_log['phase2a_visits'].append({
                'idx': self.greedy_idx,
                'x': float(wp[0]), 'y': float(wp[1]),
                'reward': reward,
                'dominant_concept': dominant,
                'is_landmark': is_lm,
            })
            if is_lm:
                self._mission_log['greedy_landmarks'].append({
                    'x': float(wp[0]), 'y': float(wp[1]),
                    'reward': reward, 'dominant_concept': dominant})
                self._mission_log['landmarks'].append({
                    'x': float(wp[0]), 'y': float(wp[1]),
                    'reward': reward, 'dominant_concept': dominant,
                    'policy': 'greedy'})
            self._save_log()
            self.greedy_idx += 1

    # ── Phase 2b: Entropy on greedy's rejected waypoints ─────────────────────
    def _phase2b_step(self):
        if self.belief_mean is None:
            return

        next_idx = self._select_entropy_waypoint()
        if next_idx is None:
            self.get_logger().info(
                f'Phase 2b (Entropy) complete: '
                f'{len(self.entropy_landmarks)} landmarks found '
                f'in greedy-rejected terrain.')
            self.get_logger().info(
                f'Total: greedy={len(self.greedy_landmarks)} '
                f'entropy={len(self.entropy_landmarks)}')
            self.phase = 'DONE'
            self._save_log()
            return

        wp = self.entropy_waypoints[next_idx]
        self._pub_traj(*wp)
        if self._dist(*wp) < 1.5:
            reward, h_full, frame = self._observe(
                f'phase2b_{self.entropy_step:04d}')
            is_lm = reward > self.REWARD_THRESHOLD
            dominant = self._dominant_concept(h_full)
            obs_entr = activation_entropy(h_full)

            if is_lm:
                self.entropy_landmarks.append(
                    {'pos': wp[:2], 'reward': reward,
                     'dominant_concept': dominant})
                self.landmark_map.append(
                    {'pos': wp[:2], 'reward': reward,
                     'policy': 'entropy'})
                self.get_logger().info(
                    f'Phase2b[{self.entropy_step}] wp={next_idx} '
                    f'R={reward:.3f} H={obs_entr:.3f} '
                    f'→ LANDMARK ★ (entropy recovered!)')
            else:
                self.get_logger().info(
                    f'Phase2b[{self.entropy_step}] wp={next_idx} '
                    f'R={reward:.3f} H={obs_entr:.3f}')

            # Update belief
            self.belief_mean[next_idx] = reward
            self.belief_std[next_idx]  = 0.01
            self.visited_mask[next_idx] = True

            # Spatial neighbor update
            for j, other_wp in enumerate(self.entropy_waypoints):
                if j == next_idx or self.visited_mask[j]:
                    continue
                dist = math.sqrt(
                    (wp[0]-other_wp[0])**2 + (wp[1]-other_wp[1])**2)
                if dist < self.GRID_STEP * 1.5:
                    alpha = 0.3 * (1.0 - dist / (self.GRID_STEP * 1.5))
                    self.belief_mean[j] = (
                        (1-alpha) * self.belief_mean[j] + alpha * reward)
                    self.belief_std[j] = max(
                        self.belief_std[j] * (1 - alpha * 0.5), 0.05)

            self._mission_log['phase2b_visits'].append({
                'step': self.entropy_step,
                'wp_idx': int(next_idx),
                'x': float(wp[0]), 'y': float(wp[1]),
                'reward': reward,
                'observed_entropy': obs_entr,
                'dominant_concept': dominant,
                'is_landmark': is_lm,
            })
            if is_lm:
                self._mission_log['entropy_landmarks'].append({
                    'x': float(wp[0]), 'y': float(wp[1]),
                    'reward': reward, 'dominant_concept': dominant})
                self._mission_log['landmarks'].append({
                    'x': float(wp[0]), 'y': float(wp[1]),
                    'reward': reward, 'dominant_concept': dominant,
                    'policy': 'entropy'})
            self._save_log()
            self.entropy_step += 1

    # ── Entropy waypoint selection ────────────────────────────────────────────
    def _select_entropy_waypoint(self):
        unvisited = np.where(~self.visited_mask)[0]
        if len(unvisited) == 0:
            return None
        entropies = np.array([
            predicted_entropy(self.belief_mean[i], self.belief_std[i])
            for i in unvisited
        ])
        best = unvisited[np.argmax(entropies)]
        self.get_logger().info(
            f'Entropy selection: wp={best} H={entropies.max():.3f} '
            f'({len(unvisited)} unvisited)')
        return int(best)

    # ── Observation helper ────────────────────────────────────────────────────
    def _observe(self, frame_name):
        with self._img_lock:
            frame = self.latest_rgb.copy() \
                if self.latest_rgb is not None else None
        reward, h_full = 0.0, np.zeros(512)
        if frame is not None:
            fp = self._frames_dir / f'{frame_name}.jpg'
            cv2.imwrite(str(fp), frame)
            emb = self._embed(frame)
            reward, h_full = self._sae_reward(emb)
        return float(reward), h_full, frame

    def _dominant_concept(self, h_full):
        if self.selective_idx is not None and len(self.selective_idx) > 0:
            return int(self.selective_idx[
                np.argmax(h_full[self.selective_idx])])
        return 0

    # ── SAE training ─────────────────────────────────────────────────────────
    def _train_sae(self):
        self.get_logger().info(
            f'Training SAE on {len(self.phase1_embeddings)} patches...')
        embs = np.stack(self.phase1_embeddings)
        sae  = SparseAutoencoder(
            input_dim=384, dict_size=512, topk=32).to(self.device)
        opt  = torch.optim.Adam(sae.parameters(), lr=1e-3)
        ds   = torch.utils.data.TensorDataset(
            torch.tensor(embs, dtype=torch.float32))
        dl   = torch.utils.data.DataLoader(
            ds, batch_size=64, shuffle=True, num_workers=0)

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
        torch.save(sae.state_dict(), '/tmp/sae_checkpoint_v5.pth')
        self.get_logger().info('SAE checkpoint → /tmp/sae_checkpoint_v5.pth')

        with torch.no_grad():
            t_all = torch.tensor(embs, dtype=torch.float32).to(self.device)
            h_all = sae.encode(t_all).cpu().numpy()

        firing = (h_all > 0).mean(axis=0)
        self.selective_idx = np.where(
            (firing >= 0.05) & (firing <= 0.30))[0]
        self.get_logger().info(
            f'Selective concepts: {len(self.selective_idx)}')

        # Score all Phase 1 waypoints
        scores = h_all[:, self.selective_idx].sum(axis=1)

        # ── Split into greedy (top N) and entropy (bottom N) ──────────────────
        N = len(self.phase1_waypoints)
        ranked_order = np.argsort(scores)[::-1]  # descending

        greedy_indices  = ranked_order[:self.GREEDY_N]
        entropy_indices = ranked_order[self.GREEDY_N:]  # greedy's rejects

        self.greedy_waypoints  = [
            self.phase1_waypoints[i] for i in greedy_indices]
        self.entropy_waypoints = [
            self.phase1_waypoints[i] for i in entropy_indices]

        self.get_logger().info(
            f'Phase 2a: top {self.GREEDY_N} waypoints (greedy)')
        self.get_logger().info(
            f'Phase 2b: bottom {len(entropy_indices)} waypoints (entropy)')
        self.get_logger().info(
            f'Top-3 greedy scores: '
            f'{scores[greedy_indices[:3]].tolist()}')
        self.get_logger().info(
            f'Bottom-3 entropy scores: '
            f'{scores[entropy_indices[-3:]].tolist()}')

        # Initialize entropy beliefs for Phase 2b
        M = len(self.entropy_waypoints)
        entropy_scores = scores[entropy_indices]
        score_range = entropy_scores.max() - entropy_scores.min() + 1e-9
        norm_scores  = (entropy_scores - entropy_scores.min()) / score_range

        self.belief_mean  = norm_scores.astype(np.float64)
        self.belief_std   = np.full(M, 1.0, dtype=np.float64)
        self.visited_mask = np.zeros(M, dtype=bool)

        # Log
        self._mission_log['phase1_scores']           = scores.tolist()
        self._mission_log['sae_concepts']            = int(len(self.selective_idx))
        self._mission_log['concept_activations']     = h_all.tolist()
        self._mission_log['greedy_waypoint_indices'] = greedy_indices.tolist()
        self._mission_log['entropy_waypoint_indices']= entropy_indices.tolist()
        self._save_log()

        self.greedy_idx = 0
        self.phase = 'PHASE2A_GREEDY'
        self.get_logger().info('→ PHASE2A_GREEDY')

    # ── SAE reward ────────────────────────────────────────────────────────────
    def _sae_reward(self, emb):
        if self.sae is None or self.selective_idx is None or \
                len(self.selective_idx) == 0:
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
        msg.position = True
        msg.velocity = msg.acceleration = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _pub_traj(self, x=0.0, y=0.0, z=0.0):
        msg = TrajectorySetpoint()
        msg.position  = [float(x), float(y), float(z)]
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = cmd
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._pub_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0)
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
    node = ConceptTerrainMissionV5()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
