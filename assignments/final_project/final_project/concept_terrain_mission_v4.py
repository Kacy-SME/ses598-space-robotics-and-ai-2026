#!/usr/bin/env python3
"""
concept_terrain_mission_v4.py
=============================
Changes from v3:
- Phase 2 uses ENTROPY-SEEKING waypoint selection instead of ranked greedy.
- After each visit, beliefs over concept activations are updated using the
  observed SAE activation vector.
- Next waypoint is selected by MAXIMUM ENTROPY REDUCTION over unvisited arms:
    next_wp = argmax_w H(belief_w)
  where H is Shannon entropy of the normalized predicted activation distribution.
- This instantiates Bayesian active learning over the concept space:
  the drone seeks terrain that maximally reduces uncertainty about which
  concepts are present in the environment.
- Both policies (greedy v3 vs entropy v4) log enough data to compare
  concept coverage and spatial distribution post-hoc.
- Early stop removed (inherited from v3 edit).
- Frames saved to /tmp/mission_frames_v4/ to avoid collision with v3 runs.
- SAE checkpoint saved to /tmp/sae_checkpoint_v4.pth.
- Mission log saved to /tmp/mission_log_v4.json.
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
    """
    Shannon entropy of a normalized activation vector.
    Treats activations as an unnormalized probability distribution.
    Higher entropy = more uncertain / more spread across concepts.
    Zero activation = zero entropy (certain: nothing fires here).
    """
    h = np.array(h_vec, dtype=np.float64)
    h = np.clip(h, 0, None)
    total = h.sum()
    if total < 1e-9:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def predicted_entropy(belief_mean, belief_std):
    """
    Expected entropy of a Gaussian belief over activation strengths.
    Uses the analytical entropy of a Gaussian: 0.5 * log(2*pi*e*sigma^2).
    Summed over selective concept dimensions.
    Higher value = more uncertain about what will activate here.
    """
    var = np.clip(belief_std ** 2, 1e-9, None)
    return float(0.5 * np.sum(np.log(2 * np.pi * np.e * var)))


# ── Mission node ──────────────────────────────────────────────────────────────
class ConceptTerrainMissionV4(Node):
    GRID_X_MIN, GRID_X_MAX = -12.0, 11.0
    GRID_Y_MIN, GRID_Y_MAX = -10.0, 10.0
    GRID_STEP   = 5.0
    ALTITUDE    = -5.0
    REWARD_THRESHOLD = 0.5

    def __init__(self):
        super().__init__('concept_terrain_mission')
        self.get_logger().info('=== ConceptTerrainMission v4 (Entropy-Seeking) Initializing ===')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub     = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',  qos)
        self.cmd_pub      = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',      qos)
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, qos)

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

        # SAE
        self.sae           = None
        self.selective_idx = None

        # ── Entropy-seeking Phase 2 state ─────────────────────────────────────
        # belief_mean[i], belief_std[i]: Gaussian belief over selective concept
        # activation sum at waypoint i. Initialized from Phase 1 scores after
        # SAE training, then updated online as Phase 2 visits arrive.
        self.belief_mean   = None   # (N_wps,) float array
        self.belief_std    = None   # (N_wps,) float array — uncertainty
        self.visited_mask  = None   # (N_wps,) bool
        self.phase2_queue  = None   # ordered list of waypoint indices to visit
        self.phase2_step   = 0      # how many Phase 2 visits completed
        self.landmark_map  = []

        # Frame + log paths (v4-specific to avoid collision with v3)
        self._frames_dir = pathlib.Path('/tmp/mission_frames_v4')
        self._frames_dir.mkdir(exist_ok=True)
        self._frame_counter = 0
        self._log_path    = pathlib.Path('/tmp/mission_log_v4.json')
        self._mission_log = {
            'policy':               'entropy_seeking',
            'phase1_waypoints':     [],
            'phase1_scores':        [],
            'sae_concepts':         0,
            'concept_activations':  [],
            'phase2_visits':        [],   # includes entropy at each step
            'landmarks':            [],
            'visit_order':          [],   # actual order Phase 2 flew
        }

        # DINOv2
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dinov2  = None
        self._load_dinov2()
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info('=== ConceptTerrainMission v4 Ready ===')

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
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.latest_rgb = img
            except Exception as e:
                self.get_logger().warn(f'Image conversion failed: {e}')

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
                self.phase      = 'PHASE1_FLY'
                self.phase1_idx = 0

        elif self.phase == 'PHASE1_FLY':
            if self.phase1_idx >= len(self.phase1_waypoints):
                self.get_logger().info(
                    f'Phase 1 done: {len(self.phase1_embeddings)} patches. Training SAE...')
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
                    frame_path = self._frames_dir / f'phase1_{self._frame_counter:04d}.jpg'
                    cv2.imwrite(str(frame_path), frame)
                    self._frame_counter += 1
                    self.get_logger().info(
                        f'Phase1[{len(self.phase1_embeddings)}] at '
                        f'({wp[0]:.0f},{wp[1]:.0f})')
                self._mission_log['phase1_waypoints'].append(
                    {'idx': self.phase1_idx,
                     'x': float(wp[0]), 'y': float(wp[1]), 'z': float(wp[2])})
                self._save_log()
                self.phase1_idx += 1

        elif self.phase == 'TRAIN_SAE':
            wp = self.phase1_waypoints[-1]
            self._pub_traj(*wp)

        elif self.phase == 'PHASE2_FLY':
            self._phase2_step()

        elif self.phase == 'DONE':
            self._pub_traj(0.0, 0.0, self.ALTITUDE)

    # ── Phase 2: entropy-seeking step ────────────────────────────────────────
    def _phase2_step(self):
        if self.belief_mean is None or self.visited_mask is None:
            return

        # Select next waypoint: maximum predicted entropy among unvisited
        next_idx = self._select_entropy_waypoint()

        if next_idx is None:
            self.get_logger().info(
                f'Phase 2 complete: all waypoints visited. '
                f'{len(self.landmark_map)} landmarks.')
            self.phase = 'DONE'
            self._save_log()
            return

        wp = self.phase1_waypoints[next_idx]
        self._pub_traj(*wp)

        if self._dist(*wp) < 1.5:
            with self._img_lock:
                frame = self.latest_rgb.copy() if self.latest_rgb is not None else None

            reward   = 0.0
            h_full   = np.zeros(512)
            obs_entr = 0.0

            if frame is not None:
                frame_path = self._frames_dir / f'phase2_{self.phase2_step:04d}.jpg'
                cv2.imwrite(str(frame_path), frame)
                emb    = self._embed(frame)
                reward, h_full = self._sae_reward(emb)
                reward = float(reward)
                obs_entr = activation_entropy(h_full)

                # ── Belief update (Bayesian online update) ────────────────────
                # Treat observed selective activation sum as a noisy observation.
                # Update belief for this arm with a simple running mean + std.
                # Also update neighbors within GRID_STEP distance (spatial prior).
                self._update_belief(next_idx, reward, h_full)

            self.visited_mask[next_idx] = True

            is_lm = reward > self.REWARD_THRESHOLD
            if is_lm:
                self.landmark_map.append({
                    'pos': wp[:2], 'reward': reward})

            dominant_concept = (
                int(self.selective_idx[np.argmax(h_full[self.selective_idx])])
                if self.selective_idx is not None and len(self.selective_idx) > 0
                else 0
            )

            pred_entr_before = predicted_entropy(
                self.belief_mean[[next_idx]],
                self.belief_std[[next_idx]]
            )

            self.get_logger().info(
                f'Phase2[{self.phase2_step}] wp={next_idx} '
                f'R={reward:.3f} H_obs={obs_entr:.3f} '
                f'H_pred={pred_entr_before:.3f} '
                f'{"→ LANDMARK ★" if is_lm else "(below threshold)"}')

            self._mission_log['phase2_visits'].append({
                'step':              self.phase2_step,
                'wp_idx':            int(next_idx),
                'x':                 float(wp[0]),
                'y':                 float(wp[1]),
                'reward':            reward,
                'observed_entropy':  obs_entr,
                'predicted_entropy': pred_entr_before,
                'dominant_concept':  dominant_concept,
                'is_landmark':       is_lm,
            })
            self._mission_log['visit_order'].append(int(next_idx))

            if is_lm:
                self._mission_log['landmarks'].append({
                    'x': float(wp[0]), 'y': float(wp[1]),
                    'reward': reward,
                    'dominant_concept': dominant_concept,
                })

            self._save_log()
            self.phase2_step += 1

    # ── Entropy waypoint selection ────────────────────────────────────────────
    def _select_entropy_waypoint(self):
        """
        Select the unvisited waypoint with highest predicted entropy.
        Entropy is computed from the current belief (mean, std) over
        selective concept activation sum at each waypoint.
        """
        unvisited = np.where(~self.visited_mask)[0]
        if len(unvisited) == 0:
            return None

        entropies = np.array([
            predicted_entropy(
                self.belief_mean[[i]],
                self.belief_std[[i]]
            )
            for i in unvisited
        ])

        best = unvisited[np.argmax(entropies)]
        self.get_logger().info(
            f'Entropy selection: wp={best} '
            f'H={entropies.max():.3f} '
            f'(from {len(unvisited)} unvisited)')
        return int(best)

    # ── Belief update ─────────────────────────────────────────────────────────
    def _update_belief(self, idx, reward, h_full):
        """
        Update belief at visited waypoint and propagate to spatial neighbors.
        Visited waypoint: collapse std to near-zero (observed).
        Neighbors within GRID_STEP*1.5: partial update toward observed reward.
        This implements a simple spatial prior: nearby terrain is likely similar.
        """
        # Collapse belief at observed waypoint
        self.belief_mean[idx] = reward
        self.belief_std[idx]  = 0.01  # near-certain after observation

        # Spatial neighbor update
        wp = self.phase1_waypoints[idx]
        for j, other_wp in enumerate(self.phase1_waypoints):
            if j == idx or self.visited_mask[j]:
                continue
            dist = math.sqrt(
                (wp[0] - other_wp[0])**2 + (wp[1] - other_wp[1])**2)
            if dist < self.GRID_STEP * 1.5:
                # Partial belief update weighted by proximity
                alpha = 0.3 * (1.0 - dist / (self.GRID_STEP * 1.5))
                self.belief_mean[j] = (
                    (1 - alpha) * self.belief_mean[j] + alpha * reward)
                # Reduce uncertainty for neighbors (but not fully)
                self.belief_std[j] = max(
                    self.belief_std[j] * (1 - alpha * 0.5), 0.05)

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
        torch.save(sae.state_dict(), '/tmp/sae_checkpoint_v4.pth')
        self.get_logger().info('SAE checkpoint saved → /tmp/sae_checkpoint_v4.pth')

        # Selective concepts: firing rate 5-30%
        with torch.no_grad():
            t_all = torch.tensor(embs, dtype=torch.float32).to(self.device)
            h_all = sae.encode(t_all).cpu().numpy()

        firing = (h_all > 0).mean(axis=0)
        self.selective_idx = np.where((firing >= 0.05) & (firing <= 0.30))[0]
        self.get_logger().info(
            f'Selective concepts: {len(self.selective_idx)}')

        # Phase 1 scores (selective concept activation sum per patch)
        scores = h_all[:, self.selective_idx].sum(axis=1)

        # ── Initialize beliefs for entropy-seeking Phase 2 ────────────────────
        N = len(self.phase1_waypoints)

        # Prior mean = Phase 1 score (normalized to [0,1])
        score_range = scores.max() - scores.min() + 1e-9
        norm_scores = (scores - scores.min()) / score_range

        # belief_mean initialized from Phase 1 scores
        # belief_std initialized uniformly high (maximum uncertainty)
        # Waypoints with no Phase 1 observation get mean=0.5 (neutral prior)
        self.belief_mean  = np.full(N, 0.5, dtype=np.float64)
        self.belief_std   = np.full(N, 1.0, dtype=np.float64)  # high uncertainty
        self.visited_mask = np.zeros(N, dtype=bool)

        # Update beliefs for Phase 1 visited waypoints
        for i, score in enumerate(norm_scores):
            if i < N:
                self.belief_mean[i]  = float(score)
                self.belief_std[i]   = 0.3  # reduced uncertainty from Phase 1 obs

        # Log
        self._mission_log['phase1_scores']            = scores.tolist()
        self._mission_log['sae_concepts']             = int(len(self.selective_idx))
        self._mission_log['concept_activations']      = h_all.tolist()
        self._mission_log['phase1_embeddings_shape']  = list(embs.shape)
        self._mission_log['belief_init_mean']         = self.belief_mean.tolist()
        self._mission_log['belief_init_std']          = self.belief_std.tolist()
        self._save_log()

        self.phase2_step = 0
        self.phase       = 'PHASE2_FLY'
        self.get_logger().info(
            '→ PHASE2_FLY (entropy-seeking active learning)')

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
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.timestamp    = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _pub_traj(self, x=0.0, y=0.0, z=0.0):
        msg = TrajectorySetpoint()
        msg.position  = [float(x), float(y), float(z)]
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.traj_pub.publish(msg)

    def _pub_cmd(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command            = cmd
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system      = msg.target_component = 1
        msg.source_system      = msg.source_component = 1
        msg.from_external      = True
        msg.timestamp          = self.get_clock().now().nanoseconds // 1000
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
    node = ConceptTerrainMissionV4()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
