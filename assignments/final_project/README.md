# From Passive Mapping to Active Science: Concept-Steered Planetary Exploration

**SES 598 Space Robotics & AI — Final Project**  
Kacy Hatfield · Arizona State University

---

## Overview

An autonomous Mars drone that surveys terrain, trains a Sparse Autoencoder (SAE) on its own observations, and uses the resulting concept activations as a reward signal to allocate its remaining flight budget — without labels, ground truth, or Earth uplink.

Two reward objectives are demonstrated in a single flight:

- **Phase 2a (Greedy):** Visit the top-15 survey waypoints by SAE activation score. Pure exploitation.
- **Phase 2b (Entropy):** Visit the bottom-15 waypoints greedy rejected, ordered by Gaussian belief uncertainty. Uncertainty-driven exploration.

The same SAE drives both phases. The divergence in flight behavior is the steerability demonstration.

---

## System Requirements

| Component | Version |
|-----------|---------|
| ROS 2 | Humble |
| Gazebo | Harmonic |
| PX4 SITL | main (tested April 2026) |
| MicroXRCEAgent | v2.x |
| Python | 3.10 |
| PyTorch | 2.x |

**GPU not required** — DINOv2 inference runs on CPU in the simulation environment.

---

## Repository Structure

```
assignments/final_project/
├── final_project/
│   ├── concept_terrain_mission_v5.py   # Main mission node (3-phase sequential)
│   ├── live_mission_dashboard_v5.py    # Real-time rqt dashboard
│   └── ...
├── launch/
│   ├── concept_mission_v5_launch.py    # Primary launch file
│   └── ...
├── models/
│   └── artburysol175/                  # Sol 175 Mars terrain mesh (.stl)
├── mission_log.json                    # Logged mission results
├── sae_checkpoint.pth                  # Trained SAE weights (logged run)
└── mission_frames/                     # Per-waypoint camera frames
```

---

## Pipeline

```
Phase 1: Boustrophedon Survey
  └─ 30 waypoints, 25×20 m footprint, 5 m spacing, altitude 5 m
  └─ DINOv2-S CLS token embedding (384-dim) at each waypoint
  └─ SAE trains online on collected embeddings

SAE Training
  └─ TopK SAE: dict_size=512, k=32
  └─ Selective concepts: firing rate 5–30% across survey patches
  └─ Reward = mean activation of selective neurons

Phase 2a: Greedy Exploitation
  └─ Top-15 waypoints by Phase 1 score, visited in rank order
  └─ Landmark if reward > 0.5 threshold

Phase 2b: Entropy Exploration
  └─ Bottom-15 waypoints (greedy rejects)
  └─ Gaussian belief per cell, select argmax H(j) = 0.5·ln(2πeσ²)
  └─ Spatial neighbor belief update after each observation
```

---

## Startup Sequence

**Always follow this order — out-of-order startup causes arming failures.**

**Terminal 0 — QGroundControl**
```bash
./QGroundControl.AppImage
```
QGroundControl must be running in the background before PX4 launches.

**Terminal 1 — PX4 SITL**
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_depth_mono
```
Wait for Gazebo to fully load and the terrain mesh to appear.

**Terminal 2 — MicroXRCEAgent**
```bash
MicroXRCEAgent udp4 -p 8888
```
Start only after Gazebo has settled.

**Terminal 3 — ROS 2 Mission**
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch final_project concept_mission_v5_launch.py
```

**Terminal 4 — Dashboard (optional)**
```bash
python ~/ses598-space-robotics-and-ai-2026/assignments/final_project/final_project/live_mission_dashboard_v5.py
```

---

## Known Configuration Requirements

If arming fails, set these PX4 parameters once in the MAVLink console (QGroundControl):

```
param set COM_ARM_WO_GPS 1
param set NAV_DLL_ACT 0
param set NAV_RCL_ACT 0
param set COM_RCL_EXCEPT 4
param set UXRCE_DDS_SYNCT 0
param save
```

`UXRCE_DDS_SYNCT 0` is required to disable time sync checking, which causes arming failures in SITL.

> **Do not run `param reset_all`** — this will wipe the working parameter configuration.

To make parameter changes persistent, edit the installed site-packages file directly:
```
~/ros2_ws/install/final_project/lib/python3.10/site-packages/final_project/
```

---

## Building

```bash
cd ~/ros2_ws
colcon build --packages-select final_project --symlink-install
source install/setup.bash
```

---

## Mission Output

The mission logs to `/tmp/mission_log_v5.json` with the following structure:

```json
{
  "phase1_waypoints":      [...],   // 30 entries with x, y, z
  "phase1_scores":         [...],   // (30,) aggregate SAE reward per patch
  "concept_activations":   [...],   // (30, 512) full per-neuron activation matrix
  "sae_concepts":          63,      // number of selective concepts (5–30% firing rate)
  "phase2a_visits":        [...],   // greedy visit log with reward, dominant_concept, is_landmark
  "phase2b_visits":        [...],   // entropy visit log with reward, observed_entropy, is_landmark
  "greedy_landmarks":      [...],
  "entropy_landmarks":     [...],
  "landmarks":             [...]    // combined with policy tag
}
```

Per-waypoint camera frames are saved to `/tmp/mission_frames_v5/`.  
SAE checkpoint is saved to `/tmp/sae_checkpoint_v5.pth`.

---

## Logged Run Results

Single logged run, 40×40 m simulation environment, 25×20 m surveyed footprint.

| Policy | Landmarks | Visits | Efficiency η | Pattern |
|--------|-----------|--------|-------------|---------|
| Greedy (Phase 2a) | 9 | 14 | 0.64 | Dense clustering |
| Entropy (Phase 2b) | 1 | 15 | 0.07 | Broad coverage |

**Steerability:** 14 of 63 selective concepts have spatial selectivity S > 0.7 (p < 0.0001, selectivity-to-reward correlation).

**Notable finding:** Concept 469 fires selectively on rover hardware silhouettes — discovered with no labels.

> These results are from a single execution and should be treated as descriptive rather than statistically generalizable. A uniform-coverage baseline run is required to assess whether the Phase 1 score genuinely predicts reward in unvisited terrain.

---

## Spatial Selectivity Metric

Computed from `mission_log.json` using the `concept_activations` matrix:

```python
import json, numpy as np
from scipy.stats import entropy

with open('mission_log.json') as f:
    log = json.load(f)

h_all = np.array(log['concept_activations'])   # (30, 512)
firing = (h_all > 0).mean(axis=0)
sel_idx = np.where((firing >= 0.05) & (firing <= 0.30))[0]

H_max = np.log(h_all.shape[0])
selectivity = {}
for k in sel_idx:
    acts = h_all[:, k]
    if acts.sum() == 0: continue
    p = acts / acts.sum()
    selectivity[k] = 1.0 - entropy(p) / H_max

S = np.array(list(selectivity.values()))
print(f"Mean S: {S.mean():.3f}  |  High-selectivity (S>0.7): {(S>0.7).sum()}")
```

---

## Honest Scope

- **Simulation only.** Gazebo terrain textures do not replicate Mars photometric properties.
- **Earth-pretrained vision model.** DINOv2 was pretrained on web imagery. Whether its embedding geometry transfers to real Mars geology is an open empirical question.
- **Small surveyed footprint.** 25×20 m is far below operational Mars exploration scales.
- **Single logged run.** All quantitative claims are descriptive summaries of one execution.
- **No baseline comparison.** Whether greedy or entropy ordering outperforms random selection of the same waypoints remains untested.

---

## Future Work

1. Uniform-coverage baseline run to assess whether Phase 1 scores genuinely predict reward in unvisited terrain
2. Return-pass re-identification evaluation (cosine similarity vs. uniform baseline)
3. Concept clustering by terrain class (silhouette score, linear separability)
4. Domain-specific SAE pretraining on HiRISE and CRISM orbital imagery
5. Integration with horizon re-registration pipeline for orbital-surface correspondence at correct physical scale

---

## Citation

```bibtex
@misc{hatfield2026conceptsteered,
  title  = {From Passive Mapping to Active Science: Concept-Steered Planetary Exploration},
  author = {Hatfield, Kacy},
  year   = {2026},
  note   = {SES 598 Space Robotics and AI, Arizona State University}
}
```

---

## References

- Auer et al. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47, 235–256.
- Bricken et al. (2023). Towards Monosemanticity. *Transformer Circuits Thread*.
- Bau et al. (2017). Network Dissection. *CVPR 2017*.
- Joseph et al. (2025). Steering CLIP's Vision Transformer with Sparse Autoencoders. arXiv:2504.08729.
- Latorella et al. (2025). Assessment of Communication Delay Research for Missions Beyond LEO. NASA/TM–20250003885.
- Oquab et al. (2023). DINOv2. arXiv:2304.07193.
- Stevens et al. (2025). Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models. arXiv:2502.06755.
- Templeton et al. (2024). Scaling Monosemanticity. Anthropic Technical Report.
- Das, J. (2026). SES 598 Space Robotics and AI: Terrain Mapping Drone Control Models. ASU DREAMS Lab.

---

*Arizona State University · SES 598 Space Robotics and AI · 2026*
