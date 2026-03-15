# Assignment 3: Rocky Times Challenge - Search, Map, & Analyze

## Overview

This ROS2 package implements an autonomous drone system for geological feature detection, 3D mapping, and precision landing using an RGBD camera and PX4 SITL simulation. The drone autonomously searches for two cylindrical rock formations of different heights, estimates their dimensions, maps them in 3D using RTAB-Map, and performs a precision landing on the taller cylinder using ArUco marker guidance.

## Environment Setup

### Prerequisites
- ROS2 Humble
- PX4 SITL (v1.17.0-alpha)
- Gazebo Harmonic
- RTAB-Map ROS2 package
- Micro-XRCE-DDS-Agent (built from source)
- OpenCV 4.5+
- Python 3.10+

### Installation

```bash
# Clone the repository
cd ~/ros2_ws/src
git clone https://github.com/Kacy-SME/ses598-space-robotics-and-ai-2026.git
ln -s ses598-space-robotics-and-ai-2026/assignments/terrain_mapping_drone_control .

# Deploy PX4 model files
cd ~/ros2_ws/src/terrain_mapping_drone_control
bash scripts/deploy_px4_model.sh -p /root/PX4-Autopilot

# Build
cd ~/ros2_ws
colcon build --packages-select terrain_mapping_drone_control --symlink-install
source install/setup.bash
```

### Running the Mission

Start each component in order in separate terminals:

**Terminal 1 — Micro-XRCE-DDS Agent:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 2 — Gazebo/PX4 Simulation:**
```bash
ros2 launch terrain_mapping_drone_control cylinder_landing.launch.py
```

**Terminal 3 — RTAB-Map 3D Mapping:**
```bash
ros2 launch terrain_mapping_drone_control rtabmap.launch.py
```

**Terminal 4 — Autonomous Mission:**
```bash
ros2 launch terrain_mapping_drone_control mission.launch.py
```

**Optional — QGroundControl monitoring:**
```bash
cd ~/squashfs-root && ./AppRun
```

## Autonomous Mission Flow

The mission is implemented in `auto_detect_land.py` as a state machine with the following states:

- **WAIT_INTRINSICS**: Waits for camera intrinsics to be received before proceeding.
- **PRE_ARM**: Streams offboard control setpoints for 1.5 seconds before arming. This is a critical fix — PX4 requires a steady stream of offboard commands before it will accept an arm command, otherwise it immediately disarms with "preflight inaction."
- **ARM_TAKEOFF**: Two-stage takeoff: ascends vertically to 5m, then moves to the search entry point.
- **CIRCLE**: Executes a boustrophedon (lawnmower) search pattern to systematically cover the search area. Sweeps begin at y=0 to immediately pass through the cylinder locations, then expand outward in ±y strips.
- **SERVO**: Once a cylinder is detected, the drone adjusts its position to maintain a 15m standoff distance using depth measurements from the RGBD camera.
- **HOVER**: Holds position for 7 seconds accumulating bounding box measurements. Computes median width and height using pinhole projection. If dimensions match a previously seen cylinder, transitions to landing; otherwise records the new cylinder and resumes search.
- **ARUCO_HOVER**: Climbs to 20m altitude for a stable overhead view of both ArUco markers.
- **ARUCO_SELECT**: Selects the marker with the smallest z-distance (i.e., the tallest cylinder) as the landing target. Marker 0 is on the tall cylinder, Marker 1 is on the short cylinder.
- **ARUCO_MOVE**: Navigates to the selected marker position.
- **ARUCO_LAND**: Issues NAV_LAND command and disarms on touchdown.
- **COMPLETE**: Logs mission duration and battery usage.

## Key Implementation Improvements

### 1. PRE_ARM State (Arming Fix)
The original starter code fired the arm and offboard commands at a fixed counter value, which caused PX4 to immediately disarm with "Disarmed by auto preflight disarming." The fix introduces a dedicated PRE_ARM state that streams 15 setpoints (~1.5 seconds) before sending the arm command, satisfying PX4's offboard mode requirements.

### 2. Boustrophedon Search Strategy
The original implementation used a fixed circular orbit at 15m radius for search. This was replaced with a boustrophedon (lawnmower) pattern with the following advantages:
- **Guaranteed coverage**: Systematically covers the entire search area rather than orbiting at a fixed radius
- **Faster detection**: Search strips start at y=0, passing directly through the cylinder locations at (5,0) and (-5,0) on the first sweep
- **Configurable bounds**: Search area easily adjustable via `search_x_min/max` and `search_y_min/max` parameters

Search parameters:
```python
self.search_x_min = -10.0   # meters
self.search_x_max = 10.0    # meters  
self.search_y_min = -10.0   # meters
self.search_y_max = 10.0    # meters
self.search_strip_spacing = 6.0  # meters between strips
```

### 3. ArUco Marker ID Fix
Both cylinders originally shared ArUco marker ID 0, making them indistinguishable. The short cylinder's marker was updated to ID 1, enabling the ARUCO_SELECT state to correctly identify and land on the taller cylinder.

### 4. RTAB-Map Integration
RTAB-Map runs concurrently with the mission, building a 3D map of the environment using synchronized RGB and depth images from the front-facing camera. The map is saved to `~/.ros/rtabmap.db` and can be exported as a mesh at any time.

## Cylinder Detection Pipeline

Detection uses synchronized RGB and depth images:

1. **Color segmentation**: HSV thresholding isolates the cylinder color
2. **Depth filtering**: Masks points outside 1-30m range
3. **Morphological processing**: Closing operation reduces noise
4. **Contour detection**: Extracts bounding boxes around candidate objects
5. **Dimension estimation**: Pinhole projection computes real-world dimensions:
   ```
   width_m = (pixel_width * depth) / fx
   height_m = (pixel_height * depth) / fy
   ```

## Performance Results

### Cylinder Detection Accuracy

| Cylinder | Detected Dimensions | Ground Truth | Error |
|----------|-------------------|--------------|-------|
| Tall (front) | ~1.1m × 10.1m | 1.0m × 10.0m | ~±0.1m |
| Short (back) | ~1.5m × 7.0m | 1.0m × 7.0m | ~±0.5m |

### Mission Timing (3 Official Trials)

| Trial | Duration | Battery Used | Landing Target |
|-------|----------|--------------|----------------|
| 1 | 151.98s | N/A* | Tall cylinder (Marker 0) |
| 2 | 120.69s | N/A* | Tall cylinder (Marker 0) |
| 3 | 123.50s | N/A* | Tall cylinder (Marker 0) |
| **Average** | **132.1s** | | |

*Battery SOC field remained static throughout all trials in PX4 SITL — this is a known simulator limitation documented by other students as well.

**Comparison**: Reference implementation completed in 267 seconds. The tuned boustrophedon search achieves an average of 132 seconds — approximately 2x faster.

### Landing Precision
The drone consistently landed on Marker 0 (tall cylinder) with final approach positions within ~0.5m of the marker center based on ArUco pose estimates.

## 3D Reconstruction (Extra Credit)

RTAB-Map was used to build a 3D map of both cylinders during the mission. The map was exported as a polygon mesh using:

```bash
# Save RTAB-Map database
ros2 service call /rtabmap/backup std_srvs/srv/Empty "{}"

# Export as PLY mesh
mkdir -p /root/.ros/tmp
rtabmap-export --mesh --output /tmp/map.ply /root/.ros/rtabmap.db
```

The exported mesh contains **20,549 polygons** and is available at `models/terrain/map.ply`.

## Known Limitations

- Battery consumption tracking is unavailable in PX4 SITL as the voltage-based SOC estimate remains constant
- Cylinder width measurements have higher error (~±0.5m) than height measurements due to the front-facing camera angle
- The boustrophedon search bounds are tuned for this specific environment; a larger or unknown environment would require wider bounds

## License

This assignment is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
