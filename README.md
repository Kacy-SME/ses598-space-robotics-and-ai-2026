# SES 598: Space Robotics and AI (2026)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

This repository contains course materials and assignments for **SES 598: Space Robotics and AI** taught at Arizona State University by the [DREAMS Lab](https://deepgis.org/dreamslab/).

**Course Website:** [https://deepgis.org/dreamslab/ses598/](https://deepgis.org/dreamslab/ses598/)

## 📚 Course Overview

This course explores the intersection of robotics, artificial intelligence, and space exploration. Students gain hands-on experience with:

- **Autonomous Navigation & Control**: Path planning, trajectory optimization, and controller tuning
- **Optimal Control Theory**: LQR controllers and reinforcement learning for robotic systems
- **Computer Vision & Perception**: Feature detection, SLAM, and 3D reconstruction
- **Drone Operations**: PX4 autopilot, offboard control, and mission planning
- **ROS2 Development**: Building and deploying robotic systems using modern software frameworks

All assignments use industry-standard tools including ROS2, Gazebo simulation, PX4 autopilot, and real-time visualization with RViz.

## 📖 Documentation

### New to the Course? Having issues? 
- **[Quick Start Guide](QUICK_START.md)** - Get up and running in 15 minutes
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions to common issues, check this first when stuck. It contains solutions to 50+ common issues organized by category.

## 🗂️ Repository Structure

```
ses598-space-robotics-and-ai-2026/
├── assignments/               # Course assignments
│   ├── least_squares_and_probability_theory/  # Assignment 0 (optional)
│   ├── first_order_boustrophedon_navigator/   # Assignment 1
│   ├── cart_pole_optimal_control/             # Assignment 2
│   └── terrain_mapping_drone_control/         # Assignment 3
├── lectures/                  # Lecture materials and slides
├── QUICK_START.md            # Fast-track setup guide
├── TROUBLESHOOTING.md        # Common issues and solutions
├── LICENSE
└── README.md
```

## 🎯 Assignments

### Assignment 0: Least Squares and Probability Theory
**Topics:** Least Squares Estimation, Maximum Likelihood Estimation, Probability Theory

**Points:** 100, does not count towards grade - *Optional but recommended*

This foundational assignment introduces core mathematical concepts used throughout the course. Students apply least squares estimation to determine the Pacific Plate's velocity using volcanic island age and distance data. While not graded, completing this assignment demonstrates mastery of fundamental estimation theory and may be considered in final grade determinations if needed.

**Key Skills:**
- Least squares parameter estimation
- Maximum likelihood estimation
- Statistical analysis and interpretation
- Data-driven modeling
- Uncertainty quantification

**Assignment Materials:**
- 📄 [Assignment PDF](assignments/least_squares_and_probability_theory/SES598_2026_Assignment0_least_squares_probability_theory.pdf)
- 📊 [Volcanoes Dataset](assignments/least_squares_and_probability_theory/volcanoes_data_ses598_2026.csv)
- 📖 [Lecture Notes](assignments/least_squares_and_probability_theory/SES598_2026_notes_least_squares_MLE-1-1.pdf)

---

### Assignment 1: First-Order Boustrophedon Navigator
**Topics:** PD Control, Trajectory Tracking, Coverage Path Planning
**Points:** 100 

<img width="488" height="529" alt="Boustrophedon Pattern" src="https://github.com/user-attachments/assets/100a2f14-7dde-4dd2-b730-a1d65d6cc49c" />

Students tune a PD controller to execute precise lawnmower survey patterns (boustrophedon) using the ROS2 Turtlesim. This fundamental pattern is used in space exploration, precision agriculture, and search-and-rescue operations.

**Key Skills:**
- PD controller parameter tuning
- Cross-track error minimization
- Real-time performance visualization with RQT
- Trajectory optimization

**[View Full Assignment →](assignments/first_order_boustrophedon_navigator/README.md)**

---

### Assignment 2: Cart-Pole Optimal Control
**Topics:** LQR Control, System Dynamics, Reinforcement Learning
**Points:** 100 

<img width="1069" height="1069" alt="Cart pole System Gazebo view" src="https://github.com/user-attachments/assets/9b8e438e-5cff-4679-abdb-c42d7b31a01d" />
<img src="https://github.com/user-attachments/assets/c8591475-3676-4cdf-8b4a-6539e5a2325f" alt="Cart-Pole System Rviz view" width="500"/>

Students analyze and tune an LQR controller for an inverted pendulum (cart-pole) system subject to earthquake disturbances. This assignment develops skills critical for controlling systems under dynamic perturbations, applicable to lunar landers and orbital debris removal robots.

**Key Skills:**
- LQR controller analysis and tuning
- Q and R matrix optimization
- Disturbance rejection
- Optional: Deep Q-Network (DQN) implementation for extra credit

**[Watch Demo Video](https://drive.google.com/file/d/1UEo88tqG-vV_pkRSoBF_-FWAlsZOLoIb/view?usp=sharing)** | **[View Full Assignment →](assignments/cart_pole_optimal_control/README.md)**

---

### Assignment 3: Rocky Times Challenge - Search, Map, & Analyze
**Topics:** Autonomous Drones, Computer Vision, SLAM, PX4 Control
**Points:** 100 

<img src="https://github.com/user-attachments/assets/6e3d9610-a63a-4949-88a1-a14166a9ed50" alt="Terrain Mapping Challenge" width="500"/>

Students develop an autonomous drone system to search for, map, and analyze cylindrical rock formations in unknown terrain. The drone must estimate dimensions, plan efficient search patterns, and land precisely on the target—simulating geological exploration missions.

**Key Skills:**
- PX4 offboard control
- Real-time feature detection and tracking
- 3D reconstruction with RTAB-Map
- Energy-efficient path planning
- Precision landing

**[View Full Assignment →](assignments/terrain_mapping_drone_control/README.md)**

---

## 🚀 Getting Started

> **💡 Quick Start:** For streamlined setup instructions, see the **[Quick Start Guide](QUICK_START.md)**

### System Requirements

This course supports multiple ROS2 distributions. Choose the combination that matches your system:

| Ubuntu Version | ROS2 Distribution |
|----------------|-------------------|
| 22.04 LTS | Humble |
| 23.04 | Iron |
| 23.10 | Iron |
| 24.04 LTS | Jazzy |

### Prerequisites

#### Core Tools
```bash
# Set your ROS distribution
export ROS_DISTRO=humble  # or iron, jazzy

# Install ROS2 (if not already installed)
# Follow: https://docs.ros.org/en/humble/Installation.html

# Install common dependencies
sudo apt update
sudo apt install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-ros-gz-bridge \
    ros-$ROS_DISTRO-ros-gz-sim \
    ros-$ROS_DISTRO-turtlesim \
    ros-$ROS_DISTRO-rqt* \
    python3-pip

# Install Python packages
pip3 install numpy scipy matplotlib opencv-python control
```

#### For Drone Assignments
```bash
# Install PX4 dependencies
sudo apt install -y \
    ros-$ROS_DISTRO-px4-msgs \
    ros-$ROS_DISTRO-rtabmap-ros

# Clone PX4-Autopilot (for SITL simulation)
cd ~/
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout 9ac03f03eb  # Tested version
bash Tools/setup/ubuntu.sh
```

### Repository Setup

#### Option 1: Fork and Clone (Recommended for Students)

1. **Fork this repository:**
   - Visit: [https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026](https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026)
   - Click "Fork" in the top-right corner
   - Select your GitHub account

2. **Clone your fork:**
```bash
cd ~/
git clone https://github.com/YOUR_USERNAME/ses598-space-robotics-and-ai-2026.git
```

3. **Add upstream remote (to sync with course updates):**
```bash
cd ~/ses598-space-robotics-and-ai-2026
git remote add upstream https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026.git
```

4. **Stay updated with course materials:**
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

#### Option 2: Direct Clone (Read-Only)

```bash
cd ~/
git clone https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026.git
```

### Setting Up Assignments

Each assignment should be symlinked to your ROS2 workspace:

```bash
# Create ROS2 workspace if it doesn't exist
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Symlink specific assignment (example: Assignment 1)
ln -s ~/ses598-space-robotics-and-ai-2026/assignments/first_order_boustrophedon_navigator .

# Build the package
cd ~/ros2_ws
colcon build --packages-select first_order_boustrophedon_navigator --symlink-install
source install/setup.bash
```

Repeat for each assignment as needed.

### External Resources
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [PX4 User Guide](https://docs.px4.io/main/en/)
- [Gazebo Documentation](https://gazebosim.org/docs)
- [Underactuated Robotics (MIT)](https://underactuated.mit.edu/)

### Recommended Books

**Introduction to Linear Algebra (Fifth Edition)**
- Author: Gilbert Strang
- Essential foundations for understanding state estimation, control theory, and machine learning algorithms.

**Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches**
- Author: Dan Simon
- A comprehensive guide to state estimation techniques with practical applications in navigation and control systems.

**Optimal Control and Estimation**
- Author: Robert F. Stengel
- A classic text bridging theory and practice in optimal control, estimation, and stochastic systems analysis.

**Pattern Recognition and Machine Learning**
- Author: Christopher M. Bishop
- The definitive text on modern pattern recognition methods with a focus on Bayesian techniques and machine learning algorithms.

**Multiple View Geometry in Computer Vision**
- Authors: Richard Hartley and Andrew Zisserman
- Essential resource for understanding geometric methods in computer vision and 3D reconstruction.

### Need Help?
- **[Quick Start Guide](QUICK_START.md)** - Fast-track setup for each assignment
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

### For Students

1. Fork this repository
2. Complete assignments in your fork
3. Submit assignment URLs via the course portal
4. Keep your fork updated with upstream changes

### For Instructors/Contributors

1. Create a feature branch
2. Make your changes
3. Submit a pull request with a clear description
4. Follow the existing code style and documentation patterns

## 📝 License

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)


## 👥 Course Staff

**Instructor:** Jnaneshwar Das, Arizona State University

**Course Website:** [https://deepgis.org/dreamslab/ses598/](https://deepgis.org/dreamslab/ses598/)


## 🌟 Acknowledgments

This course builds upon:
- ROS2 and the Open Robotics community
- PX4 Autopilot project
- MIT's Underactuated Robotics course materials
- Aldrin Inbaraj A.'s solutions from 2025 version of the course. 

---

**Ready to begin?** 
- 🚀 New students: Start with the [Quick Start Guide](QUICK_START.md)
- 📐 Review fundamentals: [Assignment 0: Least Squares & Probability](assignments/least_squares_and_probability_theory/) (optional)
- 📚 Jump to [Assignment 1: Boustrophedon Navigator](assignments/first_order_boustrophedon_navigator/README.md)
- 🔧 Having issues? Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
