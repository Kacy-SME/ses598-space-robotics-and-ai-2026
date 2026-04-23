#!/bin/bash
echo "=== SES598 System Check ==="
echo ""

# Check OS
echo "OS: $(lsb_release -d | cut -f2)"

# Check ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "✓ ROS2 Humble found"
    export ROS_DISTRO=humble
elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "✓ ROS2 Jazzy found"
    export ROS_DISTRO=jazzy
else
    echo "✗ ROS2 not found - install ROS2 first"
    exit 1
fi

# Check Python
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python: $PYTHON_VERSION"

# Check Gazebo
if command -v gz &> /dev/null; then
    GZ_VERSION=$(gz sim --version 2>&1 | head -n1)
    echo "✓ Gazebo: $GZ_VERSION"
else
    echo "⚠ Gazebo not found"
fi

# Check disk space
DISK_FREE=$(df -h ~ | tail -n1 | awk '{print $4}')
echo "Free disk space: $DISK_FREE"

# Check RAM
RAM_TOTAL=$(free -h | grep Mem | awk '{print $2}')
echo "Total RAM: $RAM_TOTAL"

echo ""
echo "=== Ready to proceed! ==="
