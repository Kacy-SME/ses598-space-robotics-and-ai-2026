#!/usr/bin/env python3
"""
Mission Metrics Logger for SES598 Assignment 3
Logs per-trial performance data to CSV for analysis and reporting.
Run this in a separate terminal alongside the mission.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, BatteryStatus
from std_msgs.msg import String
import csv
import time
import math
import os
from datetime import datetime


# Cylinder ground-truth positions (from launch file)
CYLINDER_FRONT = (5.0, 0.0)   # taller
CYLINDER_BACK  = (-5.0, 0.0)  # shorter


class MissionMetricsLogger(Node):
    def __init__(self):
        super().__init__('mission_metrics_logger')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry',
                                 self.odom_cb, qos)
        self.create_subscription(BatteryStatus, '/fmu/out/battery_status',
                                 self.battery_cb, qos)
        self.create_subscription(String, '/aruco/marker_pose',
                                 self.aruco_cb, 10)

        # State
        self.position = [0.0, 0.0, 0.0]
        self.battery_percent = None
        self.battery_start = None
        self.mission_start_time = None
        self.mission_active = False
        self.landed = False
        self.landing_position = None
        self.markers_detected = {}

        # Trial tracking
        self.trial_number = self._get_next_trial_number()
        self.csv_path = os.path.join(
            os.environ['HOME'],
            'ros2_ws', 'src', 'terrain_mapping_drone_control',
            'mission_metrics.csv'
        )
        self._init_csv()

        # Control loop
        self.create_timer(0.5, self.monitor_loop)
        self.prev_height = 0.0
        self.takeoff_detected = False
        self.land_detected = False

        self.get_logger().info(
            f'Metrics logger started — Trial #{self.trial_number}. '
            f'Writing to {self.csv_path}'
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        self.position = list(msg.position)

    def battery_cb(self, msg):
        if not math.isnan(msg.volt_based_soc_estimate):
            self.battery_percent = msg.volt_based_soc_estimate
            if self.battery_start is None and self.mission_active:
                self.battery_start = self.battery_percent

    def aruco_cb(self, msg):
        import re
        match = re.match(
            r"Marker (\d+) detected at x:([-\d.]+)m, y:([-\d.]+)m, z:([-\d.]+)m",
            msg.data
        )
        if match:
            mid = int(match.group(1))
            self.markers_detected[mid] = (
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4))
            )

    # ── Monitor Loop ─────────────────────────────────────────────────────────

    def monitor_loop(self):
        height = -self.position[2]  # NED → altitude

        # Detect takeoff (crossed 0.5m going up)
        if not self.takeoff_detected and height > 0.5:
            self.takeoff_detected = True
            self.mission_active = True
            self.mission_start_time = time.time()
            if self.battery_percent is not None:
                self.battery_start = self.battery_percent
            self.get_logger().info(
                f'Takeoff detected at {datetime.now().strftime("%H:%M:%S")}. '
                f'Mission timer started.'
            )

        # Detect landing (was above 0.5m, now below 0.2m)
        if self.takeoff_detected and not self.land_detected and \
                self.prev_height > 0.5 and height < 0.2:
            self.land_detected = True
            self.landing_position = (self.position[0], self.position[1])
            self._record_trial()

        self.prev_height = height

    # ── CSV Helpers ───────────────────────────────────────────────────────────

    def _get_next_trial_number(self):
        path = os.path.join(
            os.environ['HOME'],
            'ros2_ws', 'src', 'terrain_mapping_drone_control',
            'mission_metrics.csv'
        )
        if not os.path.exists(path):
            return 1
        with open(path, 'r') as f:
            return max(1, sum(1 for _ in f) - 1 + 1)  # rows minus header + 1

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trial', 'date', 'mission_duration_s',
                    'battery_used_pct',
                    'landing_x', 'landing_y',
                    'landing_error_front_m', 'landing_error_back_m',
                    'nearest_cylinder', 'landing_precision_m',
                    'markers_detected', 'notes'
                ])
            self.get_logger().info('Created new metrics CSV.')

    def _record_trial(self):
        duration = time.time() - self.mission_start_time if self.mission_start_time else 0.0
        battery_used = None
        if self.battery_start is not None and self.battery_percent is not None:
            battery_used = (self.battery_start - self.battery_percent) * 100.0

        lx, ly = self.landing_position or (0.0, 0.0)

        # Distance to each cylinder top
        err_front = math.sqrt((lx - CYLINDER_FRONT[0])**2 + (ly - CYLINDER_FRONT[1])**2)
        err_back  = math.sqrt((lx - CYLINDER_BACK[0])**2  + (ly - CYLINDER_BACK[1])**2)
        nearest   = 'front' if err_front < err_back else 'back'
        precision = min(err_front, err_back)

        row = [
            self.trial_number,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f'{duration:.1f}',
            f'{battery_used:.3f}' if battery_used is not None else 'N/A',
            f'{lx:.3f}', f'{ly:.3f}',
            f'{err_front:.3f}', f'{err_back:.3f}',
            nearest, f'{precision:.3f}',
            len(self.markers_detected),
            ''
        ]

        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # Print summary
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'TRIAL #{self.trial_number} COMPLETE')
        self.get_logger().info(f'  Duration:         {duration:.1f}s')
        self.get_logger().info(f'  Battery used:     {battery_used:.3f}%' if battery_used else '  Battery used:     N/A')
        self.get_logger().info(f'  Landing position: ({lx:.3f}, {ly:.3f})')
        self.get_logger().info(f'  Landing error:    {precision:.3f}m (landed on {nearest} cylinder)')
        self.get_logger().info(f'  Markers detected: {len(self.markers_detected)}')
        self.get_logger().info(f'  Results saved to: {self.csv_path}')
        self.get_logger().info('=' * 50)


def main(args=None):
    rclpy.init(args=args)
    node = MissionMetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Metrics logger stopped.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
