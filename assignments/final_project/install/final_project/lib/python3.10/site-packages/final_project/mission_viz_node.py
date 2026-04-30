#!/usr/bin/env python3
"""
mission_viz_node.py
-------------------
ROS2 node that publishes:
  1. /viz/flight_trail     — LINE_STRIP showing drone path in real time
  2. /viz/concept_markers  — SPHERE_LIST showing SAE concept activations
                             at each Phase 2 landmark (color = dominant concept)

Add to your launch file:
    viz_node = Node(
        package='final_project',
        executable='mission_viz',
        name='mission_viz',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    # Add mission_viz to LaunchDescription list

Add to setup.py entry_points:
    'mission_viz = final_project.mission_viz_node:main',
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import json
import pathlib
import threading
import time


# ── Concept color palette (one color per concept, up to 8) ──────────────────
CONCEPT_COLORS = [
    (0.95, 0.30, 0.10, 1.0),  # C0 — Mars red
    (0.20, 0.75, 0.85, 1.0),  # C1 — cyan
    (0.95, 0.80, 0.10, 1.0),  # C2 — gold
    (0.50, 0.90, 0.30, 1.0),  # C3 — green
    (0.85, 0.20, 0.85, 1.0),  # C4 — magenta
    (0.30, 0.50, 0.95, 1.0),  # C5 — blue
    (0.95, 0.55, 0.10, 1.0),  # C6 — orange
    (0.70, 0.70, 0.70, 1.0),  # C7 — grey
]


def rgba(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


class MissionVizNode(Node):

    LOG_PATH = pathlib.Path('/tmp/mission_log.json')
    POLL_INTERVAL = 2.0   # seconds between log re-reads

    def __init__(self):
        super().__init__('mission_viz')

        # ── Publishers ──────────────────────────────────────────────────────
        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._trail_pub = self.create_publisher(
            Marker, '/viz/flight_trail', latched_qos)
        self._concept_pub = self.create_publisher(
            MarkerArray, '/viz/concept_markers', latched_qos)

        # ── Odometry subscriber ─────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry,
            '/fmu/out/vehicle_odometry',
            self._odom_cb,
            best_effort_qos,
        )

        # ── State ───────────────────────────────────────────────────────────
        self._trail_points = []          # list of (x, y, z) floats
        self._last_pos = None
        self._min_dist = 0.5             # only add point if moved > 0.5 m
        self._known_landmarks = 0        # how many landmarks already rendered

        # ── Timers ──────────────────────────────────────────────────────────
        self.create_timer(0.2,                self._publish_trail)
        self.create_timer(self.POLL_INTERVAL, self._poll_log)

        self.get_logger().info('MissionVizNode ready — watching odometry + log')

    # ── Odometry callback: accumulate trail ─────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        if self._last_pos is not None:
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            dz = z - self._last_pos[2]
            if (dx*dx + dy*dy + dz*dz) ** 0.5 < self._min_dist:
                return

        self._trail_points.append((x, y, z))
        self._last_pos = (x, y, z)

    # ── Publish LINE_STRIP trail marker ─────────────────────────────────────
    def _publish_trail(self):
        if len(self._trail_points) < 2:
            return

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'flight_trail'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.scale.x = 0.3          # line width in metres
        m.color = rgba(0.95, 0.55, 0.10, 0.85)   # orange trail

        m.pose.orientation.w = 1.0

        for (x, y, z) in self._trail_points:
            p = Point()
            p.x, p.y, p.z = x, y, z
            m.points.append(p)

        self._trail_pub.publish(m)

    # ── Poll log file and publish concept spheres ────────────────────────────
    def _poll_log(self):
        if not self.LOG_PATH.exists():
            return
        try:
            with open(self.LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            return

        landmarks   = log.get('landmarks', [])
        activations = log.get('concept_activations', [])   # per Phase-1 patch
        visits      = log.get('phase2_visits', [])

        if len(landmarks) <= self._known_landmarks:
            return   # nothing new

        self._publish_concept_markers(landmarks, activations, visits)
        self._known_landmarks = len(landmarks)

    # ── Build SPHERE_LIST / TEXT markers for each landmark ──────────────────
    def _publish_concept_markers(self, landmarks, activations, visits):
        marker_array = MarkerArray()

        # Build a lookup: visit position → dominant concept
        visit_concept = {}
        for v in visits:
            if not v.get('is_landmark'):
                continue
            key = (round(v['x'], 1), round(v['y'], 1))
            # Find matching activation (by Phase1 idx approximation — best effort)
            visit_concept[key] = v.get('dominant_concept', 0)

        for i, lm in enumerate(landmarks):
            x, y = lm['x'], lm['y']
            reward = lm['reward']

            # Dominant concept (0 if not logged yet)
            concept_idx = visit_concept.get(
                (round(x, 1), round(y, 1)), i % len(CONCEPT_COLORS))
            cr, cg, cb, ca = CONCEPT_COLORS[concept_idx % len(CONCEPT_COLORS)]

            # ── Sphere ──────────────────────────────────────────────────────
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'concept_spheres'
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = -15.0 + 1.0   # hover just above flight alt
            sphere.pose.orientation.w = 1.0

            # Size ∝ reward (clamped)
            size = float(min(max(reward / 25.0 * 3.0, 0.8), 3.5))
            sphere.scale.x = size
            sphere.scale.y = size
            sphere.scale.z = size
            sphere.color = rgba(cr, cg, cb, 0.80)
            marker_array.markers.append(sphere)

            # ── Text label ──────────────────────────────────────────────────
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = 'concept_labels'
            text.id = 1000 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = -15.0 + 3.5
            text.pose.orientation.w = 1.0
            text.scale.z = 1.2    # text height
            text.color = rgba(1.0, 1.0, 1.0, 0.9)
            text.text = f'L{i+1}\nC{concept_idx}\nR={reward:.1f}'
            marker_array.markers.append(text)

            # ── Vertical pole (line from ground to sphere) ───────────────────
            pole = Marker()
            pole.header.frame_id = 'map'
            pole.header.stamp = self.get_clock().now().to_msg()
            pole.ns = 'concept_poles'
            pole.id = 2000 + i
            pole.type = Marker.LINE_STRIP
            pole.action = Marker.ADD
            pole.pose.orientation.w = 1.0
            pole.scale.x = 0.12
            pole.color = rgba(cr, cg, cb, 0.40)
            ground = Point(); ground.x = x; ground.y = y; ground.z = 0.0
            top    = Point(); top.x    = x; top.y    = y; top.z    = -15.0 + 1.0
            pole.points = [ground, top]
            marker_array.markers.append(pole)

        self._concept_pub.publish(marker_array)
        self.get_logger().info(
            f'Published {len(landmarks)} concept markers')


def main(args=None):
    rclpy.init(args=args)
    node = MissionVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
