#!/usr/bin/env python3
from first_order_boustrophedon_navigator.msg import ControllerMetrics
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np
import math
from collections import deque
from std_msgs.msg import Float64
from rcl_interfaces.msg import SetParametersResult
import matplotlib.pyplot as plt

class BoustrophedonController(Node):
    def __init__(self):
        super().__init__('lawnmower_controller')
        
        # Declare turtle_name parameter FIRST
        self.declare_parameter('turtle_name', 'turtle1')
        self.turtle_name = self.get_parameter('turtle_name').value
        
        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('Kp_linear', 1.0),
                ('Kd_linear', 0.1),
                ('Kp_angular', 1.0),
                ('Kd_angular', 0.1),
                ('spacing', 0.5)
            ]
        )

        # Get initial parameter values
        self.Kp_linear = self.get_parameter('Kp_linear').value
        self.Kd_linear = self.get_parameter('Kd_linear').value
        self.Kp_angular = self.get_parameter('Kp_angular').value
        self.Kd_angular = self.get_parameter('Kd_angular').value
        self.spacing = self.get_parameter('spacing').value
        
        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Create publisher and subscriber FOR THIS TURTLE
        self.velocity_publisher = self.create_publisher(
            Twist, 
            f'/{self.turtle_name}/cmd_vel', 
            10
        )
        self.pose_subscriber = self.create_subscription(
            Pose, 
            f'/{self.turtle_name}/pose', 
            self.pose_callback, 
            10
        )
        
        # Lawnmower pattern parameters
        self.waypoints = self.generate_waypoints()
        self.current_waypoint = 0
        
        # Cross-track error calculation
        self.cross_track_errors = deque(maxlen=1000)  # Store last 1000 errors
        
        # Data for plots
        self.trajectory = []  # To store x, y positions
        self.velocities = []  # To store linear and angular velocities
        
        # State variables
        self.pose = Pose()
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0
        self.prev_time = self.get_clock().now()
        
        # Create control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Add publisher for cross-track error FOR THIS TURTLE
        self.metrics_pub = self.create_publisher(
            ControllerMetrics, 
            f'/{self.turtle_name}/controller_metrics', 
            10
        )
        
        self.get_logger().info(f'Lawnmower controller started for {self.turtle_name}')
        self.get_logger().info(f'PD Params: Kp_lin={self.Kp_linear:.3f}, Kd_lin={self.Kd_linear:.3f}, '
                              f'Kp_ang={self.Kp_angular:.3f}, Kd_ang={self.Kd_angular:.3f}')
        self.get_logger().info(f'Following waypoints: {self.waypoints}')

    def generate_waypoints(self):
        waypoints = []
        y = 8.0  # Start higher in the window
        
        while y >= 3.0:
            if len(waypoints) % 2 == 0:
                waypoints.append((2.0, y))
                waypoints.append((9.0, y))
            else:
                waypoints.append((9.0, y))
                waypoints.append((2.0, y))
            y -= self.spacing
        
        return waypoints

    def calculate_cross_track_error(self):
        if self.current_waypoint < 1:
            return 0.0

        start = np.array(self.waypoints[self.current_waypoint - 1])
        end = np.array(self.waypoints[self.current_waypoint])
        pos = np.array([self.pose.x, self.pose.y])

        path_vector = end - start
        path_length = np.linalg.norm(path_vector)
        if path_length < 1e-6:
            return np.linalg.norm(pos - start)

        path_unit = path_vector / path_length
        pos_vector = pos - start

        projection_length = np.dot(pos_vector, path_unit)
        projection_length = max(0, min(path_length, projection_length))
        projected_point = start + projection_length * path_unit

        error_vector = pos - projected_point
        error_sign = np.sign(np.cross(path_unit, error_vector / np.linalg.norm(error_vector)))
        error = np.linalg.norm(error_vector) * error_sign

        self.cross_track_errors.append(abs(error))

        return error

    def pose_callback(self, msg):
        self.pose = msg

    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)

    def control_loop(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info(f'{self.turtle_name}: Lawnmower pattern complete')
            if self.cross_track_errors:
                final_avg_error = sum(self.cross_track_errors) / len(self.cross_track_errors)
                final_max_error = max(self.cross_track_errors)
                self.get_logger().info(f'{self.turtle_name}: Average cross-track error: {final_avg_error:.4f}')
                self.get_logger().info(f'{self.turtle_name}: Maximum cross-track error: {final_max_error:.4f}')
            self.timer.cancel()
            self.plot_data()
            return

        cross_track_error = self.calculate_cross_track_error()

        target_x, target_y = self.waypoints[self.current_waypoint]
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9

        distance = self.get_distance(self.pose.x, self.pose.y, target_x, target_y)
        target_angle = self.get_angle(self.pose.x, self.pose.y, target_x, target_y)
        angular_error = target_angle - self.pose.theta

        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi

        linear_error_derivative = (distance - self.prev_linear_error) / dt
        angular_error_derivative = (angular_error - self.prev_angular_error) / dt

        linear_velocity = self.Kp_linear * distance + self.Kd_linear * linear_error_derivative
        angular_velocity = self.Kp_angular * angular_error + self.Kd_angular * angular_error_derivative

        vel_msg = Twist()
        vel_msg.linear.x = min(linear_velocity, 2.0)
        vel_msg.angular.z = angular_velocity
        self.velocity_publisher.publish(vel_msg)

        self.trajectory.append((self.pose.x, self.pose.y))
        self.velocities.append((linear_velocity, angular_velocity))

        self.prev_linear_error = distance
        self.prev_angular_error = angular_error
        self.prev_time = current_time

        if distance < 0.1:
            self.current_waypoint += 1
            self.get_logger().info(f'{self.turtle_name}: Reached waypoint {self.current_waypoint}')
        # Create and publish custom metrics message
        metrics_msg = ControllerMetrics()
        metrics_msg.header = Header()
        metrics_msg.header.stamp = self.get_clock().now().to_msg()
        metrics_msg.header.frame_id = self.turtle_name
        
        # Cross-track error metrics
        metrics_msg.cross_track_error = abs(cross_track_error)
        avg_error = (sum(self.cross_track_errors) / len(self.cross_track_errors) 
                     if self.cross_track_errors else 0.0)
        max_error = max(self.cross_track_errors) if self.cross_track_errors else 0.0
        metrics_msg.avg_cross_track_error = avg_error
        metrics_msg.max_cross_track_error = max_error
        
        # Velocity information
        metrics_msg.linear_velocity = linear_velocity
        metrics_msg.angular_velocity = angular_velocity
        
        # Navigation progress
        metrics_msg.distance_to_waypoint = distance
        metrics_msg.current_waypoint = self.current_waypoint
        metrics_msg.total_waypoints = len(self.waypoints)
        completion_pct = (self.current_waypoint / len(self.waypoints)) * 100.0
        metrics_msg.completion_percentage = completion_pct
        
        # Additional metrics
        metrics_msg.path_deviation = abs(cross_track_error)
        metrics_msg.controller_effort = abs(linear_velocity) + abs(angular_velocity)
        
        self.metrics_pub.publish(metrics_msg)
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'Kp_linear':
                self.Kp_linear = param.value
            elif param.name == 'Kd_linear':
                self.Kd_linear = param.value
            elif param.name == 'Kp_angular':
                self.Kp_angular = param.value
            elif param.name == 'Kd_angular':
                self.Kd_angular = param.value
            elif param.name == 'spacing':
                self.spacing = param.value
                self.waypoints = self.generate_waypoints()
        return SetParametersResult(successful=True)

    def plot_data(self):
        trajectory = np.array(self.trajectory)
        velocities = np.array(self.velocities)

        # Plot Cross-Track Error
        plt.figure()
        plt.plot(self.cross_track_errors)
        plt.title(f"Cross-Track Error Over Time - {self.turtle_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.savefig(f"cross_track_error_{self.turtle_name}.png")

        # Plot Trajectory
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
        plt.scatter([wp[0] for wp in self.waypoints], [wp[1] for wp in self.waypoints], c='red', label="Waypoints")
        plt.title(f"Trajectory Plot - {self.turtle_name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig(f"trajectory_{self.turtle_name}.png")

        # Plot Velocity Profiles
        plt.figure()
        plt.plot(velocities[:, 0], label="Linear Velocity")
        plt.plot(velocities[:, 1], label="Angular Velocity")
        plt.title(f"Velocity Profiles - {self.turtle_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.legend()
        plt.savefig(f"velocity_profiles_{self.turtle_name}.png")

        self.get_logger().info(f"Plots saved for {self.turtle_name}")

def main(args=None):
    rclpy.init(args=args)
    controller = BoustrophedonController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
