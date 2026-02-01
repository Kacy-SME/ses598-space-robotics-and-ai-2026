# Custom message for detailed controller performance metrics

# Header with timestamp
std_msgs/Header header

# Cross-track error metrics
float64 cross_track_error
float64 avg_cross_track_error
float64 max_cross_track_error

# Velocity information
float64 linear_velocity
float64 angular_velocity

# Navigation progress
float64 distance_to_waypoint
int32 current_waypoint
int32 total_waypoints
float64 completion_percentage

# Additional metrics
float64 path_deviation
float64 controller_effort
