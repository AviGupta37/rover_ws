ALL launch command:

SPHERE TRAKER NODE :

TERMINAL 1: rover ackermann gz  

TERMINAL 2: ROS2 TO GZ BRIDGE FOR CAMERA 

TERMINAL 3: MICROXRCE AGENT 

TERMINAL 4: SPHERE TRACKER NODE WITH PX COMMAND 

CAMERA:

TERMINAL 1 :ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true

TERMINAL 2: ros2 run mission_planner object_pointcloud

TERMINAL 3: ros2 topic list #To check topic camera topic pulish or not 
# ================= PX4 + MAVROS + OFFBOARD SETUP =================

# Source ROS2
source /opt/ros/humble/setup.bash

# (Optional) your workspace
if [ -f ~/rover_ws/install/setup.bash ]; then
    source ~/rover_ws/install/setup.bash
fi

# Alias to launch MAVROS with serial + UDP for QGC
alias start_mavros='ros2 launch mavros mavros.launch.py \
fcu_url:=serial:///dev/ttyACM0:115200 \
gcs_url:=udp://0.0.0.0:14550'

# Arm vehicle
alias arm_px4='ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"'

# Set OFFBOARD mode
alias offboard_px4='ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: OFFBOARD}"'

# Publish velocity continuously (forward motion)
alias forward_px4='ros2 topic pub -r 10 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}, angular: {z: 0.0}}"'

# Stop vehicle
alias stop_px4='ros2 topic pub -r 10 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"'

# FULL AUTO START FUNCTION
start_rover() {
    echo "🚀 Starting MAVROS..."
    start_mavros &

    sleep 5

    echo "🛰 Waiting for FCU connection..."
    ros2 topic echo /mavros/state --once

    echo "📡 Sending initial setpoints..."
    ros2 topic pub -r 10 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" &
    
    sleep 3

    echo "🔓 Arming PX4..."
    arm_px4

    sleep 2

    echo "🎯 Switching to OFFBOARD..."
    offboard_px4

    echo "✅ READY — Use forward_px4 / stop_px4"
}
