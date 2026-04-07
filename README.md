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
