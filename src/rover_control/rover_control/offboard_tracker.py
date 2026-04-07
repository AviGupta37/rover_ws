#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class SphereTrackerPX4SITL(Node):
    """
    Sphere Tracker for PX4 SITL (Simulation In The Loop)
    - PX4 autopilot is the BRAIN
    - This node is just VISION & CONTROL
    - Subscribes to /fmu/out/* and publishes to /fmu/in/*
    """

    def __init__(self):
        super().__init__('sphere_tracker_px4')

        self.bridge = CvBridge()
        self.depth_image = None
        self.rgb_frame = None

        # ========== PX4 Publishers ==========
        # Send commands to PX4 autopilot
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # ========== PX4 Subscribers ==========
        # Get status from PX4 autopilot
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, 10)

        # ========== Camera Subscribers ==========
        # Get vision data (RGB + Depth)
        self.rgb_sub = self.create_subscription(Image, '/rover/rgb_camera', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/rover/depth_camera', self.depth_callback, 10)

        # ========== Control Loop Timer ==========
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        # ========== Sphere Detection State ==========
        self.cx = None
        self.cy = None
        self.depth = None
        self.radius = None

        # PID smoothing
        self.prev_error = 0.0
        self.control_counter = 0

        # Target color for sphere (HSV ranges)
        # Blue sphere by default
        self.lower_hsv = np.array([100, 100, 100])
        self.upper_hsv = np.array([130, 255, 255])

        # ========== PX4 State ==========
        self.vehicle_armed = False
        self.offboard_mode_active = False
        self.px4_ready = False

        self.get_logger().info("=" * 60)
        self.get_logger().info("🚁 PX4 SITL Sphere Tracker Started!")
        self.get_logger().info("PX4 is the BRAIN - This node handles VISION & CONTROL")
        self.get_logger().info("=" * 60)

    # ============ VEHICLE STATUS CALLBACK ============
    def vehicle_status_callback(self, msg):
        """Monitor PX4 autopilot status"""
        self.vehicle_armed = msg.arming_state == VehicleStatus.ARM_STATE_ARMED
        self.offboard_mode_active = msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        
        if not self.px4_ready:
            self.get_logger().info(f"✓ PX4 Connected (armed={self.vehicle_armed}, offboard={self.offboard_mode_active})")
            self.px4_ready = True

    # ============ DEPTH CALLBACK ============
    def depth_callback(self, msg):
        """Store latest depth image from camera"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth conversion error: {e}")

    # ============ DETECT SPHERE (HSV METHOD) ============
    def detect_sphere_hsv(self, frame):
        """
        Detect sphere using HSV color segmentation
        Returns: (cx, cy, radius, mask) or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for sphere color
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold
        if area < 500:
            return None
        
        # Fit circle to contour
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Circularity check (sphere should be circular)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < 0.5:  # Too elongated
                return None
        
        return (int(cx), int(cy), int(radius), mask)

    # ============ RGB CALLBACK ============
    def rgb_callback(self, msg):
        """Process RGB frame and detect sphere"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.rgb_frame = frame.copy()
            h, w, _ = frame.shape

            # Detect sphere using HSV
            result = self.detect_sphere_hsv(frame)

            if result is not None:
                cx, cy, r, mask = result
                
                # Get depth at sphere center
                if self.depth_image is not None:
                    dh, dw = self.depth_image.shape

                    # Scale sphere coordinates to depth image resolution
                    cx_d = int(cx * (dw - 1) / w)
                    cy_d = int(cy * (dh - 1) / h)

                    cx_d = np.clip(cx_d, 0, dw - 1)
                    cy_d = np.clip(cy_d, 0, dh - 1)

                    # Get depth at sphere center
                    depth = self.depth_image[cy_d, cx_d]

                    if not np.isnan(depth) and depth > 0:
                        self.cx = cx
                        self.cy = cy
                        self.radius = r
                        self.depth = float(depth)

                # Draw detection on frame
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)  # Green circle
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red center
                cv2.line(frame, (320, 0), (320, 480), (255, 0, 0), 1)  # Center line

                if self.depth is not None:
                    cv2.putText(frame, f"Distance: {self.depth:.2f}m",
                                (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
            else:
                self.cx = None
                self.cy = None
                self.radius = None

            # Display
            cv2.imshow("Sphere Tracking - RGB", frame)
            if result is not None and result[3] is not None:
                cv2.imshow("Sphere Tracking - Mask", result[3])
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    # ============ ARM VEHICLE ============
    def arm(self):
        """Send ARM command to PX4"""
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0  # 1=arm, 0=disarm
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)
        self.get_logger().info("📡 ARM command sent to PX4")

    # ============ SET OFFBOARD MODE ============
    def set_offboard_mode(self):
        """Switch PX4 to OFFBOARD mode (user control)"""
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0  # Main mode
        msg.param2 = 6.0  # OFFBOARD mode
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)
        self.get_logger().info("📡 OFFBOARD mode command sent to PX4")

    # ============ CONTROL LOOP ============
    def control_loop(self):
        """
        Main control loop - generates commands for PX4 autopilot.
        PX4 is the BRAIN, this node sends setpoints.
        """

        # ========== OFFBOARD HEARTBEAT (REQUIRED) ==========
        # Must send this every 50ms or PX4 will kill control
        offboard = OffboardControlMode()
        offboard.velocity = True  # We'll send velocity setpoints
        offboard.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(offboard)

        # ========== ARM & SWITCH TO OFFBOARD (first time only) ==========
        if self.control_counter == 10:  # ~500ms after startup
            self.arm()
            time.sleep(1.0)  # Wait for arm to complete
            self.set_offboard_mode()
            self.get_logger().info("✅ PX4 Armed and in OFFBOARD mode - Ready for control!")

        self.control_counter += 1

        # ========== VELOCITY SETPOINT ==========
        vx = 0.0   # Forward velocity (m/s)
        vy = 0.0   # Lateral velocity (m/s)
        vz = 0.0   # Vertical velocity (m/s) - keep at 0 for ground rover
        wz = 0.0   # Yaw rate (rad/s)

        # ========== VISION-BASED CONTROL ==========
        if self.cx is not None and self.depth is not None and self.px4_ready:

            image_center_x = 320.0  # Image width = 640, center = 320

            # ---- ERROR CALCULATION (CENTERING) ----
            pixel_error = self.cx - image_center_x
            norm_error = pixel_error / image_center_x
            
            # Smoothing with exponential moving average
            alpha = 0.3
            norm_error = alpha * norm_error + (1 - alpha) * self.prev_error
            self.prev_error = norm_error

            # ---- STOPPING CONDITION ----
            # Stop when sphere is very close (1.0m)
            if self.depth <= 1.0:
                vx = 0.0
                wz = 0.0
                self.get_logger().info("🎯 SPHERE REACHED! Hovering in place...")
            else:
                # ---- SPEED CONTROL (based on distance) ----
                if self.depth > 5.0:
                    base_speed = 3.0  # Fast approach for far objects
                elif self.depth > 3.0:
                    base_speed = 2.0  # Medium speed
                elif self.depth > 2.0:
                    base_speed = 1.2  # Slower
                elif self.depth > 1.0:
                    base_speed = 0.6  # Very slow final approach
                else:
                    base_speed = 0.0

                # Reduce forward speed while turning hard
                turn_factor = max(0.3, 1.0 - abs(norm_error))
                vx = base_speed * turn_factor

                # ---- STEERING (angular velocity / yaw rate) ----
                kp = 1.0  # Proportional gain for steering
                wz = kp * norm_error  # Positive = turn right, negative = turn left

        # ========== SEND TRAJECTORY SETPOINT TO PX4 ==========
        traj = TrajectorySetpoint()
        
        # Position setpoint (NaN = no position control, use velocity only)
        traj.position[0] = float('nan')
        traj.position[1] = float('nan')
        traj.position[2] = float('nan')
        
        # Velocity setpoint (m/s)
        # For ground rover: only vx (forward) and wz (yaw rate) matter
        traj.velocity[0] = vx    # Forward velocity
        traj.velocity[1] = vy    # Lateral velocity (unused for rover)
        traj.velocity[2] = vz    # Vertical velocity (unused for rover, keep at 0)
        
        # Yaw setpoint (NaN = no yaw control, use yaw rate only)
        traj.yaw = float('nan')
        
        # Yaw rate (rad/s) - this controls turning
        traj.yawspeed = wz
        
        traj.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(traj)

        # ========== LOGGING ==========
        if self.cx is not None and self.control_counter % 10 == 0:  # Log every 500ms
            self.get_logger().info(
                f"🔍 Sphere: cx={self.cx}px, depth={self.depth:.2f}m | "
                f"📤 Command: vx={vx:.2f}m/s, wz={wz:.2f}rad/s | "
                f"🚁 Armed={self.vehicle_armed}, Offboard={self.offboard_mode_active}"
            )

    # ============ COLOR CALIBRATION ============
    def set_sphere_color(self, lower_hsv, upper_hsv):
        """Adjust HSV range for your sphere color"""
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.get_logger().info(f"HSV range updated: {lower_hsv} to {upper_hsv}")


def main(args=None):
    rclpy.init(args=args)
    node = SphereTrackerPX4SITL()
    
    # Optional: Calibrate for your sphere color
    # node.set_sphere_color([20, 100, 100], [30, 255, 255])  # Yellow sphere
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
