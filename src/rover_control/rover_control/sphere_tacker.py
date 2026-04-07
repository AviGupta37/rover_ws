#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from px4_msgs.msg import (
    OffboardControlMode,
    RoverThrottleSetpoint,
    RoverSteeringSetpoint,
    VehicleCommand
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class SphereTracker(Node):

    def __init__(self):
        super().__init__('sphere_tracker_px4')

        self.bridge = CvBridge()
        self.depth_image = None

        # PX4 publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.throttle_pub = self.create_publisher(RoverThrottleSetpoint, '/fmu/in/rover_throttle_setpoint', 10)
        self.steer_pub = self.create_publisher(RoverSteeringSetpoint, '/fmu/in/rover_steering_setpoint', 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # Camera subscribers
        self.rgb_sub = self.create_subscription(Image, '/rover/rgb_camera', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/rover/depth_camera', self.depth_callback, 10)

        self.timer = self.create_timer(0.05, self.control_loop)

        self.cx = None
        self.depth = None
        self.prev_error = 0.0
        self.counter = 0

        self.get_logger().info("🔥 FINAL PX4 SPHERE TRACKER RUNNING")

    # ---------------- DEPTH ----------------
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ---------------- DETECTION ----------------
    def detect_sphere(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 2)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=80,
            param1=50, param2=25,
            minRadius=20, maxRadius=300
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            return max(circles[0], key=lambda c: c[2])

        return None

    # ---------------- RGB ----------------
    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w, _ = frame.shape

        result = self.detect_sphere(frame)

        if result is not None and self.depth_image is not None:
            cx, cy, r = result

            dh, dw = self.depth_image.shape

            cx_d = int(cx * (dw - 1) / w)
            cy_d = int(cy * (dh - 1) / h)

            cx_d = np.clip(cx_d, 0, dw - 1)
            cy_d = np.clip(cy_d, 0, dh - 1)

            depth = self.depth_image[cy_d, cx_d]

            if not np.isnan(depth) and depth > 0:
                self.cx = cx
                self.depth = float(depth)

                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.putText(frame, f"{self.depth:.2f} m",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

        cv2.imshow("Sphere Tracking", frame)
        cv2.waitKey(1)

    # ---------------- ZERO COMMAND ----------------
    def publish_zero_commands(self):
        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        throttle_msg = RoverThrottleSetpoint()
        throttle_msg.timestamp = timestamp
        throttle_msg.throttle_body_x = 0.0

        steer_msg = RoverSteeringSetpoint()
        steer_msg.timestamp = timestamp
        steer_msg.normalized_steering_setpoint = 0.0

        self.throttle_pub.publish(throttle_msg)
        self.steer_pub.publish(steer_msg)

    # ---------------- CONTROL ----------------
    def control_loop(self):

        # OFFBOARD heartbeat
        offboard = OffboardControlMode()
        offboard.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        offboard.position = False
        offboard.velocity = False
        offboard.acceleration = False
        offboard.attitude = False
        offboard.body_rate = True
        offboard.direct_actuator = True

        self.offboard_pub.publish(offboard)

        # 🔥 SEND INITIAL SETPOINTS (VERY IMPORTANT)
        if self.counter < 20:
            self.publish_zero_commands()
            self.counter += 1
            return

        # Arm + Offboard
        if self.counter == 20:
            self.arm()
            self.set_offboard_mode()

        self.counter += 1

        throttle = 0.0
        steering = 0.0

        if self.cx is not None:

            w = 640
            error = (self.cx - w/2) / (w/2)

            # smoothing
            error = 0.6 * self.prev_error + 0.4 * error
            self.prev_error = error

            # steering
            steering = 0.5 * error
            steering = np.clip(steering, -1.0, 1.0)

            if abs(error) < 0.05:
                steering = 0.0

            # speed control
            if self.depth is not None:
                if self.depth > 2.0:
                    base_speed = 0.6
                elif self.depth > 1.2:
                    base_speed = 0.4
                else:
                    base_speed = 0.0
            else:
                base_speed = 0.0

            throttle = base_speed * (1 - 1.5 * abs(error))

            # 🔥 IMPORTANT FIX (deadzone)
            throttle = max(0.3, throttle)

            # stop near sphere
            if self.depth is not None and self.depth <= 1.0:
                throttle = 0.0
                steering = 0.0

        # publish
        timestamp = int(self.get_clock().now().nanoseconds / 1000)

        throttle_msg = RoverThrottleSetpoint()
        throttle_msg.timestamp = timestamp
        throttle_msg.throttle_body_x = float(throttle)

        steer_msg = RoverSteeringSetpoint()
        steer_msg.timestamp = timestamp
        steer_msg.normalized_steering_setpoint = float(steering)

        self.throttle_pub.publish(throttle_msg)
        self.steer_pub.publish(steer_msg)

        self.get_logger().info(
            f"err={self.prev_error:.2f}, throttle={throttle:.2f}, steer={steering:.2f}, depth={self.depth}"
        )

    # ---------------- ARM ----------------
    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    # ---------------- OFFBOARD ----------------
    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SphereTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
