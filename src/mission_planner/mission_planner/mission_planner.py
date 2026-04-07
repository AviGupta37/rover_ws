#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        # Publishers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10)

        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10)

        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10)

        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.counter = 0
        self.current_point = 0

        # Square path (NED frame)
        self.square = [
            [0.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [5.0, 5.0, -5.0],
            [0.0, 5.0, -5.0],
            [0.0, 0.0, -5.0]
        ]


    def timer_callback(self):

        timestamp = self.get_clock().now().nanoseconds // 1000

        # Send Offboard control mode
        offboard = OffboardControlMode()
        offboard.position = True
        offboard.velocity = False
        offboard.acceleration = False
        offboard.attitude = False
        offboard.body_rate = False
        offboard.timestamp = timestamp

        self.offboard_pub.publish(offboard)

        # Send trajectory setpoint
        traj = TrajectorySetpoint()
        traj.position = self.square[self.current_point]
        traj.yaw = 0.0
        traj.timestamp = timestamp

        self.traj_pub.publish(traj)

        # Switch to Offboard after some setpoints
        if self.counter == 10:
            self.arm()
            self.set_offboard_mode()

        # Change waypoint every ~5 seconds
        if self.counter % 50 == 0 and self.counter > 20:
            self.current_point = (self.current_point + 1) % len(self.square)
            self.get_logger().info(f"Moving to {self.square[self.current_point]}")

        self.counter += 1


    def arm(self):

        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1

        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        self.cmd_pub.publish(msg)


    def set_offboard_mode(self):

        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE

        msg.param1 = 1.0
        msg.param2 = 6.0

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1

        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        self.cmd_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = OffboardControl()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
