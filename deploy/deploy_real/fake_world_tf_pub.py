#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from unitree_hg.msg import LowState as LowStateHG
from tf2_ros import TransformBroadcaster, TransformStamped

import numpy as np
import pinocchio as pin
import pink
from common.np_math import (index_map, with_dir)
from math_utils import (as_np, quat_rotate)

quat_rotate = as_np(quat_rotate)


class FakeWorldPublisher(Node):
    def __init__(self):
        super().__init__('fake_world_publisher')

        urdf_path = '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                   package_dirs=["."],
                                                   root_joint=None)
            self.robot = robot

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(
            LowStateHG,
            'lowstate',
            self.on_low_state,
            10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def on_low_state(self,
                     msg: LowStateHG):
        self.low_state = msg

        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'pelvis'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.pelvis_height(self.low_state)

        # Set world_from_pelvis quaternion based on IMU state
        # TODO(ycho): consider applying 90-deg offset?
        qw, qx, qy, qz = self.low_state.imu_state.quaternion
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def pelvis_height(self, low_state: LowStateHG):
        robot = self.robot
        q_mot = [self.low_state.motor_state[i_mot].q for i_mot in range(29)]
        q_pin = np.zeros_like(self.robot.q0)
        q_pin[self.pin_from_mot] = q_mot
        cfg = pink.Configuration(robot.model, robot.data, q_pin)

        pelvis_from_rf = cfg.get_transform_frame_to_world(
            'right_ankle_roll_link')
        pelvis_from_lf = cfg.get_transform_frame_to_world(
            'left_ankle_roll_link')

        # 0.02 = "roll_link" height (approx)
        world_from_pelvis_quat = low_state.imu_state.quaternion
        pelvis_z_rf = -quat_rotate(
            world_from_pelvis_quat,
            pelvis_from_rf.translation)[2] + 0.02
        pelvis_z_lf = -quat_rotate(
            world_from_pelvis_quat,
            pelvis_from_lf.translation)[2] + 0.02
        return 0.5 * pelvis_z_lf + 0.5 * pelvis_z_rf


def main():
    rclpy.init()
    node = FakeWorldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()