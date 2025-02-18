#!/usr/bin/env python3

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from unitree_hg.msg import LowState as LowStateHG
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped

import numpy as np
import yaml


class PelvistoTrack(Node):
    def __init__(self):
        super().__init__('pelvis_track_publisher')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.05, self.on_timer)
        
    def on_timer(self):
        try:
            t = TransformStamped()

            # Read message content and assign it to
            # corresponding tf variables

            t_lidar_pelvis = self.tf_buffer.lookup_transform(
                'mid360_link_IMU', 
                # 'zed2_camera_center',
                'pelvis', rclpy.time.Time()
            )

            t.header.stamp = self.get_clock().now().to_msg()
            # t.header.frame_id = 'zed_camera_center'
            t.header.frame_id = 'body'
            t.child_frame_id = 'pelvis'

            # Turtle only exists in 2D, thus we get x and y translation
            # coordinates from the message and set the z coordinate to 0
            t.transform.translation.x = t_lidar_pelvis.transform.translation.x
            t.transform.translation.y = t_lidar_pelvis.transform.translation.y
            t.transform.translation.z = t_lidar_pelvis.transform.translation.z

            t.transform.rotation.x = t_lidar_pelvis.transform.rotation.x
            t.transform.rotation.y = t_lidar_pelvis.transform.rotation.y
            t.transform.rotation.z = t_lidar_pelvis.transform.rotation.z
            t.transform.rotation.w = t_lidar_pelvis.transform.rotation.w

            # Send the transformation
            self.tf_broadcaster.sendTransform(t)
        except Exception as ex:
            print(f'Could not transform mid360_link_IMU to pelvis: {ex}')


def main():
    rclpy.init()
    node = PelvistoTrack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()