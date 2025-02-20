import rclpy
from rclpy.node import Node

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped, StaticTransformBroadcaster

class Tester(Node):
    def __init__(self):
        super().__init__('tester')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.01, self.on_timer)

    def on_timer(self):
        try:
            self.tf_buffer.lookup_transform(
                "world", "camera_init", rclpy.time.Time()
            )
        except Exception as ex:
            print(f'Could not transform mid360_link_IMU to pelvis as world to camera_init is yet published: {ex}')
            return
        try:
            current_left_tf = self.tf_buffer.lookup_transform( 
                                    "world",
                                    "left_ankle_roll_link", 
                                    rclpy.time.Time(),
                                    rclpy.duration.Duration(seconds=0.02))
            current_right_tf = self.tf_buffer.lookup_transform(
                                    "world",
                                    "right_ankle_roll_link", 
                                    rclpy.time.Time(),
                                    rclpy.duration.Duration(seconds=0.02))
            print(current_left_tf, current_right_tf)
        except Exception as ex:
            print(f'Could not transform mid360_link_IMU to pelvis: {ex}')

def main():
    rclpy.init()
    node = Tester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()