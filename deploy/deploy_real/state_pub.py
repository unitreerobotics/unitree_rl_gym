from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
# from tf2_ros import TransformBroadcaster, TransformStamped
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
import yaml

class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher')
        qos_profile = QoSProfile(depth=10)

        with open('./configs/ik.yaml', 'r') as fp:
            self.joint_names = yaml.safe_load(fp)['motor_joint']

        self.low_state = LowStateHG()
        self.low_state_subscriber = self.create_subscription(LowStateHG,
                    'lowstate', self.on_low_state, 10)
        self.joint_pub = self.create_publisher(JointState,
                                               'joint_states', qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))
        self.joint_state = JointState()

    def on_low_state(self, msg: LowStateHG):
        self.low_state = msg
        joint_state = self.joint_state
        now = self.get_clock().now()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = self.joint_names
        joint_state.position = [0.0 for _ in self.joint_names]
        joint_state.velocity = [0.0 for _ in self.joint_names]

        n:int = min(len(self.joint_names), len(self.low_state.motor_state))
        for i in range(n):
            joint_state.position[i] = self.low_state.motor_state[i].q
            joint_state.velocity[i] = self.low_state.motor_state[i].dq
        self.joint_pub.publish(joint_state)
    
    def run(self):
        loop_rate = self.create_rate(30)
        try:
            # rclpy.spin()
            while rclpy.ok():
                rclpy.spin_once(self)
                loop_rate.sleep()
        except KeyboardInterrupt:
            pass



def main():
    rclpy.init()
    node = StatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
