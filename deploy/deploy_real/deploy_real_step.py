from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

import rclpy as rp

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.step_command import StepCommand
from common.utils import (to_array, normalize, yaw_quat,
                        axis_angle_from_quat,
                        subtract_frame_transforms,
                        wrap_to_pi,
                        compute_pose_error
                        )
from config import Config

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped, StaticTransformBroadcaster

isaaclab_joint_order = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]

raw_joint_order = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

# Create a mapping tensor
# mapping_tensor = torch.zeros((len(sim_b_joints), len(sim_a_joints)), device=env.device)
mapping_tensor = torch.zeros((len(raw_joint_order), len(isaaclab_joint_order)))

# Fill the mapping tensor
for b_idx, b_joint in enumerate(raw_joint_order):
    if b_joint in isaaclab_joint_order:
        a_idx = isaaclab_joint_order.index(b_joint)
        mapping_tensor[a_idx, b_idx] = 1.0

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)

        self._step_command = None
        self._saved = False

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        # dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        # kps = self.config.kps + self.config.arm_waist_kps
        # kds = self.config.kds + self.config.arm_waist_kds
        # default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_idx = self.config.joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_pos = self.config.default_angles
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # for i in range(len(self.config.leg_joint2motor_idx)):
            #     motor_idx = self.config.leg_joint2motor_idx[i]
            #     self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
            #     self.low_cmd.motor_cmd[motor_idx].qd = 0
            #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            #     self.low_cmd.motor_cmd[motor_idx].tau = 0
            # for i in range(len(self.config.arm_waist_joint2motor_idx)):
            #     motor_idx = self.config.arm_waist_joint2motor_idx[i]
            #     self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            #     self.low_cmd.motor_cmd[motor_idx].qd = 0
            #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            #     self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.joint2motor_idx)):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def tf_to_pose(self, tf, order='xyzw'):
        pos = to_array(tf.transform.translation)
        quat = to_array(tf.transform.rotation)
        if order == 'wxyz':
            quat = np.roll(quat, 1, axis=-1)
        return np.concatenate((pos, quat), axis=0)

    def publish_step_command(self, next_ctarget_left, next_ctarget_right):
        left_tf = TransformStamped()
        left_tf.header.stamp = self._node.get_clock().now().to_msg()
        left_tf.header.frame_id = 'world'
        left_tf.child_frame_id = 'left_ctarget'
        left_tf.transform.translation.x = next_ctarget_left[0]
        left_tf.transform.translation.y = next_ctarget_left[1]
        left_tf.transform.translation.z = next_ctarget_left[2]

        left_tf.transform.rotation.x = next_ctarget_left[4]
        left_tf.transform.rotation.y = next_ctarget_left[5]
        left_tf.transform.rotation.z = next_ctarget_left[6]
        left_tf.transform.rotation.w = next_ctarget_left[3]

        right_tf = TransformStamped()
        right_tf.header.stamp = left_tf.header.stamp
        right_tf.header.frame_id = 'world'
        right_tf.child_frame_id = 'right_ctarget'
        right_tf.transform.translation.x = next_ctarget_right[0]
        right_tf.transform.translation.y = next_ctarget_right[1]
        right_tf.transform.translation.z = next_ctarget_right[2]

        right_tf.transform.rotation.x = next_ctarget_right[4]
        right_tf.transform.rotation.y = next_ctarget_right[5]
        right_tf.transform.rotation.z = next_ctarget_right[6]
        right_tf.transform.rotation.w = next_ctarget_right[3]

        self.tf_broadcaster.sendTransform(left_tf)
        self.tf_broadcaster.sendTransform(right_tf)

    def get_command(self, pelvis_w,
                        foot_left_b,
                        foot_right_b,
                        ctarget_left_w,
                        ctarget_right_w):
        ctarget_left_b_pos, ctarget_left_b_quat = subtract_frame_transforms(pelvis_w[:3],
                                                                            pelvis_w[3:7],
                                                                            ctarget_left_w[:3],
                                                                            ctarget_left_w[3:7])
        ctarget_right_b_pos, ctarget_right_b_quat = subtract_frame_transforms(pelvis_w[:3],
                                                                            pelvis_w[3:7],
                                                                            ctarget_right_w[:3],
                                                                            ctarget_right_w[3:7])
        pos_delta_left, axa_delta_left = compute_pose_error(foot_left_b[:3],
                                                            foot_left_b[3:7],
                                                            ctarget_left_b_pos,
                                                            ctarget_left_b_quat)
        pos_delta_right, axa_delta_right = compute_pose_error(foot_right_b[:3],
                                                            foot_right_b[3:7],
                                                            ctarget_right_b_pos,
                                                            ctarget_right_b_quat)
        return np.concatenate((pos_delta_left, axa_delta_left, pos_delta_right, axa_delta_right), axis=0)

    def run(self):
        if self._step_command is None:

            current_left_tf = self.tf_buffer.lookup_transform("world", 
                                    "left_foot", rclpy.time.Time())
            current_left_pose = self.tf_to_pose(current_left_tf, 'wxyz')
            current_left_pose[2] = 0.0
            current_left_pose[3:7] = yaw_quat(current_left_pose[3:7])
            current_right_tf = self.tf_buffer.lookup_transform("world",
                                    "right_foot", rclpy.time.Time())
            current_right_pose = self.tf_to_pose(current_right_tf, 'wxyz')
            current_right_pose[2] = 0.0
            current_right_pose[3:7] = yaw_quat(current_right_pose[3:7])
            self._step_command = StepCommand(current_left_pose, current_right_pose) 

        self.counter += 1
        next_ctarget = self._step_command.get_next_ctarget(
                                                    self.remote_controller,
                                                    self.counter * self.config.control_dt)
        next_ctarget_left, next_ctarget_right, dt_left, dt_right = next_ctarget
        self.publish_step_command(next_ctarget_left, next_ctarget_right)
        
        # Get the current joint position and velocity
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
        #     self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        for i, motor_idx in enumerate(self.config.joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq


        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            # waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            waist_yaw = self.low_state.motor_state[self.config.joint2motor_idx[12]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.joint2motor_idx[12]].dq

            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        
        # foot pose
        left_foot_from_base_tf = self.tf_buffer.lookup_transform("pelvis", 
                                                "left_ankle_roll_link",
                                                rclpy.time.Time())
        right_foot_from_base_tf = self.tf_buffer.lookup_transform("pelvis",
                                                "right_ankle_roll_link",
                                                rclpy.time.Time())
        lf_b = self.tf_to_pose(left_foot_from_base_tf, 'wxyz')
        rf_b = self.tf_to_pose(right_foot_from_base_tf, 'wxyz')
        left_foot_axa = wrap_to_pi(axis_angle_from_quat(lf_b[3:7]))
        right_foot_axa = wrap_to_pi(axis_angle_from_quat(rf_b[3:7]))
        rel_foot = np.concatenate((left_foot_from_base[:3],
                                    right_foot_from_base[:3], 
                                    left_foot_axa,
                                    right_foot_axa), axis=0)
        # hand pose
        left_hand_from_base_tf = self.tf_buffer.lookup_transform("pelvis",
                                                "left_rubber_hand",
                                                rclpy.time.Time())
        right_hand_from_base_tf = self.tf_buffer.lookup_transform("pelvis",
                                                "right_rubber_hand",
                                                rclpy.time.Time())
        left_hand_from_base = self.tf_to_pose(left_hand_from_base_tf, 'wxyz')
        right_hand_from_base = self.tf_to_pose(right_hand_from_base_tf, 'wxyz')
        left_hand_axa = wrap_to_pi(axis_angle_from_quat(left_hand_from_base[3:7]))
        right_hand_axa = wrap_to_pi(axis_angle_from_quat(right_hand_from_base[3:7]))
        rel_hand = np.concatenate((left_hand_from_base[:3],
                                    right_hand_from_base[:3],
                                    left_hand_axa,
                                    right_hand_axa), axis=0)
        # foot command
        base_pose_w = self.tf_to_pose(self.tf_buffer.lookup_transform("world", "pelvis",
                                        rclpy.time.Time()), 'wxyz')
        step_command = self.get_command(base_pose_w,
                        lf_b,
                        rf_b,
                        ctarget_left_w,
                        ctarget_right_w)
        step_command = np.concatenate((step_command, dt_left, dt_right), axis=0)

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[6:18] = rel_foot
        self.obs[18:30] = rel_hand
        self.obs[30 : 30 + num_actions] = qj_obs
        self.obs[30 + num_actions : 30 + num_actions * 2] = dqj_obs
        self.obs[30 + num_actions * 2 : 30 + num_actions * 3] = self.action
        self.obs[30 + num_actions * 3 : 30 + num_actions * 3 + 14] = step_command
        # self.obs[9 + num_actions * 3] = sin_phase
        # self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)

        # Reorder the observations
        obs_tensor[..., 30:30+num_actions] = obs_tensor[..., 30:30+num_actions] @ mapping_tensor.transpose(0, 1)
        obs_tensor[..., 30 + num_actions : 30 + num_actions * 2] = obs_tensor[..., 30 + num_actions : 30 + num_actions * 2] @ mapping_tensor.transpose(0, 1)
        obs_tensor[..., 30 + num_actions * 2 : 30 + num_actions * 3] = obs_tensor[..., 30 + num_actions * 2 : 30 + num_actions * 3] @ mapping_tensor.transpose(0, 1)

        if not self._saved:
            torch.save(obs_tensor, "obs.pt")
            self._saved = True

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        # Reorder the actions
        self.action = self.action @ mapping_tensor.detach().cpu().numpy()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     motor_idx = self.config.leg_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0

        # for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #     motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0
        if False:
            for i, motor_idx in enumerate(self.config.joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0


        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)
        
    def clear(self):
        self._node.destroy_node()
        rp.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    controller.clear()
    print("Exit")
