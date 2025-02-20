from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from common.crc import CRC

from enum import Enum

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

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5

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
mapping_tensor = torch.zeros((len(raw_joint_order), len(isaaclab_joint_order)))

# Fill the mapping tensor
for b_idx, b_joint in enumerate(raw_joint_order):
    if b_joint in isaaclab_joint_order:
        a_idx = isaaclab_joint_order.index(b_joint)
        # if 'shoulder' in b_joint or 'elbow' in b_joint or 'wrist' in b_joint:
        #     mapping_tensor[a_idx, b_idx] = 0.1
        # else:
        mapping_tensor[a_idx, b_idx] = 1.0

mask = torch.ones(len(isaaclab_joint_order))
for b_idx, b_joint in enumerate(isaaclab_joint_order):
    if 'shoulder' in b_joint or 'elbow' in b_joint or 'wrist' in b_joint:
        mask[b_idx] = 0

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

        self._cur_time = None

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type

            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG,
                                'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(LowStateHG,
                                'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            # self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            # self.lowcmd_publisher_.Init()

            # self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            # self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            raise ValueError(f"{config.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        # self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.mode = Mode.wait
        self._mode_change = True
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        self._terminate = False
        self._obs_buf = []
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            self._node.destroy_timer(self._timer)
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._node.destroy_node()
            rp.shutdown()
            torch.save(torch.cat(self._obs_buf, dim=0), "obs6.pt")
            print("Exit")

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        size = len(cmd.motor_cmd)
        # print(cmd.mode_machine)
        # for i in range(size):
        #     print(i, cmd.motor_cmd[i].q, 
        #         cmd.motor_cmd[i].dq,
        #         cmd.motor_cmd[i].kp,
        #         cmd.motor_cmd[i].kd,
        #         cmd.motor_cmd[i].tau)
        self.lowcmd_publisher_.publish(cmd)

    def wait_for_low_state(self):
        while self.low_state.crc == 0:
            print(self.low_state)
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.default_pos
        else:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

    def prepare_default_pos(self):
        # move time 2s
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.joint2motor_idx
        kps = self.config.kps 
        kds = self.config.kds 
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.asarray(self.config.default_angles)
        # np.concatenate((self.config.default_angles), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx
        
        # record the current pos
        self._init_dof_pos = np.zeros(self._dof_size,
                                dtype=np.float32)
        for i in range(self._dof_size):
            self._init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            for j in range(self._dof_size):
                motor_idx = self._dof_idx[j]
                target_pos = self._default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = (self._init_dof_pos[j] * 
                                                        (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.damping

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.joint2motor_idx)):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

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
        left_tf.transform.translation.x = float(next_ctarget_left[0])
        left_tf.transform.translation.y = float(next_ctarget_left[1])
        left_tf.transform.translation.z = float(next_ctarget_left[2])

        left_tf.transform.rotation.x = float(next_ctarget_left[4])
        left_tf.transform.rotation.y = float(next_ctarget_left[5])
        left_tf.transform.rotation.z = float(next_ctarget_left[6])
        left_tf.transform.rotation.w = float(next_ctarget_left[3])

        right_tf = TransformStamped()
        right_tf.header.stamp = left_tf.header.stamp
        right_tf.header.frame_id = 'world'
        right_tf.child_frame_id = 'right_ctarget'
        right_tf.transform.translation.x = float(next_ctarget_right[0])
        right_tf.transform.translation.y = float(next_ctarget_right[1])
        right_tf.transform.translation.z = float(next_ctarget_right[2])

        right_tf.transform.rotation.x = float(next_ctarget_right[4])
        right_tf.transform.rotation.y = float(next_ctarget_right[5])
        right_tf.transform.rotation.z = float(next_ctarget_right[6])
        right_tf.transform.rotation.w = float(next_ctarget_right[3])

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

    def run_policy(self):
        if self._step_command is None:

            current_left_tf = self.tf_buffer.lookup_transform( 
                                    "world",
                                    "left_ankle_roll_link", 
                                    rp.time.Time())
                                    # rp.duration.Duration(seconds=0.02))
            current_left_pose = self.tf_to_pose(current_left_tf, 'wxyz')
            current_left_pose[2] = 0.0
            current_left_pose[3:7] = yaw_quat(current_left_pose[3:7])
            current_right_tf = self.tf_buffer.lookup_transform(
                                    "world",
                                    "right_ankle_roll_link", 
                                    rp.time.Time())
                                    # rp.duration.Duration(seconds=0.02))
            current_right_pose = self.tf_to_pose(current_right_tf, 'wxyz')
            current_right_pose[2] = 0.0
            current_right_pose[3:7] = yaw_quat(current_right_pose[3:7])
            self._step_command = StepCommand(current_left_pose, current_right_pose) 
        
        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1
        # Get the current joint position and velocity
        next_ctarget = self._step_command.get_next_ctarget(
                                                    self.remote_controller,
                                                    self.counter * self.config.control_dt)
        print(next_ctarget)
        next_ctarget_left, next_ctarget_right, dt_left, dt_right = next_ctarget
        self.publish_step_command(next_ctarget_left, next_ctarget_right)
        
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
       
        # foot pose
        left_foot_from_base_tf = self.tf_buffer.lookup_transform( 
                                                "pelvis",
                                                "left_ankle_roll_link",
                                                rp.time.Time())
        right_foot_from_base_tf = self.tf_buffer.lookup_transform(
                                                "pelvis",
                                                "right_ankle_roll_link",
                                                rp.time.Time())

        lf_b = self.tf_to_pose(left_foot_from_base_tf, 'wxyz')
        rf_b = self.tf_to_pose(right_foot_from_base_tf, 'wxyz')
        left_foot_axa = wrap_to_pi(axis_angle_from_quat(lf_b[3:7]))
        right_foot_axa = wrap_to_pi(axis_angle_from_quat(rf_b[3:7]))
        rel_foot = np.concatenate((lf_b[:3],
                                    rf_b[:3], 
                                    left_foot_axa,
                                    right_foot_axa), axis=0)
        # hand pose
        left_hand_from_base_tf = self.tf_buffer.lookup_transform(
                                                "pelvis",
                                                "left_rubber_hand",
                                                rp.time.Time())
        right_hand_from_base_tf = self.tf_buffer.lookup_transform(
                                                "pelvis",
                                                "right_rubber_hand",
                                                rp.time.Time())
        lh_b = self.tf_to_pose(left_hand_from_base_tf, 'wxyz')
        rh_b = self.tf_to_pose(right_hand_from_base_tf, 'wxyz')
        left_hand_axa = wrap_to_pi(axis_angle_from_quat(lh_b[3:7]))
        right_hand_axa = wrap_to_pi(axis_angle_from_quat(rh_b[3:7]))
        rel_hand = np.concatenate((lh_b[:3],
                                    rh_b[:3],
                                    left_hand_axa,
                                    right_hand_axa), axis=0)

        # foot command
        base_pose_w = self.tf_to_pose(self.tf_buffer.lookup_transform(
            "world", "pelvis",
                                        rp.time.Time()), 'wxyz')
        dt_left = dt_right = 0.0
        step_command = self.get_command(base_pose_w,
                        lf_b,
                        rf_b,
                        next_ctarget_left,
                        next_ctarget_right)
        step_command = np.concatenate((step_command, 
                            np.asarray([dt_left, dt_right])), axis=0)

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:18] = rel_foot
        self.obs[18:30] = rel_hand
        self.obs[30 : 30 + num_actions] = qj_obs
        self.obs[30 + num_actions : 30 + num_actions * 2] = dqj_obs
        self.obs[30 + num_actions * 2 : 30 + num_actions * 3] = self.action
        self.obs[30 + num_actions * 3 : 30 + num_actions * 3 + 14] = step_command

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)        
        obs_tensor[..., 30:30+num_actions] = obs_tensor[..., 30:30+num_actions] @ mapping_tensor.transpose(0, 1)
        obs_tensor[..., 30 + num_actions : 30 + num_actions * 2] = obs_tensor[..., 30 + num_actions : 30 + num_actions * 2] @ mapping_tensor.transpose(0, 1)
        obs_tensor[..., 30 + num_actions * 2 : 30 + num_actions * 3] = obs_tensor[..., 30 + num_actions * 2 : 30 + num_actions * 3] @ mapping_tensor.transpose(0, 1)

        # if not self._saved:
        #     torch.save(obs_tensor, "obs.pt")
        #     self._saved = True
        self._obs_buf.append(obs_tensor.clone())

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        # self.action = self.action * mask.numpy()
        # Reorder the actions
        self.action = self.action @ mapping_tensor.detach().cpu().numpy()

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale *0.8

        # Build low cmd
        if True:
            for i, motor_idx in enumerate(self.config.joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
        # send the command
        self.send_cmd(self.low_cmd)

    def run_wrapper(self):
        # print("hello", self.mode,
        # self.mode == Mode.zero_torque)
        if self.mode == Mode.wait:
            if self.low_state.crc != 0:
                self.mode = Mode.zero_torque
                self.low_cmd.mode_machine = self.mode_machine_
                print("Successfully connected to the robot.")
        elif self.mode == Mode.zero_torque:
            if self._mode_change:
                print("Enter zero torque state.")
                print("Waiting for the start signal...")
                self._mode_change = False
            self.zero_torque_state()
        elif self.mode == Mode.default_pos:
            if self._mode_change:
                print("Moving to default pos.")
                self._mode_change = False
                self.prepare_default_pos()                
            self.move_to_default_pos()
        elif self.mode == Mode.damping:
            if self._mode_change:
                print("Enter default pos state.")
                print("Waiting for the Button A signal...")
                try:
                    current_left_tf = self.tf_buffer.lookup_transform( 
                                    "world",
                                    "left_ankle_roll_link", 
                                    rp.time.Time())
                    self._mode_change = False
                except Exception as ex:
                    print(ex)
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.null:
            self._terminate = True

        # time.sleep(self.config.control_dt)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)
