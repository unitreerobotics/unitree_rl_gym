from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union, List
import numpy as np
import time
import torch
import torch as th
from pathlib import Path

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from common.crc import CRC
from enum import Enum
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz
from yourdfpy import URDF

import math_utils
import random as rd
from act_to_dof import ActToDof
from common.utils import (to_array, normalize, yaw_quat,
                        axis_angle_from_quat,
                        subtract_frame_transforms,
                        wrap_to_pi,
                        compute_pose_error,
                        quat_apply
                        )

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5


def index_map(k_to, k_from):
    """
    Returns an index mapping from k_from to k_to.

    Given k_to=a, k_from=b,
    returns an index map "a_from_b" such that
    array_a[a_from_b] = array_b

    Missing values are set to -1.
    """
    index_dict = {k: i for i, k in enumerate(k_to)}  # O(len(k_from))
    return [index_dict.get(k, -1) for k in k_from]  # O(len(k_to))


class Observation:
    def __init__(self,
                 urdf_path: str,
                 config,
                 tf_buffer: Buffer):
        self.links = list(URDF.load(urdf_path).link_map.keys())
        self.config = config
        self.num_lab_joint = len(config.lab_joint)
        self.tf_buffer = tf_buffer
        self.lab_from_mot = index_map(config.lab_joint,
                                      config.motor_joint)

    def __call__(self,
                 low_state: LowStateHG,
                 last_action: np.ndarray,
                 hands_command: np.ndarray
                 ):
        lab_from_mot = self.lab_from_mot
        num_lab_joint = self.num_lab_joint

        # FIXME(ycho): dummy value
        # base_lin_vel = np.zeros(3)
        ang_vel = np.array([low_state.imu_state.gyroscope],
                           dtype=np.float32)

        if True:
            # NOTE(ycho): requires running `fake_world_tf_pub.py`.
            world_from_pelvis = self.tf_buffer.lookup_transform(
                'world',
                'pelvis',
                rp.time.Time()
            )
            rxn = world_from_pelvis.transform.rotation
            quat = np.array([rxn.w, rxn.x, rxn.y, rxn.z])
        else:
            quat = low_state.imu_state.quaternion

        if self.config.imu_type == "torso":
            waist_yaw = low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel)

        # NOTE(ycho): `ang_vel` is _probably_ in the pelvis frame,
        # since otherwise transform_imu_data() would be unnecessary for
        # `ang_vel`.
        base_ang_vel = ang_vel.squeeze(0)

        # TODO(ycho): check if the convention "q_base^{-1} @ g" holds.
        projected_gravity = get_gravity_orientation(quat)

        # Map `low_state` to index-mapped joint_{pos,vel}
        joint_pos = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_vel = np.zeros(num_lab_joint,
                             dtype=np.float32)
        joint_pos[lab_from_mot] = [low_state.motor_state[i_mot].q for i_mot in
                                   range(len(lab_from_mot))]
        joint_pos -= config.lab_joint_offsets
        joint_vel[lab_from_mot] = [low_state.motor_state[i_mot].dq for i_mot in
                                   range(len(lab_from_mot))]
        actions = last_action

        obs = [
            base_ang_vel,
            projected_gravity,
            hands_command,
            joint_pos,
            joint_vel,
            actions,
        ]
        # print([np.shape(o) for o in obs])
        return np.concatenate(obs, axis=-1)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        # init mapping tensor for joint order.
        self.mapping_tensor = torch.zeros((len(config.lab_joint), len(config.motor_joint)))
        for b_idx, b_joint in enumerate(config.motor_joint):
            if b_joint in config.lab_joint:
                a_idx = config.lab_joint.index(b_joint)
                self.mapping_tensor[a_idx, b_idx] = 1.0

        self.mot_from_lab = index_map(config.motor_joint, config.lab_joint)
        
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # Data buffers
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # command : x[m] y[m] z[m] heading[rad]
        self.cmd = np.array([0., 0., 0., 0.]) 
        self.given_cmd = np.array([0., 0., 0., 0.])
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)
        self.obsmap = Observation(
            '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
            config, self.tf_buffer)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type

            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG,
                                                                 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            

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

        # NOTE(ycho):
        # if running from real robot:
        self.mode = Mode.wait
        # if running from rosbag:
        # self.mode = Mode.policy

        self._mode_change = True
        self._timer = self._node.create_timer(
            self.config.control_dt, self.run_wrapper)
        self._terminate = False
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

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx

        # record the current pos
        # self._init_dof_pos = np.zeros(self._dof_size,
        #                               dtype=np.float32)
        # for i in range(self._dof_size):
        #     self._init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        self._init_dof_pos = np.zeros(29)
        for i in range(29):
            self._init_dof_pos[i] = self.low_state.motor_state[i].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            # for j in range(self._dof_size):
            for j in range(29):
                # motor_idx = self._dof_idx[j]
                # target_pos = self._default_pos[j]
                motor_idx = j
                target_pos = self.config.default_angles[j]

                self.low_cmd.motor_cmd[motor_idx].q = (
                    self._init_dof_pos[j] * (1 - alpha) + target_pos * alpha)
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
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(
                    self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(
                    self.config.arm_waist_target[i])
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
    
    def update_command(self):
        """update command """
        # Get controller input, update the target navigation command from the world coordinate
        pass
    
    def get_command(self, base_pose_w):
        self.update_command()
        # Get the target navigation command, which is converted to the body coordinate system
        x_w, y_w, _, heading_w = self.given_cmd

        # Compute xyz navigation goal from body coordinate system
        """
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(
            yaw_quat(self.robot.data.root_quat_w), target_vec
        )
        """
        xyz_w , quat_w = base_pose_w[:3], base_pose_w[3:]
        xyz_cmd_w = np.array(x_w - xyz_w[0], y_w - xyz_w[1], 0)
        xyz_cmd_b = math_utils.quat_rotate_inverse(
                    math_utils.yaw_quat(torch.as_tensor(quat_w).float().to("cpu")),
                    torch.as_tensor(xyz_cmd_w).float().to("cpu")
                ).float().numpy()
        
        # Compute heading direction from body coordinate system
        """
        forward_w = math_utils.quat_apply(self.root_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w
        )
        """
        robot_forward_vec_w = quat_apply(quat_w, torch.tensor([1,0,0]).float().to("cpu"))
        robot_heading_w = torch.atan2(robot_forward_vec_w[:, 1], robot_forward_vec_w[:, 0])
        heading_command_b = wrap_to_pi(heading_w - robot_heading_w)
        return np.append(xyz_cmd_b, heading_command_b)

    def run_policy(self):
        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1

        base_pose_w = self.tf_to_pose(self.tf_buffer.lookup_transform(
            "world", "pelvis",
            rp.time.Time()), 'wxyz')
        
        self.cmd = self.get_command(base_pose_w)

        self.obs[:] = self.obsmap(self.low_state,
                                  self.action,
                                  self.cmd)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone()

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        action = np.zeros_like(self.action,
                          dtype=np.float32)
        action[self.mot_from_lab] = self.action # index_map works ?

        target_dof_pos = self.config.default_angles + action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 1.0 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 1.0 * float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

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
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.null:
            self._terminate = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1_nav.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)
