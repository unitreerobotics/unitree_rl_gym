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


class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)

# Create a mapping tensor
# mapping_tensor = torch.zeros((len(sim_b_joints), len(sim_a_joints)), device=env.device)


def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rp.time.Time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        raise

    txn = t.transform.translation
    rxn = t.transform.rotation

    xyz = np.array([txn.x, txn.y, txn.z])
    quat_wxyz = np.array([rxn.w, rxn.x, rxn.y, rxn.z])

    xyz = np.array(xyz)
    if rot_type == 'axa':
        axa = axis_angle_from_quat(quat_wxyz)
        axa = wrap_to_pi(axa)
        return (xyz, axa)
    elif rot_type == 'quat':
        return (xyz, quat_wxyz)
    raise ValueError(f"Unknown rot_type: {rot_type}")


from common.xml_helper import extract_link_data


def compute_com(tf_buffer, body_frames: List[str]):
    """compute com of body frames"""
    mass_list = []
    com_list = []

    # bring default values
    com_data = extract_link_data(
        '../../resources/robots/g1_description/g1_29dof_rev_1_0.xml')

    # iterate for frames
    for frame in body_frames:
        try:
            frame_data = com_data[frame]
        except KeyError:
            continue

        try:
            link_pos, link_wxyz = body_pose(tf_buffer,
                                            frame, rot_type='quat')
        except TransformException:
            continue

        com_pos_b, com_wxyz = frame_data['pos'], frame_data['quat']

        # compute com from world coordinates
        # NOTE 'math_utils' package will be brought from isaaclab
        com_pos = link_pos + quat_rotate(link_wxyz, com_pos_b)
        com_list.append(com_pos)

        # get math
        mass = frame_data['mass']
        mass_list.append(mass)

    com = sum([m * pos for m, pos in zip(mass_list, com_list)]) / sum(mass_list)
    return com


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
        
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # Data buffers
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # command : x[m] y[m] z[m] heading[rad]
        self.cmd = np.array([0., 0., 0., 0.]) 
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
        # FIXME(ycho): give `root_state_w`

        if True:
            q_mot = np.array(config.default_angles)
            q_pin = np.zeros_like(self.ikctrl.cfg.q)
            q_pin[self.pin_from_mot] = q_mot
            default_pose = self.ikctrl.fk(q_pin)
            xyz = default_pose.translation
            quat_wxyz = xyzw2wxyz(
                pin.Quaternion(
                    default_pose.rotation).coeffs())
            self.default_pose_b = np.concatenate([xyz, quat_wxyz])
            # self.target_pose = np.copy(self.default_pose_b)
            self.target_pose = None
            # print('default_pose', self.default_pose_b)

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

    def run_policy(self):
        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1

        # self.cmd[0] = x
        # self.cmd[1] = y
        # self.cmd[2] = z
        # self.cmd[3] = heading

        self.obs[:] = self.obsmap(self.low_state,
                                  self.action,
                                  self.cmd)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone()

        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        self.action = self.action @ self.mapping_tensor.detach().cpu().numpy()
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 1.0 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 1.0 * float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        # reduce KP for non-arm joints
        # for i in self.mot_from_nonarm:
        #     self.low_cmd.motor_cmd[i].kp = 0.05 * float(self.config.kps[i])
        #     self.low_cmd.motor_cmd[i].kd = 0.05 * float(self.config.kds[i])

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
