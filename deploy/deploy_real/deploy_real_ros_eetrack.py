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


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


class eetrack:
    def __init__(self, root_state_w, tf_buffer):
        self.tf_buffer = tf_buffer
        self.eetrack_midpt = root_state_w.clone()
        self.eetrack_midpt[..., 1] += 0.3
        self.eetrack_end = None
        self.eetrack_subgoal = None
        self.number_of_subgoals = 30
        self.eetrack_line_length = 0.3
        self.device = "cpu"
        self.create_eetrack()
        self.eetrack_subgoal = self.create_subgoal()
        self.sg_idx = 0
        # first subgoal sampling time = 1.0s
        self.init_time = rp.time.Time().nanoseconds / 1e9 + 1.0

    def create_eetrack(self):
        self.eetrack_start = self.eetrack_midpt.clone()
        self.eetrack_end = self.eetrack_midpt.clone()
        is_hor = rd.choice([True, False])
        eetrack_offset = rd.uniform(-0.5, 0.5)
        # For testing
        is_hor = True
        eetrack_offset = 0.0
        if is_hor:
            self.eetrack_start[..., 2] += eetrack_offset
            self.eetrack_end[..., 2] += eetrack_offset
            self.eetrack_start[..., 0] -= (self.eetrack_line_length) / 2.
            self.eetrack_end[..., 0] += (self.eetrack_line_length) / 2.
        else:
            self.eetrack_start[..., 0] += eetrack_offset
            self.eetrack_end[..., 0] += eetrack_offset
            self.eetrack_start[..., 2] += (self.eetrack_line_length) / 2.
            self.eetrack_end[..., 2] -= (self.eetrack_line_length) / 2.
        return self.eetrack_start, self.eetrack_end

    def create_direction(self):
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * np.pi - np.pi / 2
        # For testing
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi / 2
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * 0
        roll = torch.zeros(1, device=self.device)
        pitch = angle_from_xyplane_in_global_frame
        yaw = angle_from_eetrack_line
        euler = torch.stack([roll, pitch, yaw], dim=1)
        quat = math_utils.quat_from_euler_xyz(
            euler[:, 0], euler[:, 1], euler[:, 2])
        return quat

    def create_subgoal(self):
        eetrack_subgoals = interpolate_position(
            self.eetrack_start, self.eetrack_end, self.number_of_subgoals)
        eetrack_subgoals = [
            (
                l.clone().to(self.device, dtype=torch.float32)
                if isinstance(l, torch.Tensor)
                else torch.tensor(l, device=self.device, dtype=torch.float32)
            )
            for l in eetrack_subgoals
        ]
        eetrack_subgoals = torch.stack(eetrack_subgoals, axis=1)
        eetrack_ori = self.create_direction().unsqueeze(
            1).repeat(1, self.number_of_subgoals + 1, 1)
        # welidng_subgoals -> Nenv x Npoints x (3 + 4)
        return torch.cat([eetrack_subgoals, eetrack_ori], dim=2)

    def update_command(self):
        time = self.init_time - rp.time.Time().nanoseconds / 1e9
        if (time >= 0):
            self.sg_idx = int(time / 0.1 + 1)
        # self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        self.sg_idx = min(self.sg_idx, self.number_of_subgoals + 1)
        self.next_command_s_left = self.eetrack_subgoal[...,
                                                        self.sg_idx, :]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose(
            self.tf_buffer,
            "left_hand_palm_link",
            rot_type='quat'
        )

        lerp_command_w_left = self.next_command_s_left

        (lerp_command_b_left_pos,
         lerp_command_b_left_quat) = math_utils.subtract_frame_transforms(
            root_state_w[..., 0:3],
            root_state_w[..., 3:7],
            lerp_command_w_left[:, 0:3],
            lerp_command_w_left[:, 3:7],
        )

        # lerp_command_b_left = lerp_command_w_left

        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            torch.from_numpy(pos_hand_b_left)[None],
            torch.from_numpy(quat_hand_b_left)[None],
            lerp_command_b_left_pos,
            lerp_command_b_left_quat,
        )
        axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)

        hand_command = torch.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)
        return hand_command


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
        # observation terms (order preserved)

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

        fp_l = body_pose(self.tf_buffer, 'left_ankle_roll_link')
        fp_r = body_pose(self.tf_buffer, 'right_ankle_roll_link')
        foot_pose = np.concatenate([fp_l[0], fp_r[0], fp_l[1], fp_r[1]])

        hp_l = body_pose(self.tf_buffer, 'left_hand_palm_link')
        hp_r = body_pose(self.tf_buffer, 'right_hand_palm_link')
        hand_pose = np.concatenate([hp_l[0], hp_r[0], hp_l[1], hp_r[1]])

        # FIXME(ycho): implement com_pos_wrt_pelvis
        projected_com = compute_com(self.tf_buffer, self.links)[..., :2]
        # projected_zmp = _ # IMPOSSIBLE

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

        # Given as delta_pos {xyz,axa}; i.e. 6D vector
        # hands_command = self.eetrack.get_command()

        right_arm_com = compute_com(self.tf_buffer, [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_pitch_link"
            "right_wrist_roll_link",
            "right_wrist_yaw_link"
        ])
        left_arm_com = compute_com(self.tf_buffer, [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_pitch_link"
            "left_wrist_roll_link",
            "left_wrist_yaw_link"
        ])
        world_from_pelvis = self.tf_buffer.lookup_transform(
            'world',
            'pelvis',
            rp.time.Time()
        )
        pelvis_height = [world_from_pelvis.transform.translation.z]

        obs = [
            base_ang_vel,
            projected_gravity,
            foot_pose,
            hand_pose,
            projected_com,
            joint_pos,
            0.0 * joint_vel,
            actions,
            hands_command,
            right_arm_com,
            left_arm_com,
            pelvis_height
        ]
        # print([np.shape(o) for o in obs])
        return np.concatenate(obs, axis=-1)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
            config.arm_joint)
        self.actmap = ActToDof(config, self.ikctrl)
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit

        # == build index map ==
        arm_joint = config.arm_joint
        self.mot_from_pin = index_map(
            self.config.motor_joint,
            self.ikctrl.joint_names)
        self.pin_from_mot = index_map(
            self.ikctrl.joint_names,
            self.config.motor_joint
        )
        self.mot_from_arm = index_map(
            self.config.motor_joint,
            self.config.arm_joint
        )
        self.mot_from_nonarm = index_map(
            self.config.motor_joint,
            self.config.non_arm_joint
        )
        self.lab_from_mot = index_map(self.config.lab_joint,
                                      self.config.motor_joint)
        self.config.default_angles = np.asarray(self.config.lab_joint_offsets)[
            self.lab_from_mot
        ]

        # Data buffers
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.obsmap = Observation(
            '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
            config, self.tf_buffer)
        # FIXME(ycho): give `root_state_w`
        self.eetrack = None

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
        # self.mode = Mode.wait
        # if running from rosbag:
        self.mode = Mode.policy

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

        # TODO(ycho): consider using `cmd` for `hands_command`
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        if self.target_pose is None:
            # NOTE(ycho): ensure target pose is defined in world frame!
            world_from_pelvis = body_pose(
                self.tf_buffer,
                'pelvis',
                'world',
                rot_type='quat'
            )
            xyz0, quat_wxyz0 = world_from_pelvis
            xyz1, quat_wxyz1 = self.default_pose_b[:3], self.default_pose_b[3:7]
            xyz, quat = combine_frame_transforms(
                xyz0, quat_wxyz0,
                xyz1, quat_wxyz1)
            self.target_pose = np.concatenate([xyz, quat])
            print('validation...',
                  self.target_pose,
                  body_pose(self.tf_buffer,
                            'left_hand_palm_link',
                            'world', rot_type='quat'))

        if True:
            self.target_pose[..., :3] += 0.01 * self.cmd

        # FIXME(ycho): implement `_hands_command_`
        # to use the output of `eetrack`.
        if True:
            # NOTE(ycho): requires running `fake_world_tf_pub.py`.
            world_from_pelvis = body_pose(
                self.tf_buffer,
                'pelvis',
                'world',
                rot_type='quat'
            )
            xyz, quat_wxyz = world_from_pelvis
            root_state_w = np.zeros(7)
            root_state_w[0:3] = xyz
            root_state_w[3:7] = quat_wxyz

        if self.eetrack is None:
            self.eetrack = eetrack(torch.from_numpy(root_state_w)[None],
                                   self.tf_buffer)

        if False:
            _hands_command_ = self.eetrack.get_command(
                torch.from_numpy(root_state_w)[None])[0].detach().cpu().numpy()
        else:
            _hands_command_ = np.zeros(6)

            # TODO(ycho): restore updating to `hands_command`
            # from `target_pose`.
            if False:
                q_mot = [self.low_state.motor_state[i_mot].q
                         for i_mot in range(29)]
                # print('q_mot (out)', q_mot)
                q_pin = np.zeros_like(self.ikctrl.cfg.q)
                q_pin[self.pin_from_mot] = q_mot

                current_pose = self.ikctrl.fk(q_pin)
                _hands_command_ = np.zeros(6)
                _hands_command_[0:3] = (self.target_pose[:3]
                                        - current_pose.translation)

                quat_wxyz = xyzw2wxyz(pin.Quaternion(
                    current_pose.rotation).coeffs())
                # q_target @ q_current^{-1}
                d_quat = quat_mul(
                    self.target_pose[3:7],
                    quat_conjugate(quat_wxyz)
                )
                d_axa = axis_angle_from_quat(d_quat)
                _hands_command_[3:6] = d_axa
                # bprint('hands_command', _hands_command_)

        # print(_hands_command_)
        # _hands_command_ = np.zeros(6)
        # _hands_command_[0] = self.cmd[0] * 0.03
        # _hands_command_[2] = self.cmd[1] * 0.03

        self.obs[:] = self.obsmap(self.low_state,
                                  self.action,
                                  _hands_command_)
        # logpath = Path('/tmp/eet3/')
        # logpath.mkdir(parents=True, exist_ok=True)
        # np.save(F'{logpath}/obs{self.counter:03d}.npy',
        #         self.obs)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone()

        # hands_command = obs[..., 119:125]
        if False:
            world_from_body_quat = math_utils.quat_from_euler_xyz(
                th.as_tensor([0], dtype=th.float32),
                th.as_tensor([0], dtype=th.float32),
                th.as_tensor([1.57], dtype=th.float32)).reshape(4)
            obs_tensor[..., 119:122] = math_utils.quat_rotate(
                world_from_body_quat[None],
                obs_tensor[..., 119:122])
            obs_tensor[..., 122:125] = math_utils.quat_rotate(
                world_from_body_quat[None],
                obs_tensor[..., 122:125])
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        # np.save(F'{logpath}/act{self.counter:03d}.npy',
        #         self.action)

        target_dof_pos = self.actmap(
            self.obs,
            self.action,
            root_state_w[3:7]
        )
        # print('??',
        #         target_dof_pos,
        #         [self.low_state.motor_state[i_mot].q for i_mot in range(29)])

        # np.save(F'{logpath}/dof{self.counter:03d}.npy',
        #         target_dof_pos)

        # Build low cmd
        for i in range(len(self.config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.05 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 0.05 * float(self.config.kds[i])
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

        # time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)
