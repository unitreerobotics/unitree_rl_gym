from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union, List
import numpy as np
import time
import torch
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


from math_utils import *
import random as rd

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 -
    # theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / angle,
        0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles[..., None]

def quat_from_angle_axis(
        angle: torch.Tensor,
        axis: torch.Tensor=None) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    if axis is None:
        axa   = angle
        angle = torch.linalg.norm(axa, dim=-1)
        axis  = axa / angle[..., None].clamp_min(1e-6)
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)



def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = q_vec * np.einsum("...i,...i->...", q_vec, v)[..., None] * 2.0
    return a + b + c


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = q_vec * np.einsum("...i,...i->...", q_vec, v)[..., None] * 2.0
    return a - b + c


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
        axa = (axa + np.pi) % (2 * np.pi)
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
    com_data = extract_link_data('../../resources/robots/g1_description/g1_29dof_rev_1_0.xml')

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
    returns an index mapping from k_from to k_to;
    i.e. k_to[index_map] = k_from
    Missing values are set to -1.
    """
    out = []
    for k in k_to:
        try:
            i = k_from.index(k)
        except ValueError:
            i = -1
        out.append(i)

    return out


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos

class eetrack:
    def __init__(self, root_state_w):
        self.eetrack_midpt = root_state_w.clone()
        self.eetrack_midpt[1] += 0.3
        self.eetrack_end = None
        self.eetrack_subgoal = None
        self.number_of_subgoals = 30
        self.eetrack_line_length = 0.3
        self.device = "cuda"
        self.create_eetrack()
        self.eetrack_subgoal = self.create_subgoal()
        self.sg_idx = 0
        self.init_time = rp.time.Time() + 1.0 # first subgoal sampling time = 1.0s

    def create_eetrack(self):
        self.eetrack_start = self.eetrack_midpt.clone() 
        self.eetrack_end = self.eetrack_midpt.clone() 
        is_hor = rd.choice([True, False])
        eetrack_offset = rd.uniform(-0.5, 0.5)
        # For testing
        is_hor = True
        eetrack_offset = 0.0
        if is_hor:
            self.eetrack_start[2] += eetrack_offset
            self.eetrack_end[2] += eetrack_offset
            self.eetrack_start[0] -= (self.eetrack_line_length) / 2.
            self.eetrack_end[0] += (self.eetrack_line_length) / 2.
        else:
            self.eetrack_start[0] += eetrack_offset
            self.eetrack_end[0] += eetrack_offset
            self.eetrack_start[2] += (self.eetrack_line_length) / 2.
            self.eetrack_end[2] -= (self.eetrack_line_length) / 2.
        return self.eetrack_start, self.eetrack_end

    def create_direction(self):
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi
        angle_from_xyplane_in_global_frame  = torch.rand(1, device=self.device) * np.pi - np.pi/2
        # For testing
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi/2
        angle_from_xyplane_in_global_frame  = torch.rand(1, device=self.device) * 0
        roll = torch.zeros(1, device=self.device)
        pitch = angle_from_xyplane_in_global_frame 
        yaw = angle_from_eetrack_line
        euler = torch.stack([roll, pitch, yaw], dim=1)
        quat = math_utils.quat_from_euler_xyz(euler[:,0], euler[:,1], euler[:,2])
        return quat
        
    def create_subgoal(self):
        eetrack_subgoals = interpolate_position(self.eetrack_start, self.eetrack_end, self.number_of_subgoals)
        eetrack_subgoals = [
            (
                l.clone().to(self.device, dtype=torch.float32)
                if isinstance(l, torch.Tensor) 
                else torch.tensor(l, device=self.device, dtype=torch.float32)
            )
            for l in eetrack_subgoals
        ]
        eetrack_subgoals = torch.stack(eetrack_subgoals,axis=1)
        eetrack_ori = self.create_direction().unsqueeze(1).repeat(1, self.number_of_subgoals + 1, 1)
        # welidng_subgoals -> Nenv x Npoints x (3 + 4)
        return torch.cat([eetrack_subgoals, eetrack_ori], dim=2)

    def update_command(self):
        time = self.init_time - rp.time.Time()
        if (time>=0):
            self.sg_idx = time / 0.1 + 1
        self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        self.next_command_s_left = self.eetrack_subgoal[self.sg_idx]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose_axa("left_hand_palm_link")

        lerp_command_w_left = self.next_command_s_left

        # lerp_command_b_left = math_utils.subtract_frame_transforms(
        #     root_state_w[..., 0:3],
        #     root_state_w[..., 3:7],
        #     lerp_command_w_left[:, 0:3],
        #     lerp_command_w_left[:, 3:7],
        # )

        lerp_command_b_left = lerp_command_w_left

        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            pos_hand_b_left,
            quat_hand_b_left,
            lerp_command_b_left[:, :3],
            lerp_command_b_left[:, 3:],
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
        quat = low_state.imu_state.quaternion
        if self.config.imu_type == "torso":
            waist_yaw = low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel)
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
        if True:  # hack
            lf_from_pelvis = self.tf_buffer.lookup_transform(
                'left_ankle_roll_link',  # to
                'pelvis',
                rp.time.Time()
            )
            rf_from_pelvis = self.tf_buffer.lookup_transform(
                'right_ankle_roll_link',  # to
                'pelvis',
                rp.time.Time()
            )
            # NOTE(ycho): we assume at least one of the feet is on the ground
            #            and use the higher of the two as the pelvis height.
            pelvis_height = max(lf_from_pelvis.transform.translation.z,
                                rf_from_pelvis.transform.translation.z)
            pelvis_height = [pelvis_height]
        else:
            pelvis_height = np.abs(np.dot(
                projected_gravity,  # world frame
                fp_l[0]
            )
            )
        obs = [
            base_ang_vel,
            projected_gravity,
            foot_pose,
            hand_pose,
            projected_com,
            joint_pos,
            joint_vel,
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
        # self.eetrack = eetrack(root_state_w=None) 

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

        self.mode = Mode.wait
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
        # self.cmd[0] = self.remote_controller.ly
        # self.cmd[1] = self.remote_controller.lx * -1
        # self.cmd[2] = self.remote_controller.rx * -1

        # FIXME(ycho): implement `_hands_command_`
        # to use the output of `eetrack`.
        _hands_command_ = np.zeros(6)

        self.obs[:] = self.obsmap(self.low_state,
                                  self.action,
                                  _hands_command_)
        Path('/tmp/eet/').mkdir(parents=True,
                exist_ok=True)
        np.save(F'/tmp/eet/obs{self.counter:03d}.npy',
                self.obs)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        np.save(F'/tmp/eet/act{self.counter:03d}.npy',
                self.action)

        non_arm_joint_pos = self.action[..., :22]
        left_arm_residual = self.action[..., 22:29]

        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        for i_mot in range(len(self.config.motor_joint)):
            i_pin = self.pin_from_mot[i_mot]
            q_pin[i_pin] = self.low_state.motor_state[i_mot].q

        
        d_quat = quat_from_angle_axis(
                torch.from_numpy(_hands_command_[..., 3:])
        ).detach().cpu().numpy()

        source_pose = self.ikctrl.fk(q_pin)
        source_xyz = source_pose.translation
        source_quat = xyzw2wxyz(pin.Quaternion(source_pose.rotation).coeffs())

        target_xyz = source_xyz + _hands_command_[..., :3]
        target_quat = quat_mul(torch.from_numpy(d_quat),
                torch.from_numpy(source_quat)).detach().cpu().numpy()
        target = np.concatenate([target_xyz, target_quat])

        res_q_ik = self.ikctrl(
            q_pin,
            target
        )
        print('res_q_ik', res_q_ik)

        target_dof_pos = np.zeros(29)
        for i_arm in range(len(res_q_ik)):
            i_mot = self.mot_from_arm[i_arm]
            i_pin = self.pin_from_mot[i_mot]
            target_q = (
                self.low_state.motor_state[i_mot].q
                + res_q_ik[i_arm]
                + np.clip(0.3 * left_arm_residual[i_arm],
                          -0.2, 0.2)
            )
            target_q = np.clip(target_q,
                               self.lim_lo_pin[i_pin],
                               self.lim_hi_pin[i_pin])
            target_dof_pos[i_mot] = target_q
        np.save(F'/tmp/eet/dof{self.counter:03d}.npy',
                target_dof_pos)

        # Build low cmd
        for i in range(len(self.config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0 * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 0.0 * float(self.config.kds[i])
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
