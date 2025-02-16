#!/usr/bin/env python3

import numpy as np
import pinocchio as pin

from common.np_math import (xyzw2wxyz, index_map)
from math_utils import (
    as_np,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse
)
quat_from_angle_axis = as_np(quat_from_angle_axis)
quat_mul = as_np(quat_mul)
quat_rotate_inverse = as_np(quat_rotate_inverse)

from config import Config
from ikctrl import IKCtrl


class ActToDof:
    def __init__(self,
                 config: Config,
                 ikctrl: IKCtrl):
        self.config = config
        self.ikctrl = ikctrl
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit

        self.mot_from_pin = index_map(
            self.config.motor_joint,
            self.ikctrl.joint_names)
        self.pin_from_mot = index_map(
            self.ikctrl.joint_names,
            self.config.motor_joint
        )
        self.pin_from_lab = index_map(
            self.ikctrl.joint_names,
            self.config.lab_joint
        )
        self.pin_from_arm = index_map(
            self.ikctrl.joint_names,
            self.config.arm_joint
        )
        self.mot_from_arm = index_map(
            self.config.motor_joint,
            self.config.arm_joint
        )
        self.mot_from_lab = index_map(
            self.config.motor_joint,
            self.config.lab_joint
        )
        self.lab_from_mot = index_map(
            self.config.lab_joint,
            self.config.motor_joint
        )
        self.mot_from_nonarm = index_map(
            self.config.motor_joint,
            self.config.non_arm_joint
        )

        self.lab_from_nonarm = index_map(
            self.config.lab_joint,
            self.config.non_arm_joint
        )
        # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 22, 24, 26, 28]

        self.default_nonarm = (
            np.asarray(self.config.lab_joint_offsets)[self.lab_from_nonarm]
        )

    def __call__(self, obs, action, root_quat_wxyz=None):
        hands_command_w = obs[..., 119:125]
        non_arm_joint_pos = action[..., :22]
        left_arm_residual = action[..., 22:29]

        q_lab = obs[..., 32:61]
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_lab] = q_lab
        q_pin[self.pin_from_lab] += np.asarray(self.config.lab_joint_offsets)

        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q_lab
        q_mot[self.mot_from_lab] += np.asarray(self.config.lab_joint_offsets)
        # print('q_mot (inside)', q_mot)
        # print('q0', q_mot[self.mot_from_arm])
        # q_mot[i_mot] = q_lab[ lab_from_mot[i_mot] ]

        if root_quat_wxyz is None:
            hands_command_b = hands_command_w
        else:
            world_from_pelvis_quat = root_quat_wxyz.astype(np.float32)
            hands_command_w = hands_command_w.astype(np.float32)
            hands_command_b = np.concatenate([
                quat_rotate_inverse(world_from_pelvis_quat,
                                    hands_command_w[..., :3]),
                quat_rotate_inverse(world_from_pelvis_quat,
                                    hands_command_w[..., 3:6])
            ], axis=-1)

        axa = hands_command_b[..., 3:]
        angle = np.asarray(np.linalg.norm(axa, axis=-1))
        axis = axa / np.maximum(angle, 1e-6)
        d_quat = quat_from_angle_axis(angle, axis)

        source_pose = self.ikctrl.fk(q_pin)
        source_xyz = source_pose.translation
        source_quat = xyzw2wxyz(pin.Quaternion(source_pose.rotation).coeffs())

        target_xyz = source_xyz + hands_command_b[..., :3]
        target_quat = quat_mul(d_quat, source_quat)
        target = np.concatenate([target_xyz, target_quat])
        res_q_ik = self.ikctrl(
            q_pin,
            target
        )
        # print('res_q_ik', res_q_ik)

        # q_pin2 = np.copy(q_pin)
        # q_pin2[self.pin_from_arm] += res_q_ik
        # print('target', target)
        # se3=self.ikctrl.fk(q_pin2)
        # print('fk(IK(target))', se3.translation,
        #         xyzw2wxyz(pin.Quaternion(se3.rotation).coeffs()))

        # print('res_q_ik', res_q_ik)
        # print('left_arm_residual',
        #         0.3 * left_arm_residual,
        #         np.clip(0.3 * left_arm_residual, -0.2, 0.2))

        target_dof_pos = np.zeros(29)
        target_dof_pos += q_mot
        target_dof_pos[self.mot_from_arm] += res_q_ik

        if True:
            target_dof_pos[self.mot_from_arm] += np.clip(
                0.3 * left_arm_residual,
                -0.2, 0.2)

        if True:
            # print('default joint pos', self.default_nonarm)
            # print('joint order', self.config.non_arm_joint)
            # print('mot_from_nonarm', self.mot_from_nonarm)
            target_dof_pos[self.mot_from_nonarm] = (
                self.default_nonarm + 0.5 * non_arm_joint_pos
            )

            target_dof_pos = np.clip(
                target_dof_pos,
                self.lim_lo_pin[self.pin_from_mot],
                self.lim_hi_pin[self.pin_from_mot]
            )

        return target_dof_pos


def main():
    from matplotlib import pyplot as plt
    import yaml

    with open('configs/g1_eetrack.yaml', 'r') as fp:
        d = yaml.safe_load(fp)
    config = Config('configs/g1_eetrack.yaml')
    ikctrl = IKCtrl(
        '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
        config.arm_joint)
    act_to_dof = ActToDof(config, ikctrl)

    exps = []
    cals = []
    poss = []
    # nonarm_offset = [-0.2000, -0.2000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    #       0.0000,  0.4200,  0.4200, -0.2000, -0.2300, -0.2300, -0.3500,  0.0000,
    #       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000]
    # nonarm_joint_names=['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    # print( np.asarray(config.lab_joint_offsets)[act_to_dof.lab_from_nonarm] - nonarm_offset)
    # print( list(config.non_arm_joint) == list(nonarm_joint_names))

    mot_from_arm = index_map(d['motor_joint'], d['arm_joint'])
    for i in range(61):
        obs = np.load(F'/gate/eet6/obs{i:03d}.npy')[0]
        # print(obs.shape)
        act = np.load(F'/gate/eet6/act{i:03d}.npy')[0]
        # print('act', act.shape)
        dof_lab = np.load(F'/gate/eet6/dof{i:03d}.npy')[0]
        mot_from_lab = index_map(d['motor_joint'], d['lab_joint'])
        lab_from_mot = index_map(d['lab_joint'], d['motor_joint'])
        target_dof_pos = np.zeros_like(dof_lab)
        target_dof_pos[:] = dof_lab[lab_from_mot]

        dof = act_to_dof(obs, act)

        export = target_dof_pos
        calc = dof

        # print('exported', target_dof_pos[mot_from_arm],
        #       'calculated', dof[mot_from_arm])
        # print( (export - calc)[mot_from_arm] )

        # exps.append(target_dof_pos[mot_from_arm])
        # cals.append(dof[mot_from_arm])

        exps.append(target_dof_pos)
        cals.append(dof)

        q_lab = obs[..., 32:61]
        q_mot = (q_lab + config.lab_joint_offsets)[act_to_dof.lab_from_mot]
        q_arm = q_mot[mot_from_arm]
        # poss.append(q_arm)
        poss.append(q_mot)

        # print('q_mot', q_mot,
        #       'sim - q_mot', target_dof_pos - q_mot,
        #       'real - q_mot', dof - q_mot)
        # break

    exps = np.asarray(exps, dtype=np.float32)
    cals = np.asarray(cals, dtype=np.float32)
    poss = np.asarray(poss, dtype=np.float32)
    # print(exps.shape)
    # print(cals.shape)

    # fig, ax = plt.subplots(29, 1)
    fig, axs = plt.subplots(6, 5)

    q_lo = act_to_dof.lim_lo_pin[act_to_dof.pin_from_mot]
    q_hi = act_to_dof.lim_hi_pin[act_to_dof.pin_from_mot]
    RES = False
    for i in range(29):

        ii = i // 5
        jj = i % 5
        ax = axs[ii, jj]

        ax.set_title(config.motor_joint[i])
        if RES:
            ax.axhline(0)
        else:
            ax.axhline(q_lo[i], color='k', linestyle='--')
            ax.axhline(q_hi[i], color='k', linestyle='--')
            pass
        if RES:
            ax.plot(exps[:, i] - poss[:, i], label='sim')
            ax.plot(cals[:, i] - poss[:, i], label='real')
        else:
            ax.plot(poss[:, i], label='pos')
            ax.plot(exps[:, i], 'x-', label='sim')
            ax.plot(cals[:, i], '+-', label='real')
        ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
