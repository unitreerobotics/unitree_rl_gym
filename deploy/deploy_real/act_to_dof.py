import numpy as np
import torch
import pinocchio as pin
from common.np_math import (quat_from_angle_axis, quat_mul,
                            xyzw2wxyz, index_map)
from config import Config
from ikctrl import IKCtrl

class ActToDof:
    def __init__(self, config):
        self.config=config
        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
            config.arm_joint
        )
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
        self.mot_from_arm = index_map(
            self.config.motor_joint,
            self.config.arm_joint
        )
        self.mot_from_lab = index_map(
            self.config.motor_joint,
            self.config.lab_joint
        )
        self.mot_from_nonarm = index_map(
            self.config.motor_joint,
            self.config.non_arm_joint
        )
    def __call__(self, obs, action):
        hands_command = obs[..., 119:225]
        non_arm_joint_pos = action[..., :22]
        left_arm_residual = action[..., 22:29]
        
        q_lab = obs[..., 32:61]
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_lab] = q_lab

        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q_lab

        d_quat = quat_from_angle_axis(
            torch.from_numpy(hands_command[..., 3:])
        ).detach().cpu().numpy()

        source_pose = self.ikctrl.fk(q_pin)
        source_xyz = source_pose.translation
        source_quat = xyzw2wxyz(pin.Quaternion(source_pose.rotation).coeffs())

        target_xyz = source_xyz + hands_command[..., :3]
        target_quat = quat_mul(
            torch.from_numpy(d_quat),
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
                q_mot[i_mot]
                + res_q_ik[i_arm]
                + np.clip(0.3 * left_arm_residual[i_arm],
                          -0.2, 0.2)
            )
            target_q = np.clip(target_q,
                               self.lim_lo_pin[i_pin],
                               self.lim_hi_pin[i_pin])
            target_dof_pos[i_mot] = target_q
        return target_dof_pos

def main():
    import yaml

    with open('configs/g1_eetrack.yaml', 'r') as fp:
        d = yaml.safe_load(fp)

    act_to_dof = ActToDof(Config('configs/g1_eetrack.yaml'))
    obs = np.load('/tmp/eet4/obs001.npy')[0]
    act = np.load('/tmp/eet4/act001.npy')[0]
    dof_lab = np.load('/tmp/eet4/dof001.npy')[0]
    mot_from_lab = index_map(d['motor_joint'], d['lab_joint'])
    target_dof_pos = np.zeros_like(dof_lab)
    target_dof_pos[mot_from_lab] = dof_lab


    dof = act_to_dof(obs, act)

if __name__ == '__main__':
    main()