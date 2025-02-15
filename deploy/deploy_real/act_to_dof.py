import numpy as np
import torch
import pinocchio as pin
from common.np_math import (quat_from_angle_axis, quat_mul,
                            xyzw2wxyz, index_map)
from config import Config
from ikctrl import IKCtrl

class ActToDof:
    def __init__(self, config, ikctrl):
        self.config=config
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
        self.default_nonarm = (
            np.asarray(self.config.lab_joint_offsets)[self.lab_from_nonarm]
        )

    def __call__(self, obs, action):
        hands_command = obs[..., 119:125]
        non_arm_joint_pos = action[..., :22]
        left_arm_residual = action[..., 22:29]
        
        q_lab = obs[..., 32:61]
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_lab] = q_lab
        q_pin[self.pin_from_lab] += np.asarray(self.config.lab_joint_offsets)

        q_mot = np.zeros(29)
        q_mot[self.mot_from_lab] = q_lab
        q_mot[self.mot_from_lab] += np.asarray(self.config.lab_joint_offsets)
        #print('q_mot (inside)', q_mot)

        # print('q0', q_mot[self.mot_from_arm])
        # q_mot[i_mot] = q_lab[ lab_from_mot[i_mot] ]

        d_quat = quat_from_angle_axis(
            torch.from_numpy(hands_command[..., 3:])
        ).detach().cpu().numpy()
        # print('d_quat', d_quat)

        source_pose = self.ikctrl.fk(q_pin)
        source_xyz = source_pose.translation
        source_quat = xyzw2wxyz(pin.Quaternion(source_pose.rotation).coeffs())

        target_xyz = source_xyz + hands_command[..., :3]
        target_quat = quat_mul(
            torch.from_numpy(d_quat),
            torch.from_numpy(source_quat)).detach().cpu().numpy()
        target = np.concatenate([target_xyz, target_quat])
        # print('target', target)
        # print('q_pin', q_pin)
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

            # print('default joint pos', self.default_nonarm)
            # print('joint order', self.config.non_arm_joint)
            # print('mot_from_nonarm', self.mot_from_nonarm)
            target_dof_pos[self.mot_from_nonarm] = (
                self.default_nonarm + 0.5 * non_arm_joint_pos
            )

            target_dof_pos[self.mot_from_arm] = np.clip(
                    target_dof_pos[self.mot_from_arm],
                    self.lim_lo_pin[self.pin_from_arm],
                    self.lim_hi_pin[self.pin_from_arm]
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
        config.arm_joint
    )
    act_to_dof = ActToDof(config, ikctrl)

    exps = []
    cals = []
    poss = []

    for i in range(70):
        obs = np.load(F'/tmp/eet5/obs{i:03d}.npy')[0]
        # print(obs.shape)
        act = np.load(F'/tmp/eet5/act{i:03d}.npy')[0]
        # print('act', act.shape)
        dof_lab = np.load(F'/tmp/eet5/dof{i:03d}.npy')[0]
        mot_from_lab = index_map(d['motor_joint'], d['lab_joint'])
        target_dof_pos = np.zeros_like(dof_lab)
        target_dof_pos[mot_from_lab] = dof_lab


        dof = act_to_dof(obs, act)

        export = target_dof_pos
        calc   = dof

        mot_from_arm = index_map(d['motor_joint'], d['arm_joint'])
        # print('exported', target_dof_pos[mot_from_arm],
        #       'calculated', dof[mot_from_arm])
        # print( (export - calc)[mot_from_arm] )
        exps.append( target_dof_pos[mot_from_arm] )
        cals.append( dof[mot_from_arm] )

        q_lab = obs[..., 32:61]
        q_mot = q_lab[act_to_dof.lab_from_mot]
        q_arm = q_mot[mot_from_arm]
        poss.append(q_arm)

    exps=np.asarray(exps, dtype=np.float32)
    cals=np.asarray(cals, dtype=np.float32)
    poss=np.asarray(poss, dtype=np.float32)
    print(exps.shape)
    print(cals.shape)

    fig, ax=plt.subplots(7,1)
    
    q_lo =act_to_dof.lim_lo_pin[act_to_dof.pin_from_arm]
    q_hi =act_to_dof.lim_hi_pin[act_to_dof.pin_from_arm]
    RES = True
    for i in range(7):
        if RES:
            ax[i].axhline(0)
        else:
            ax[i].axhline(q_lo[i], color='k', linestyle='--')
            ax[i].axhline(q_hi[i], color='k', linestyle='--')
        if RES:
            ax[i].plot(exps[:, i]-poss[:,i], label='sim')
            ax[i].plot(cals[:, i]-poss[:,i], label='real')
        else:
            ax[i].plot(poss[:, i], label='pos')
            ax[i].plot(exps[:, i], label='sim')
            ax[i].plot(cals[:, i], label='real')
        ax[i].legend()
    plt.show()

if __name__ == '__main__':
    main()
