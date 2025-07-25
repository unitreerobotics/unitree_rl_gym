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
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        act_joint = config.arm_joint
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf',
                             act_joint)
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit

        # == build index map ==
        self.pin_from_mot = np.zeros(29, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_pin = np.zeros(43, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_act = np.zeros(7, dtype=np.int32) # FIXME(ycho): hardcoded
        for i_mot, j in enumerate( self.config.motor_joint ):
            i_pin = (self.ikctrl.robot.index(j) - 1)
            self.pin_from_mot[i_mot] = i_pin
            self.mot_from_pin[i_pin] = i_mot
            if j in act_joint:
                i_act = act_joint.index(j)
                self.mot_from_act[i_act] = i_mot

        q_mot = np.array(config.default_angles)
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_mot] = q_mot

        if True:
            default_pose = self.ikctrl.fk(q_pin)
            xyz = default_pose.translation
            quat_wxyz = xyzw2wxyz(pin.Quaternion(default_pose.rotation).coeffs())
            self.default_pose = np.concatenate([xyz, quat_wxyz])
            self.target_pose = np.copy(self.default_pose)

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(43, dtype=np.float32)
        self.dqj = np.zeros(43, dtype=np.float32)
        self.action = np.zeros(7, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

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
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx
        
        # record the current pos
        self._init_dof_pos = np.zeros(29)
        for i in range(29):
            self._init_dof_pos[i] = self.low_state.motor_state[i].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            #for j in range(self._dof_size):
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
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_angles[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.arm_waist_target[i])
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

        # Get current joint positions
        for i_mot in range(len(self.config.motor_joint)):
            i_pin = self.pin_from_mot[i_mot]
            self.qj[i_pin] = self.low_state.motor_state[i_mot].q

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        if False:
            delta = np.concatenate([self.cmd,
                                    [1,0,0,0]])
            res_q = self.ikctrl(self.qj, delta, rel=True)
        else:
            # FIXME(ycho): 0.01 --> cmd_scale ?
            self.target_pose[..., :3] += 0.01 * self.cmd
            res_q = self.ikctrl(self.qj,
                                self.target_pose,
                                rel=False)

        for i_act in range(len(res_q)):
            i_mot = self.mot_from_act[i_act]
            i_pin = self.pin_from_mot[i_mot]
            target_q = (
                    self.low_state.motor_state[i_mot].q + res_q[i_act]
            )
            target_q = np.clip(target_q,
                               self.lim_lo_pin[i_pin],
                               self.lim_hi_pin[i_pin])
            self.low_cmd.motor_cmd[i_mot].q = target_q
            self.low_cmd.motor_cmd[i_mot].dq = 0.0
            # FIXME(ycho): arbitrary scaling
            self.low_cmd.motor_cmd[i_mot].kp = 0.2*float(self.config.kps[i_mot])
            self.low_cmd.motor_cmd[i_mot].kd = 0.2*float(self.config.kds[i_mot])
            self.low_cmd.motor_cmd[i_mot].tau = 0.0
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
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)
