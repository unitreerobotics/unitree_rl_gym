from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            if 'leg_joint2motor_idx' in config:
                self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            if 'joint2motor_idx' in config:
                self.joint2motor_idx = config["joint2motor_idx"]
            
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            if 'arm_waist_joint2motor_idx' in config:
                self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
                self.arm_waist_kps = config["arm_waist_kps"]
                self.arm_waist_kds = config["arm_waist_kds"]
                self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)
            else:
                self.arm_waist_joint2motor_idx = []
                self.arm_waist_kps = []
                self.arm_waist_kds = []
                self.arm_waist_target = []

            if 'motor_joint' in config:
                self.motor_joint = config['motor_joint']
            else:
                self.motor_joint=[]

            if 'arm_joint' in config:
                self.arm_joint = config['arm_joint']
            else:
                self.arm_joint=[]

            if 'non_arm_joint' in config:
                self.non_arm_joint = config['non_arm_joint']
            else:
                self.non_arm_joint=[]

            if 'lab_joint' in config:
                self.lab_joint = config['lab_joint']
            else:
                self.lab_joint=[]
                
            if 'lab_joint_offsets' in config:
                self.lab_joint_offsets = config['lab_joint_offsets']
            else:
                self.lab_joint_offsets=[]
                
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
