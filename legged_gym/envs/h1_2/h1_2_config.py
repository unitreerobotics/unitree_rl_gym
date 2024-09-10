from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.6,
            'left_knee_joint': 1.2,
            'left_ankle_pitch_joint': -0.6,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.6,
            'right_knee_joint': 1.2,
            'right_ankle_pitch_joint': -0.6,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0,

            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
        }

    class env(LeggedRobotCfg.env):
        num_actions = 21
        num_observations = 12 + num_actions * 3
        num_envs = 8192

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 60.,
            'ankle_roll_joint': 40.,
            'torso_joint': 600.,
            'shoulder_pitch_joint': 80.,
            'shoulder_roll_joint': 80.,
            'shoulder_yaw_joint': 40.,
            'elbow_pitch_joint': 60.,
        }  # [N*m/rad]
        damping = {
            'hip_yaw_joint': 5.0,
            'hip_roll_joint': 5.0,
            'hip_pitch_joint': 5.0,
            'knee_joint': 7.5,
            'ankle_pitch_joint': 1.0,
            'ankle_roll_joint': 0.3,
            'torso_joint': 15.0,
            'shoulder_pitch_joint': 2.0,
            'shoulder_roll_joint': 2.0,
            'shoulder_yaw_joint': 1.0,
            'elbow_pitch_joint': 1.0,
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_simplified.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        armature = 6e-4  # stablize semi-euler integration for end effectors

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # stablize semi-euler integration for end effectors

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            orientation = -0.1
            base_height = -10.0
            dof_acc = -3e-8
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.1
            dof_pos_limits = -10.0
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0


class H1_2RoughCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3
        activation = 'tanh'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'h1_2'
