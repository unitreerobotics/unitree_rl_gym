from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2RoughCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.98]  # x,y,z [m]
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
        
    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 2.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class env(LeggedRobotCfg.env):
        num_actions = 21
        num_policy_output = 21
        num_envs = 3000
        obs_context_len = 15
        mlp_obs_context_len = 3
        num_observations_single = 11 + num_actions * 4 #+ 9 * 4
        num_privileged_obs_single = 14 + num_actions * 4 #+ 9 * 4
        num_observations = num_observations_single * obs_context_len
        num_privileged_obs = num_privileged_obs_single * obs_context_len
        num_teaching_observations = num_observations_single
        
        action_delay = 0.02
        episode_length_s = 20

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.9
    
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
        action_scale = 0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_simplified.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = [
            "pelvis",
            "hip",
            "knee",
            "shoulder_pitch",
            "elbow"
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        eplace_cylinder_with_capsule = False
        collapse_fixed_joints = False
        armature = 6e-4  # stablize semi-euler integration for end effectors

        vhacd_enabled = False
        vhacd_params_resolution = 500000

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5
        
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0 # sec
        gravity_impulse_duration = 0.99
        
        dof_prop_rand_interval_s = 6
        randomize_pd_params = True
        kp_ratio_range = [0.5, 1.5]
        kd_ratio_range = [0.5, 1.5]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.02, 0.02]
        # randomize_decimation = False
        # decimation_range = [0.5, 1.5]
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # stablize semi-euler integration for end effectors

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9


        base_height_target = 0.98

        # for knee and ankle distance keeping
        min_dist = 0.3
        max_dist = 0.5

        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        max_contact_force = 600. # forces above this value are penalized
        cycle_time = 0.4 #.4 #0.68
        target_joint_pos_scale = 0.17
        target_feet_height = 0.06

        class scales(LeggedRobotCfg.rewards.scales):

            joint_pos = 1.6 * 4
            feet_clearance = 1. * 2
            feet_contact_number = 1.2 * 1
            feet_air_time = 1.0 * 2
            foot_slip = -0.1

            tracking_lin_vel = 1.0 * 8
            tracking_ang_vel = 1.0 * 8
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            orientation = -1
            base_height = -100.0
            dof_acc = -3e-8
            collision = -1.0

            action_rate = -0.6
            action_smoothness = -0.6

            dof_pos_limits = -10.0
            dof_vel_limits = -10.0
            torque_limits = -10.0
            default_joint_pos = 0.5 * 4
            feet_contact_forces = -3e-6 * 50
            
            ## feet reg
            feet_parallel = -1.0
            feet_symmetry = -1.0
            knee_distance = 0.2
            feet_distance = 0.2

            arm_pose = -1
            stumble = -2.0
            stand_still = -1.0 * 0
        
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0


class H1_2RoughCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3
        activation = 'tanh'
        policy_type = 'moving'
        architecture = 'Trans' # choose from 'Mix', 'Trans', 'MLP', and 'RNN'
        teaching_model_path = '/home/ziluoding/unitree_rl_gym/logs/Sep06_19-03-45_/model_13000.pt'#'/home/ps/unitree_rl_gym_o/legged_gym/model/Aug29_17-48-05_h1/model_10000.pt'
        moving_model_path = '/home/ziluoding/humanoid-gym/logs/h1/Jul11_16-30-02_/model_12000.pt'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'h1_2'
        run_name = ''
        # load and resume
        render = True
        resume = True
        load_run = 'Oct24_21-41-11_finetunexvel-12noheadingcycle0.40.2' # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
