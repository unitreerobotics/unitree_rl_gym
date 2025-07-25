from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.2,         
           'left_knee_joint' : 0.42,       
           'left_ankle_pitch_joint' : -0.23,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.2,                                       
           'right_knee_joint' : 0.42,                                             
           'right_ankle_pitch_joint': -0.23,                              
           'right_ankle_roll_joint' : 0,       
           'left_elbow_joint': 0.87,
           'right_elbow_joint': 0.87,
           'left_shoulder_roll_joint': 0.16,
           'left_shoulder_pitch_joint': 0.35,
           'left_shoulder_yaw_joint': 0.,
           'right_shoulder_roll_joint': -0.16,
           'right_shoulder_pitch_joint': 0.35,
           'right_shoulder_yaw_joint': 0.,
           'waist_roll_joint' : 0,               
           'waist_pitch_joint' : 0,               
           'waist_yaw_joint' : 0,               
           'left_wrist_roll_joint' : 0,               
           'left_wrist_pitch_joint' : 0,               
           'left_wrist_yaw_joint' : 0,               
           'right_wrist_roll_joint' : 0,               
           'right_wrist_pitch_joint' : 0,               
           'right_wrist_yaw_joint' : 0,               

        }
    
    class env(LeggedRobotCfg.env):
        # num_observations = 47
        num_observations = 96
        # num_privileged_obs = 50
        num_privileged_obs = 96
        # num_actions = 12
        num_actions = 29


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'shoulder_pitch': 100,
                     'shoulder_roll': 100,
                     'shoulder_yaw': 50,
                     'elbow': 50,
                     'wrist': 20,
                     'waist': 150,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 2,
                     'elbow': 2,
                     'wrist': 1,
                     'waist': 3,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.25
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        # terminate_after_contacts_on = ["pelvis"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        # actor_hidden_dims = [32]
        actor_hidden_dims = [256, 128, 128]
        # critic_hidden_dims = [32]
        critic_hidden_dims = [256, 128, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = "ActorCriticRecurrent"
        policy_class_name = "ActorCritic"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
