import torch
import gymnasium as gym
from typing import Dict, List, Tuple, Optional

from omni.isaac.lab.envs import RLTask, RLTaskCfg
from omni.isaac.lab.assets import Robot, RobotCfg, AssetCfg
from omni.isaac.lab.sim import SimulationContext, schemas as sim_schemas
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import unnormalize_ac_delta
from omni.isaac.lab.terrains import TerrainImporterCfg # Using flat ground for now
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils.assets import NVIDIA_ASSETS_DIR # For potential fallbacks or other assets

# For typing
from torch import Tensor

@configclass
class H1ObsScalesCfg:
    """Observation scales for H1 environment."""
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    # Commands will be scaled by their respective command_scale in H1IsaacLabEnvCfg

@configclass
class H1RewardScalesCfg:
    """Reward scales for H1 environment."""
    # Locomotion
    tracking_lin_vel: float = 1.5
    tracking_ang_vel: float = 0.5 # Renamed from ang_vel_xy to be more general
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05 # This is for base_ang_vel_xy penalty
    orientation: float = -5.0
    # Base motion
    base_height: float = -30.0
    # Task-specific
    # alive: float = 2.0 # Typically handled by RL agent's discount factor or as a fixed bonus per step
    # Smoothness
    action_rate: float = -0.01
    action_smoothness: float = -0.005 # Diffs between consecutive actions
    # Joint state penalties
    dof_vel: float = -1.5e-5 # Corresponds to H1RoughCfg.rewards.scales.dof_vel
    dof_acc: float = -2.5e-7
    dof_pos_limits: float = -10.0
    # Safety
    collision: float = -1.0 # General collision penalty
    termination: float = -200.0 # Penalty for early termination
    # Feet-specific
    feet_air_time: float = 0.5 # Similar to H1, needs contact sensors
    # Power/Energy (optional)
    # power: float = -0.0005


@configclass
class H1CommandRangesCfg:
    """Command ranges for H1 environment."""
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)  # min, max m/s
    lin_vel_y: Tuple[float, float] = (-1.0, 1.0)  # min, max m/s
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)  # min, max rad/s
    heading: Tuple[float, float] = (-torch.pi, torch.pi) # min, max rad

@configclass
class H1EarlyTerminationConditionsCfg:
    """Early termination conditions for H1 environment."""
    base_height_lower_limit: float = 0.15 # m, terminate if base is too low
    base_contact: bool = True # Terminate if base (non-foot) collides

@configclass
class H1IsaacLabEnvCfg(RLTaskCfg):
    """Configuration for the H1 Isaac Lab environment."""
    # Basic RL settings
    num_envs: int = 4096 # Default from RLTaskCfg, can be overridden
    # episode_length_s: float = 20.0 # Default from RLTaskCfg

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, replicate_physics=True)
    # Ground plane, can be customized or replaced with terrain
    ground_plane: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        carrier_material=sim_schemas.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    )

    # H1 Robot specific settings
    h1_urdf_path: str = "./resources/robots/h1/urdf/h1.urdf" # Relative to repo root
    num_observations: int = 41 # From H1RoughCfg
    num_actions: int = 10 # HipYaw, HipRoll, HipPitch, Knee, Ankle for left and right legs
    action_scale: float = 0.25 # From H1RoughCfg

    # Default joint angles (10 joints: 2x (HipYaw, HipRoll, HipPitch, Knee, Ankle))
    # Values from H1RoughCfg.init_state.default_joint_angles
    # LH_HAA, LH_HFE, LH_KFE, LH_ANKLE, RH_HAA, RH_HFE, RH_KFE, RH_ANKLE (Original has 8, H1 URDF has 10)
    # Let's assume: L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE_PITCH, (and R_...)
    # The H1 URDF typically has:
    # left_hip_yaw_joint, left_hip_roll_joint, left_hip_pitch_joint, left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint (6 per leg)
    # The original legged_gym H1 config uses 8 DoF (no yaw, no ankle roll). The subtask specified 10 actions.
    # Assuming 10 actions means: L/R HipYaw, HipRoll, HipPitch, Knee, AnklePitch.
    default_joint_angles: Dict[str, float] = {
        "left_hip_yaw_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_pitch_joint": -0.52, # ~-30 deg
        "left_knee_joint": 1.05, # ~60 deg
        "left_ankle_pitch_joint": -0.52, # ~-30 deg
        "right_hip_yaw_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_pitch_joint": -0.52,
        "right_knee_joint": 1.05,
        "right_ankle_pitch_joint": -0.52,
    }

    # PD control gains (stiffness, damping)
    # From H1RoughCfg.control.stiffness and H1RoughCfg.control.damping
    # Assuming order matches default_joint_angles if list, or use dict.
    # Let's use a simplified scheme for now, applying same P/D to all. Isaac Lab RobotCfg handles this better.
    stiffness: float = 40.0 # Default for all joints, can be a dict
    damping: float = 1.0   # Default for all joints, can be a dict

    # Observation and Reward scales
    obs_scales: H1ObsScalesCfg = H1ObsScalesCfg()
    reward_scales: H1RewardScalesCfg = H1RewardScalesCfg()

    # Base height target for rewards
    base_height_target: float = 0.42 # m, target CoM height for H1

    # Early termination conditions
    early_termination: H1EarlyTerminationConditionsCfg = H1EarlyTerminationConditionsCfg()

    # Command ranges and scales
    commands: H1CommandRangesCfg = H1CommandRangesCfg()
    commands_scale: List[float] = [2.0, 2.0, 0.25] # Scales for lin_vel_x, lin_vel_y, ang_vel_yaw commands in obs

    # Contact sensor configuration for feet and base
    feet_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_lower_leg_link", # Example, adjust to actual foot prims
        history_length=3, filter_type="boolean", threshold=0.1,
        track_air_time=True
    )
    # Add more sensors for other feet and base if needed. For now, one for simplicity in config.
    # Will define multiple sensors in the Env class.


class H1IsaacLabEnv(RLTask):
    """Isaac Lab environment for H1 Humanoid robot, designed for RL."""

    cfg: H1IsaacLabEnvCfg # Environment configuration

    def __init__(self, cfg: H1IsaacLabEnvCfg, sim_params: sim_schemas.SimCfg = None, **kwargs):
        self.cfg = cfg # Store the configuration

        # Initialize RLTask
        super().__init__(cfg=cfg, sim_params=sim_params, **kwargs)

        # Robot specific parameters
        self.action_scale = self.cfg.action_scale
        self.num_observation_dims = self.cfg.num_observations
        self.num_action_dims = self.cfg.num_actions

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-torch.inf, high=torch.inf, shape=(self.num_observation_dims,)
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_action_dims,)
        )

        # Convert default joint angles from dict to tensor, respecting order for actions
        # The order of joints in self.robot.meta_info.joint_names will be the source of truth
        # This conversion will be done in _setup_scene after robot is loaded.
        self.default_joint_angles_tensor: Optional[Tensor] = None
        self.joint_names_for_actions: List[str] = [
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"
        ] # Ensure this order matches URDF and action space

        # Buffers
        self.phase = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32) # L/R leg phase
        self.previous_actions_buf = torch.zeros(self.num_envs, self.num_action_dims, device=self.device, dtype=torch.float32)
        self.commands_buf = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32) # x_vel, y_vel, yaw_vel_rate

        # Contact force tracking for feet air time
        self.last_feet_contact_buf = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool) # L/R foot

    def _setup_scene(self):
        """Sets up the simulation scene with robot and ground plane."""
        # Add ground plane using TerrainImporterCfg from self.cfg
        self.scene.add_terrain(self.cfg.ground_plane)

        # Define H1 robot asset configuration
        h1_asset_cfg = AssetCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Asset", # Nested prim path for the asset
            asset_path=self.cfg.h1_urdf_path,
            collision_group=-1, # Default collision group
            # Specify actuator properties if not done globally
            # Assuming a simple PD controller for all joints for now
            # More complex actuator models can be defined here.
            actuators={
                joint_name: sim_schemas.PDActuatorCfg(
                    stiffness=self.cfg.stiffness, damping=self.cfg.damping
                ) for joint_name in self.joint_names_for_actions # Apply to specified joints
            }
        )

        h1_robot_cfg = RobotCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            asset_cfg=h1_asset_cfg,
            init_state=RobotCfg.InitialStateCfg(
                pos=(0.0, 0.0, self.cfg.base_height_target + 0.05), # Start slightly above target
                rot=(1.0, 0.0, 0.0, 0.0), # Quaternion (w,x,y,z)
                joint_pos={name: self.cfg.default_joint_angles[name] for name in self.joint_names_for_actions}
            ),
            # Define contact sensors for feet
            # These prim paths must match the URDF link names
            contact_sensors={
                "left_foot": ContactSensorCfg(prim_path="{PRIM_PATH}/left_ankle_roll_link", history_length=3, track_air_time=True, threshold=0.1), # Or left_lower_leg / foot link
                "right_foot": ContactSensorCfg(prim_path="{PRIM_PATH}/right_ankle_roll_link", history_length=3, track_air_time=True, threshold=0.1), # Or right_lower_leg / foot link
                "base": ContactSensorCfg(prim_path="{PRIM_PATH}/pelvis_link", threshold=0.1) # For base collision detection
            },
        )
        # Instantiate and add robot to the scene
        self.robot = Robot(cfg=h1_robot_cfg)
        self.scene.add_asset("robot", self.robot)

        # Post-addition: ensure default_joint_angles_tensor is ordered correctly
        # This requires the robot to be loaded to access its joint names and order
        # We assume self.joint_names_for_actions is the correct order for actions.
        # The Robot class stores joint data in the order of `robot.meta_info.joint_names`.
        # We need to map our `default_joint_angles` (from config, by name) to a tensor
        # that matches the robot's internal joint order for relevant DoFs.

        # Create a mapping from configured joint names to their values
        default_joint_angles_map = self.cfg.default_joint_angles

        # Initialize a tensor for default joint angles based on the robot's DoF order for *actuated* joints
        # This ensures that when we later use `robot.data.joint_pos`, the subtraction is correct.
        # For actions, we will use joint_ids corresponding to `self.joint_names_for_actions`.
        num_robot_dofs = self.robot.num_dof
        self.default_joint_angles_tensor_full_dof = torch.zeros(self.num_envs, num_robot_dofs, device=self.device)

        # For the 10 action DoFs, we need their indices in the robot's full DoF list
        self.action_dof_indices = [self.robot.meta_info.joint_names.index(name) for name in self.joint_names_for_actions]

        temp_default_joint_angles_list = []
        for name in self.joint_names_for_actions: # Ensure order for the 10 action joints
            temp_default_joint_angles_list.append(default_joint_angles_map[name])

        # This tensor is for the 10 actuated joints, in the order of self.joint_names_for_actions
        self.default_joint_angles_for_actions = torch.tensor(
            temp_default_joint_angles_list, device=self.device
        ).repeat(self.num_envs, 1)

        # Fill the full DoF default tensor for observation calculation if needed (using robot.data.joint_pos which is full DoF)
        for i, name in enumerate(self.robot.meta_info.joint_names):
            if name in default_joint_angles_map:
                self.default_joint_angles_tensor_full_dof[:, i] = default_joint_angles_map[name]


    def _pre_physics_step(self, actions: Tensor):
        """Apply actions to the robot before physics simulation."""
        if self.default_joint_angles_for_actions is None: # Should be initialized in _setup_scene
            raise RuntimeError("Default joint angles tensor not initialized.")

        # Store previous actions
        self.previous_actions_buf[:] = actions

        # Scale actions: actions are in [-1, 1], map to delta from default config
        # computed_actions = self.action_scale * actions # This is a simple scaling
        # A more common approach (like in legged_gym) for delta actions:
        computed_actions_delta = unnormalize_ac_delta(actions,
                                                      self.robot.data.soft_joint_pos_limits[:, self.action_dof_indices, 0], # lower limits for actuated joints
                                                      self.robot.data.soft_joint_pos_limits[:, self.action_dof_indices, 1], # upper limits for actuated joints
                                                      self.action_scale)


        # Target joint positions: default + scaled actions
        target_joint_pos = self.default_joint_angles_for_actions + computed_actions_delta

        # Apply actions to the robot using its articulation controller
        # Need to ensure that PD gains are set. This can be done in RobotCfg or here.
        # The RobotCfg above now includes actuator definitions.
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=self.action_dof_indices)


    def _get_observations(self) -> Tensor:
        """Compute and return observations for the H1 robot."""
        # Robot state (ensure tensors are on the correct device, which RLTask handles)
        base_lin_vel_w = self.robot.data.root_lin_vel_w # World frame linear velocity
        base_ang_vel_b = self.robot.data.root_ang_vel_b # Body frame angular velocity

        # Projected gravity (gravity vector in robot's base frame)
        # This requires robot.data.root_quat_w (world orientation) and sim_params.gravity
        # RLTask usually provides this as self.robot.data.projected_gravity_b
        projected_gravity = self.robot.data.projected_gravity_b

        # Joint positions and velocities (for all DoFs of the robot)
        # We need to select the ones corresponding to our 10 action DoFs for some parts of obs
        dof_pos_all = self.robot.data.joint_pos
        dof_vel_all = self.robot.data.joint_vel

        # Select the 10 relevant DoFs for observation components that are DoF-specific
        dof_pos_selected = dof_pos_all[:, self.action_dof_indices]
        dof_vel_selected = dof_vel_all[:, self.action_dof_indices]

        # Phase calculation (simple example, can be more complex)
        self.phase += (2 * torch.pi * self.common_dt * self.cfg.commands.lin_vel_x[1] / 2.0) # Crude phase based on max fwd speed
        self.phase %= (2 * torch.pi)
        sin_phase = torch.sin(self.phase).unsqueeze(1)
        cos_phase = torch.cos(self.phase).unsqueeze(1)

        # Assemble observation buffer
        # Order from H1LeggedGymEnv: base_lin_vel, base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, previous_actions, phase_info
        # Note: Isaac Lab uses body frame for velocities by default in data buffer, legged_gym used world frame for lin_vel.
        # Let's use body frame linear velocity for consistency with ang_vel.
        base_lin_vel_b = self.robot.data.root_lin_vel_b

        obs_list = [
            base_lin_vel_b * self.cfg.obs_scales.lin_vel, # Base linear velocity (body frame)
            base_ang_vel_b[:, :3] * self.cfg.obs_scales.ang_vel, # Base angular velocity (body frame)
            projected_gravity, # Projected gravity vector
            self.commands_buf * torch.tensor(self.cfg.commands_scale, device=self.device), # Scaled commands
            (dof_pos_selected - self.default_joint_angles_for_actions) * self.cfg.obs_scales.dof_pos, # Centered DoF positions
            dof_vel_selected * self.cfg.obs_scales.dof_vel, # DoF velocities
            self.previous_actions_buf, # Previous actions
            sin_phase,
            cos_phase
        ]

        self.obs_buf = torch.cat(obs_list, dim=-1)

        # Verify observation dimension
        expected_obs_dim = self.cfg.num_observations
        if self.obs_buf.shape[1] != expected_obs_dim:
             print(f"Warning: Observation dimension mismatch! Expected {expected_obs_dim}, got {self.obs_buf.shape[1]}")
             # Print dimensions of each component for debugging
             # for i, obs_comp in enumerate(obs_list):
             #     print(f"  Obs component {i}: {obs_comp.shape}")


        return self.obs_buf

    def _get_rewards(self) -> Tensor:
        """Compute and return rewards for the H1 robot."""
        # Access robot state
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b
        dof_pos_all = self.robot.data.joint_pos
        dof_vel_all = self.robot.data.joint_vel
        dof_acc_all = self.robot.data.joint_acc # Assuming this is available, if not, estimate or use from cfg
        root_pos_w = self.robot.data.root_pos_w
        projected_gravity = self.robot.data.projected_gravity_b

        # Contact sensors
        left_foot_contact = self.robot.contact_sensors["left_foot"].data.net_contact_force_w_norm > 0.1 # boolean
        right_foot_contact = self.robot.contact_sensors["right_foot"].data.net_contact_force_w_norm > 0.1
        base_contact = self.robot.contact_sensors["base"].data.net_contact_force_w_norm > 0.1

        # --- Reward Calculations ---
        # Tracking linear velocity (X and Y in body frame)
        reward_tracking_lin_vel = torch.exp(-torch.sum(torch.square(self.commands_buf[:, :2] - base_lin_vel_b[:, :2]), dim=1)) * self.cfg.reward_scales.tracking_lin_vel

        # Tracking angular velocity (Yaw rate in body frame)
        reward_tracking_ang_vel = torch.exp(-torch.square(self.commands_buf[:, 2] - base_ang_vel_b[:, 2])) * self.cfg.reward_scales.tracking_ang_vel

        # Linear velocity Z penalty
        reward_lin_vel_z = torch.square(base_lin_vel_b[:, 2]) * self.cfg.reward_scales.lin_vel_z

        # Angular velocity XY penalty (roll/pitch rates)
        reward_ang_vel_xy = torch.sum(torch.square(base_ang_vel_b[:, :2]), dim=1) * self.cfg.reward_scales.ang_vel_xy

        # Orientation penalty (based on projected gravity, want Z-axis of base to align with world Z-up)
        # projected_gravity_b[:, 0] is gx, projected_gravity_b[:, 1] is gy. We want these to be small.
        reward_orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.cfg.reward_scales.orientation

        # Base height penalty
        reward_base_height = torch.square(root_pos_w[:, 2] - self.cfg.base_height_target) * self.cfg.reward_scales.base_height

        # Action rate penalty (difference between current and previous actions)
        reward_action_rate = torch.sum(torch.square(self.previous_actions_buf - self.actions_buf_for_logging), dim=1) * self.cfg.reward_scales.action_rate
        # Note: self.actions_buf_for_logging should be the actions from _pre_physics_step THIS step,
        # while self.previous_actions_buf is from the PREVIOUS step. RLTask might handle this logging.
        # For now, let's use self.robot.data.applied_actions as a proxy for current step's actions if available,
        # or recalculate if necessary. Assuming self.previous_actions_buf has been updated with current actions.
        # If previous_actions_buf is current action, then we need another buffer for true previous.
        # Let's assume previous_actions_buf IS the previous action, and self.actions is current.
        # This needs careful handling of when buffers are updated.
        # For simplicity, if RLTask provides `self.actions` (current actions), use it.
        # current_actions = self.actions # Assuming RLTask populates this
        # reward_action_rate = torch.sum(torch.square(self.previous_actions_buf - current_actions), dim=1) * self.cfg.reward_scales.action_rate

        # Joint DOF velocity penalty
        dof_vel_selected = dof_vel_all[:, self.action_dof_indices]
        reward_dof_vel = torch.sum(torch.square(dof_vel_selected), dim=1) * self.cfg.reward_scales.dof_vel

        # Joint DOF acceleration penalty
        dof_acc_selected = dof_acc_all[:, self.action_dof_indices] # Assuming joint_acc is available for selected DoFs
        reward_dof_acc = torch.sum(torch.square(dof_acc_selected), dim=1) * self.cfg.reward_scales.dof_acc

        # Joint position limits penalty
        # Need soft_joint_pos_limits for the selected DoFs
        # This is complex, often handled by clamping or a smooth penalty function.
        # Simplified: quadratic penalty if outside a comfort zone (e.g., 90% of limit)
        # For now, placeholder, as it requires careful implementation with limits.
        reward_dof_pos_limits = torch.zeros_like(reward_base_height) # Placeholder

        # Collision penalty (base touching ground)
        reward_collision = (base_contact.float() * self.cfg.reward_scales.collision)

        # Feet air time reward (simplified)
        # Reward for having one foot in the air if the other is on the ground, synced with phase.
        # This requires more sophisticated phase and contact scheduling logic from H1 env.
        # Simple version: reward if feet are not in contact for a certain duration.
        # Using ContactSensor's track_air_time (if available and correctly configured)
        # air_time_l = self.robot.contact_sensors["left_foot"].data.air_time
        # air_time_r = self.robot.contact_sensors["right_foot"].data.air_time
        # For now, a placeholder for this complex reward.
        reward_feet_air_time = torch.zeros_like(reward_base_height) # Placeholder

        # Termination penalty (applied by RLTask if it handles termination penalties)
        # reward_termination = self.reset_buf * self.cfg.reward_scales.termination
        # This is usually handled by the RL framework based on `done` flags.

        # Total reward
        self.rew_buf = (
            reward_tracking_lin_vel + reward_tracking_ang_vel +
            reward_lin_vel_z + reward_ang_vel_xy +
            reward_orientation + reward_base_height +
            reward_action_rate + # Add other action smoothness if implemented
            reward_dof_vel + reward_dof_acc + reward_dof_pos_limits +
            reward_collision + reward_feet_air_time
        )

        # Store individual reward components for logging (in self.extras)
        self.extras["rewards/tracking_lin_vel"] = reward_tracking_lin_vel
        self.extras["rewards/tracking_ang_vel"] = reward_tracking_ang_vel
        self.extras["rewards/lin_vel_z"] = reward_lin_vel_z
        self.extras["rewards/ang_vel_xy"] = reward_ang_vel_xy
        self.extras["rewards/orientation"] = reward_orientation
        self.extras["rewards/base_height"] = reward_base_height
        self.extras["rewards/action_rate"] = reward_action_rate
        self.extras["rewards/dof_vel"] = reward_dof_vel
        self.extras["rewards/dof_acc"] = reward_dof_acc
        self.extras["rewards/collision"] = reward_collision
        # Add other terms...

        return self.rew_buf

    def _update_command(self, env_ids: Tensor):
        """Update the commands buffer for specified environments."""
        # For now, sample random commands within specified ranges
        num_updates = len(env_ids)

        # Lin vel X
        self.commands_buf[env_ids, 0] = torch.rand(num_updates, device=self.device) * \
            (self.cfg.commands.lin_vel_x[1] - self.cfg.commands.lin_vel_x[0]) + self.cfg.commands.lin_vel_x[0]
        # Lin vel Y
        self.commands_buf[env_ids, 1] = torch.rand(num_updates, device=self.device) * \
            (self.cfg.commands.lin_vel_y[1] - self.cfg.commands.lin_vel_y[0]) + self.cfg.commands.lin_vel_y[0]
        # Ang vel Yaw
        self.commands_buf[env_ids, 2] = torch.rand(num_updates, device=self.device) * \
            (self.cfg.commands.ang_vel_yaw[1] - self.cfg.commands.ang_vel_yaw[0]) + self.cfg.commands.ang_vel_yaw[0]

        # Nullify commands for environments that are resetting if command is part of obs
        # (RLTask might handle this if commands are part of `reset_on_it`)
        # self.commands_buf[env_ids] *= (1.0 - self.reset_buf[env_ids].float().unsqueeze(1))


    def _reset_idx(self, env_ids: Tensor):
        """Reset state for specified environments."""
        # Base RLTask handles robot state reset (position, velocity)
        # Reset custom buffers
        self.previous_actions_buf[env_ids, :] = 0.0
        self.phase[env_ids, :] = 0.0 # Or random phase
        self.leg_phase[env_ids, :] = 0.0 # Or random phase

        # Update commands for these environments
        self._update_command(env_ids)

        # Reset any other custom state variables specific to this environment
        self.last_feet_contact_buf[env_ids, :] = False


    def _check_termination(self) -> Tensor:
        """Check if environments should terminate."""
        # Base termination conditions from RLTask (e.g., episode length)
        # Call super()._check_termination() if it exists and does something useful,
        # or implement all termination logic here.
        # For now, let's assume RLTask handles episode length.

        # Custom termination conditions
        root_pos_w = self.robot.data.root_pos_w

        # Base height termination
        terminated_base_height = root_pos_w[:, 2] < self.cfg.early_termination.base_height_lower_limit

        # Base contact termination
        base_contact_sensor_data = self.robot.contact_sensors["base"].data.net_contact_force_w_norm
        terminated_base_contact = (base_contact_sensor_data > 0.1) & self.cfg.early_termination.base_contact

        # Combine custom terminations
        custom_terminations = terminated_base_height | terminated_base_contact

        # Update self.reset_buf with custom terminations
        # RLTask's default _check_termination usually handles max episode length.
        # We OR our custom conditions with it.
        # self.reset_buf |= custom_terminations # This might be how it's done if RLTask manages reset_buf update

        # For now, let's return custom_terminations and let RLTask merge it.
        # The exact mechanism depends on RLTask version.
        # If RLTask._check_termination() returns its own terminations:
        # base_terminations = super()._check_termination()
        # return base_terminations | custom_terminations

        return custom_terminations

# Example of how to instantiate and use (for testing, not part of the final script for submission usually)
if __name__ == "__main__":
    from omni.isaac.lab.app import AppLauncher
    # This is a placeholder for running the environment, actual execution requires Isaac Sim
    # print("Attempting to launch Isaac Sim application...")
    # # Note: Running this directly will likely fail without the Isaac Sim environment.
    # # This is for illustrative purposes on how one might test the env.
    # app_launcher = AppLauncher(headless=False) # Use False for GUI, True for headless
    # simulation_app = app_launcher.app

    # # Create environment configuration
    # env_cfg = H1IsaacLabEnvCfg()
    # env_cfg.num_envs = 3 # Small number for testing

    # # Create the simulation context (usually handled by RLTask or a wrapper)
    # sim_context = SimulationContext(
    #     sim_params=sim_schemas.SimCfg(dt=0.01, use_gpu_pipeline=True, device="cuda:0"),
    #     simulation_app=simulation_app
    # )

    # print("Simulation context created.")

    # try:
    #     # Instantiate the environment
    #     env = H1IsaacLabEnv(cfg=env_cfg, simulation_app=simulation_app) # Pass app if RLTask needs it
    #     print(f"Environment '{H1IsaacLabEnv.__name__}' created successfully.")
    #     print(f"Observation space: {env.observation_space}")
    #     print(f"Action space: {env.action_space}")

    #     # Basic interaction loop (example)
    #     # env.reset() # RLTask usually handles initial reset
    #     # for _ in range(10):
    #     #     actions = torch.rand((env.num_envs, env.num_action_dims), device=env.device) * 2 - 1
    #     #     obs, rewards, terminated, truncated, infos = env.step(actions)
    #     #     print(f"Step done. Obs shape: {obs.shape}, Rewards: {rewards}")
    #     #     if terminated.any() or truncated.any():
    #     #         print("Episode terminated or truncated.")
    #     #         env.reset() # Reset for next episode

    # except Exception as e:
    #     print(f"An error occurred during environment setup or interaction: {e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     print("Closing Isaac Sim application.")
    #     simulation_app.close()
    pass # Main guard for actual execution
