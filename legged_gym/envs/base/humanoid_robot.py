
import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.envs.base.legged_robot import LeggedRobot


class HumanoidRobot(LeggedRobot):
    """Shared base class for all bipedal humanoid robots (H1, H1_2, G1).

    Extends LeggedRobot with:
    - Rigid-body foot state tracking (position, velocity)
    - Phase-based gait cycle (sinusoidal left/right leg phases)
    - Phase-aware observation construction
    - Humanoid-specific reward functions

    Subclasses must define `hip_indices` — a list of DOF indices
    corresponding to the hip-abduction/adduction joints on that robot.
    Those indices are used by _reward_hip_pos() to penalise lateral hip
    excursions.

    Example::

        class H1Robot(HumanoidRobot):
            hip_indices = [0, 1, 5, 6]
    """

    # Subclasses set this to the DOF indices of the hip joints.
    hip_indices: list = []

    # ---------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    # ---------------------------------------------------------------
    # Per-step callbacks
    # ---------------------------------------------------------------

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat(
            [self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1
        )

        return super()._post_physics_step_callback()

    # ---------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------

    def _get_noise_scale_vec(self, cfg):
        """Returns the per-element noise scale vector for the observation buffer.

        [NOTE]: Must be adapted when the observation structure changes.
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9 : 9 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[9 + self.num_actions : 9 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = 0.0  # sin/cos phase
        return noise_vec

    def compute_observations(self):
        """Assembles the observation and privileged-observation buffers."""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ---------------------------------------------------------------
    # Reward functions
    # ---------------------------------------------------------------

    def _reward_contact(self):
        """Reward correct foot contacts given the current gait phase."""
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        """Penalise feet that deviate from the target swing height (0.08 m)."""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_alive(self):
        """Constant reward for remaining alive (not terminated)."""
        return 1.0

    def _reward_contact_no_vel(self):
        """Penalise foot velocity while the foot is in contact with the ground."""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        """Penalise lateral/abduction hip joint displacement.

        Uses ``self.hip_indices`` which subclasses must define to specify
        which DOF indices correspond to the hip joints on that robot.
        """
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices]), dim=1)
