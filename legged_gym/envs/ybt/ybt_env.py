"""YBT-specific validation and safe reset; observations/PD/rewards reuse LeggedRobot."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot


def validate_ybt_asset(cfg, flat_ground_only=True):
    """Fail early on missing meshes, mismatched joints or an invalid standing pose.

    ``flat_ground_only`` is what the flat ``ybt`` task asserts about itself: its
    rewards measure heights against z = 0, which is only true on a ground plane.
    The staircase task passes ``False`` because it has replaced those rewards.
    """
    path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("YBT URDF is missing or empty: {}".format(path))
    root = ET.parse(str(path)).getroot()
    for mesh in root.findall(".//mesh"):
        mesh_path = path.parent / mesh.attrib["filename"]
        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise FileNotFoundError("YBT mesh is missing or empty: {}".format(mesh_path))
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")
              if joint.attrib["type"] != "fixed"}
    if len(joints) != 12 or set(joints) != set(cfg.init_state.default_joint_angles):
        raise ValueError("YBT requires 12 movable joints matching default_joint_angles")
    if cfg.env.num_actions != 12 or cfg.env.num_observations != 48:
        raise ValueError("YBT uses 12 actions and the base environment's 48 observations")
    if flat_ground_only and cfg.terrain.mesh_type != "plane":
        raise ValueError("The YBT task currently implements flat-ground training only")
    if not math.isfinite(cfg.init_state.joint_angle_noise) or cfg.init_state.joint_angle_noise < 0:
        raise ValueError("joint_angle_noise must be a finite nonnegative number")
    for name, joint in joints.items():
        limit = joint.find("limit")
        angle = cfg.init_state.default_joint_angles[name]
        if (joint.attrib["type"] != "revolute" or limit is None
                or not math.isfinite(angle)
                or not float(limit.attrib["lower"]) < angle < float(limit.attrib["upper"])):
            raise ValueError("YBT default angle is outside the URDF limits: {}".format(name))
        gains = [key for key in cfg.control.stiffness if key in name]
        if (len(gains) != 1 or cfg.control.stiffness[gains[0]] <= 0
                or cfg.control.damping.get(gains[0], -1) < 0):
            raise ValueError("YBT joint needs one valid stiffness/damping pair: {}".format(name))
    links = {link.attrib["name"] for link in root.findall("link")}
    feet = {name for name in links if cfg.asset.foot_name in name}
    if feet != {leg + "_foot" for leg in ("FR", "FL", "RR", "RL")} or "trunk" not in links:
        raise ValueError("YBT requires the trunk and four named foot links")
    return path


class YBTRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        validate_ybt_asset(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        super()._create_envs()
        if self.num_dof != self.num_actions or set(self.dof_names) != set(self.cfg.init_state.default_joint_angles):
            raise ValueError("Isaac Gym loaded unexpected YBT DOFs: {}".format(self.dof_names))
        if self.feet_indices.numel() != 4 or torch.any(self.feet_indices < 0):
            raise ValueError("Isaac Gym must retain four YBT foot bodies")
        if self.termination_contact_indices.numel() != 1 or torch.any(self.termination_contact_indices < 0):
            raise ValueError("YBT termination contact must identify the trunk")

    def _init_buffers(self):
        super()._init_buffers()
        # The clearance reward needs the world position of the four foot bodies.
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # Refresh before the rewards are computed in post_physics_step.
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _reward_feet_clearance(self):
        # Reward swing feet for clearing the ground. Clamped at >= 0 so a foot
        # dragged along the floor (reward 0) is never better than one lifted off
        # it; the base air-time reward had the opposite incentive, which the
        # policy exploited by dragging its legs.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_z = self.rigid_body_states[:, self.feet_indices, 2]
        target = self.cfg.rewards.feet_clearance_target
        cap = self.cfg.rewards.feet_clearance_cap
        clearance = ((foot_z - target) / (cap - target)).clip(min=0., max=1.)
        return torch.sum(clearance * (~contact), dim=1)

    def _reset_dofs(self, env_ids):
        # YBT knees have positive-only limits. Multiplying 1.6 by 0.5 can
        # produce 0.8 < the URDF lower limit (1.0297), so use additive noise.
        noise = self.cfg.init_state.joint_angle_noise
        positions = self.default_dof_pos + torch_rand_float(
            -noise, noise, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = torch.maximum(
            torch.minimum(positions, self.dof_pos_limits[:, 1]),
            self.dof_pos_limits[:, 0])
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
