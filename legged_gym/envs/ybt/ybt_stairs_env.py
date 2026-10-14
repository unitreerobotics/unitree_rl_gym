"""YBT staircase environment: a tiled staircase instead of a flat ground plane.

Inherits ``YBTRobot`` (and through it ``LeggedRobot``) and overrides only what
the change of ground actually breaks:

``create_sim``            builds the staircase trimesh instead of a plane
``_get_env_origins``      gives every environment its own tile, at one difficulty
``_reset_root_states``    drops the ground-scattering of the base implementation
``_reward_feet_clearance``/``_reward_base_height``  measure against the local step
``_reward_feet_air_time`` shapes a stair-length gait, not a flat-ground stride
``check_termination``     ends the episode when the last riser is cleared

The flat ``ybt`` task is untouched: ``ybt_env.py`` keeps its behaviour and this
module is only reached through its own registered task name.
"""

import torch
from isaacgym import gymapi, gymtorch, terrain_utils
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_env import YBTRobot, validate_ybt_asset
from legged_gym.envs.ybt.ybt_stairs_config import TiledStairsTerrain


class YBTStairsRobot(YBTRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        validate_ybt_asset(cfg, flat_ground_only=False)
        LeggedRobot.__init__(self, cfg, sim_params, physics_engine, sim_device,
                             headless)
        # At creation time the actors were scattered by up to a metre and left at
        # z = 0 (see LeggedRobot._create_envs). Both are wrong here -- a metre of
        # scatter puts a robot in the next lane, and z = 0 buries it in the run-up
        # -- so re-place everyone through the normal reset path before the first
        # step. reset_idx is not called at init by the base class.
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ #
    # Simulation and terrain
    # ------------------------------------------------------------------ #
    def create_sim(self):
        """Create the sim with the staircase as the ground, then the envs."""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self._create_terrain_mesh()
        self._create_envs()

    def _create_terrain_mesh(self):
        """Turn the tiled heightfield into one static triangle mesh.

        The mesh is *not* shifted to the world origin: every tile's spawn point
        has to line up with the triangle it stands on, so tile coordinates and
        world coordinates are kept identical and ``origins()`` is the single
        source of truth for both.
        """
        cfg = self.cfg.terrain
        self.stairs = TiledStairsTerrain(cfg)
        if self.num_envs > self.stairs.num_tiles:
            # the usual cause is `--num_envs` on the command line, which changes
            # the environment count without touching the map grid
            raise ValueError(
                "ybt_stairs needs one terrain tile per environment, got {} envs "
                "for {} tiles (num_levels {} x num_lanes {}). Raise "
                "YBTStairsCfg.terrain.num_lanes to at least {} to match."
                .format(self.num_envs, self.stairs.num_tiles, cfg.num_levels,
                        cfg.num_lanes,
                        -(-self.num_envs // cfg.num_levels)))

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.stairs.height_field(),
            horizontal_scale=cfg.horizontal_scale,
            vertical_scale=cfg.vertical_scale,
            slope_threshold=cfg.slope_threshold)

        params = gymapi.TriangleMeshParams()
        params.nb_vertices = vertices.shape[0]
        params.nb_triangles = triangles.shape[0]
        params.static_friction = cfg.static_friction
        params.dynamic_friction = cfg.dynamic_friction
        params.restitution = cfg.restitution
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(),
                                   triangles.flatten(), params)

        self.terrain_origins = torch.from_numpy(self.stairs.origins()).to(
            self.device).float()
        self.level_step_height = torch.from_numpy(
            self.stairs.level_step_heights()).to(self.device).float()
        print("stairs terrain: {} vertices, {} triangles".format(
            vertices.shape[0], triangles.shape[0]))
        print(self.stairs.describe())

    def _get_env_origins(self):
        """Bind each environment to its own tile: one robot per staircase."""
        self.custom_origins = True
        num_lanes = self.cfg.terrain.num_lanes
        # One tile each, drawn without replacement. With as many tiles as robots
        # every level ends up with exactly num_lanes robots, which is the point:
        # the population trains on all the riser heights at the same time and no
        # two robots ever share ground.
        tile_ids = torch.randperm(self.stairs.num_tiles, device=self.device)
        tile_ids = tile_ids[:self.num_envs]
        self.terrain_levels = torch.div(tile_ids, num_lanes,
                                        rounding_mode="floor").long()
        self.terrain_types = (tile_ids % num_lanes).long()
        self.env_origins = self.terrain_origins[self.terrain_levels,
                                                self.terrain_types].clone()
        self.env_step_height = self.level_step_height[self.terrain_levels]

    # ------------------------------------------------------------------ #
    # Resets
    # ------------------------------------------------------------------ #
    def _reset_root_states(self, env_ids):
        """Place the trunk on its own tile. Same as the base, minus the ±1 m."""
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # a lane is only 1.2 m wide, so the spawn scatter has to stay well inside
        # half of that; the base class uses a metre and would overlap neighbours
        noise = self.cfg.init_state.spawn_noise_xy
        self.root_states[env_ids, :2] += torch_rand_float(
            -noise, noise, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def set_camera(self, position=None, lookat=None):
        """Aim at environment 0's staircase rather than at a fixed point.

        The map is tens of metres across and a robot's tile is picked at random,
        so the base config's fixed camera pose would point at empty ground.
        """
        tile = self.env_origins[0]
        look = (float(tile[0]) - self.stairs.spawn_x + self.stairs.x_top / 2.0,
                float(tile[1]), 0.5)
        super().set_camera((look[0] - 4.5, look[1] - 3.2, look[2] + 2.4), look)

    # ------------------------------------------------------------------ #
    # Geometry queries
    # ------------------------------------------------------------------ #
    def _local_ground_heights(self, world_x):
        """Height of the step under the given world x, one value per x.

        The terrain is built from an analytic staircase and every tile has the
        same layout, so which step a point stands on follows from its offset
        within its tile. That avoids a raycast per foot per step. Works for
        ``(num_envs,)`` and ``(num_envs, k)`` input alike.
        """
        local_x = world_x - self.env_origins[:, 0:1] + self.stairs.spawn_x
        # step index, 0 on the run-up. Clamping covers both ends of the climb:
        # in front of the first riser the index would go negative, and past the
        # last one it would keep growing, but the tile is flat there.
        steps = torch.floor(
            (local_x - self.stairs.x_first_riser) / self.stairs.step_depth) + 1.0
        steps = steps.clamp(min=0.0, max=float(self.stairs.num_steps))
        return steps * self.env_step_height.unsqueeze(-1)

    def _past_top_of_stairs(self):
        local_x = (self.root_states[:, 0] - self.env_origins[:, 0]
                   + self.stairs.spawn_x)
        return local_x > self.stairs.x_top

    # ------------------------------------------------------------------ #
    # Rewards that a raised, stepped ground breaks
    # ------------------------------------------------------------------ #
    def _reward_feet_clearance(self):
        """Swing-foot clearance measured above the step the foot is over.

        ``YBTRobot`` measures the foot's world z, which is right on a plane but
        would pay full marks for standing still on a raised step, and would never
        ask a foot to clear the next riser. Measuring relative to the local step
        makes 15 cm of lift pay the same on level 0 and on level 15.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_x = self.rigid_body_states[:, self.feet_indices, 0]
        foot_z = self.rigid_body_states[:, self.feet_indices, 2]
        ground = self._local_ground_heights(foot_x)
        target = self.cfg.rewards.feet_clearance_target
        cap = self.cfg.rewards.feet_clearance_cap
        clearance = ((foot_z - ground - target) / (cap - target)).clip(min=0., max=1.)
        return torch.sum(clearance * (~contact), dim=1)

    def _reward_base_height(self):
        """Trunk height above the step it is standing on, not above z = 0.

        The inherited version squares ``root_states[:, 2]`` against a fixed
        target, so climbing a 1.2 m staircase would be punished as an error of
        more than a metre. Both versions agree on level 0.
        """
        ground = self._local_ground_heights(self.root_states[:, 0:1]).squeeze(-1)
        return torch.square(self.root_states[:, 2] - ground
                            - self.cfg.rewards.base_height_target)

    def _reward_feet_air_time(self):
        """Air time above ``feet_air_time_positive_threshold`` at each touchdown.

        Same shape as the base reward, but the 0.5 s reference is a flat-ground
        stride. On 0.30 m treads at the commanded speeds a foot is off the ground
        for roughly a quarter second, so the hard-coded reference would score
        every step negatively and pay the robot to slow down.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        threshold = self.cfg.rewards.feet_air_time_positive_threshold
        reward = torch.sum((self.feet_air_time - threshold) * first_contact, dim=1)
        reward *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return reward

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def check_termination(self):
        """Resets the base conditions plus "reached the top of the staircase"."""
        super().check_termination()
        if self.cfg.terrain.terminate_at_top:
            # the staircase is a fixed-length climb: once the last riser is
            # behind the robot it has finished the task, and there is nothing
            # beyond the platform but the next tile's run-up
            self.reset_buf |= self._past_top_of_stairs()
