"""Staircase terrain/config/reward regression checks. Run in the unitree-rl environment.

These tests do not build a simulation: they exercise the analytic geometry, the
task config, and the reward terms that were rewritten for stepped ground. The
environment itself is checked by running the task -- see
``doc/ybt_stairs_training_zh.md``.
"""
import os
import unittest

import isaacgym  # must precede torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg
from legged_gym.envs.ybt.ybt_env import validate_ybt_asset
from legged_gym.envs.ybt.ybt_stairs_config import (
    StairsTerrain, StairsTerrainCfg, TiledStairsTerrain, YBTStairsCfg,
    YBTStairsCfgPPO)
from legged_gym.envs.ybt.ybt_stairs_env import YBTStairsRobot


class StairsTerrainTests(unittest.TestCase):
    """The single-tile geometry everything else is built from."""

    def setUp(self):
        self.cfg = StairsTerrainCfg()
        self.terrain = StairsTerrain(self.cfg)

    def test_layout_matches_the_configured_dimensions(self):
        t = self.terrain
        self.assertAlmostEqual(t.x_first_riser, self.cfg.approach_depth)
        self.assertAlmostEqual(
            t.x_top, self.cfg.approach_depth
            + self.cfg.num_steps * self.cfg.step_depth)
        self.assertAlmostEqual(
            t.top_height, self.cfg.num_steps * self.cfg.step_height)
        # flat run-up, steps in between, flat platform on top
        self.assertEqual(t.level_at(0.0), 0.0)
        self.assertAlmostEqual(t.level_at(t.x_first_riser + 1e-9),
                               self.cfg.step_height)
        self.assertAlmostEqual(t.level_at(t.x_top - 1e-9), t.top_height)
        self.assertAlmostEqual(t.level_at(self.cfg.terrain_length), t.top_height)

    def test_height_field_encodes_the_profile_exactly(self):
        # reconstruct the ground profile from the array and compare it against
        # the analytic one: this is what catches a slipped riser loop
        hf = self.terrain.height_field()
        self.assertEqual(hf.dtype, np.int16)
        self.assertEqual(hf.shape, (self.terrain.rows, self.terrain.cols))
        step = self.cfg.horizontal_scale
        # sample inside the staircase strip; the ground either side of it is
        # deliberately flat at every x
        y_lo = int(round(self.terrain.y_lo / step))
        y_hi = int(round(self.terrain.y_hi / step)) - 1
        # and sample cell centres, not edges: a riser sits exactly on a cell
        # boundary, so probing there asks which side of a tie a float rounds to
        for i in range(self.terrain.rows):
            x = (i + 0.5) * step
            for y in (y_lo, (y_lo + y_hi) // 2, y_hi):
                self.assertAlmostEqual(hf[i, y] * self.cfg.vertical_scale,
                                       self.terrain.level_at(x), places=9,
                                       msg="cell {} (x = {:.3f})".format(i, x))

    def test_quantisation_is_exact(self):
        # the analytic reward geometry is only reproducible if every configured
        # length lands on a whole number of cells; snapping is fine, drift is not
        t = self.terrain
        self.assertAlmostEqual(t.step_depth, self.cfg.step_depth)
        self.assertAlmostEqual(t.step_height, self.cfg.step_height)
        self.assertAlmostEqual(t.x_first_riser, self.cfg.approach_depth)

    def test_flat_tile_is_allowed(self):
        # level 0 of the map is flat, so step_height == 0 has to be legal
        flat = StairsTerrain(StairsTerrainCfg.with_overrides(step_height=0.0))
        self.assertEqual(flat.top_height, 0.0)
        self.assertFalse(flat.height_field().any())

    def test_narrow_staircase_leaves_flat_ground_either_side(self):
        terrain = StairsTerrain(StairsTerrainCfg.with_overrides(
            terrain_width=4.0, stair_width=2.0))
        hf = terrain.height_field()
        self.assertFalse(hf[:, 0].any())
        self.assertFalse(hf[:, -1].any())
        self.assertEqual(hf[:, hf.shape[1] // 2].max(),
                         self.cfg.num_steps * terrain.riser_units)

    def test_invalid_parameters_are_rejected(self):
        bad = {
            "num_steps": 0,
            "step_height": -0.1,
            "step_depth": 0.0,
            "approach_depth": -1.0,
            "platform_depth": 0.0,
        }
        for key, value in bad.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    StairsTerrain(StairsTerrainCfg.with_overrides(**{key: value}))
        for key, value in (("step_height", 0.001),   # smaller than one cell
                           ("step_depth", 0.01),
                           ("stair_width", 99.0)):   # wider than the tile
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    StairsTerrain(StairsTerrainCfg.with_overrides(**{key: value}))
        with self.assertRaises(ValueError):          # does not fit in the tile
            StairsTerrain(StairsTerrainCfg.with_overrides(
                approach_depth=10.0, terrain_length=12.0))

    def test_with_overrides_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            StairsTerrainCfg.with_overrides(num_risers=9)


class TiledStairsTerrainTests(unittest.TestCase):
    """The map the trainer lays down."""

    def setUp(self):
        self.cfg = YBTStairsCfg.terrain()
        self.map = TiledStairsTerrain(self.cfg)

    def test_map_shape_and_lane_grid(self):
        hf = self.map.height_field()
        self.assertEqual(hf.shape, (self.map.tile_rows * self.cfg.num_levels,
                                    self.map.tile_cols * self.cfg.num_lanes))
        origins = self.map.origins()
        self.assertEqual(origins.shape,
                         (self.cfg.num_levels, self.cfg.num_lanes, 3))
        ys = origins[0, :, 1]
        self.assertTrue(np.allclose(np.diff(ys), self.cfg.lane_width))
        self.assertGreater(ys[0], 0.0)
        self.assertLess(ys[-1], self.cfg.num_lanes * self.cfg.lane_width)
        xs = origins[:, 0, 0]
        self.assertTrue(np.allclose(np.diff(xs), self.map.tile_length))

    def test_every_spawn_stands_on_flat_ground(self):
        hf = self.map.height_field()
        origins = self.map.origins()
        for level in range(self.cfg.num_levels):
            for lane in (0, self.cfg.num_lanes // 2, self.cfg.num_lanes - 1):
                x, y = origins[level, lane, :2]
                cell = hf[int(round(x / self.cfg.horizontal_scale)),
                          int(round(y / self.cfg.horizontal_scale))]
                self.assertEqual(cell, 0,
                                 "spawn of level {} lane {} is not on the run-up"
                                 .format(level, lane))

    def test_difficulty_spread_runs_from_flat_to_full_height(self):
        heights = self.map.level_step_heights()
        self.assertEqual(len(heights), self.cfg.num_levels)
        self.assertAlmostEqual(heights[0], 0.0)
        self.assertAlmostEqual(heights[-1], self.cfg.step_height)
        self.assertTrue(np.all(np.diff(heights) > 0))

    def test_tile_budget_covers_the_configured_environments(self):
        self.assertGreaterEqual(self.map.num_tiles, YBTStairsCfg.env.num_envs)

    def test_describe_is_text(self):
        self.assertIn("levels", self.map.describe())


class _Stub(object):
    """Enough of an environment for the plain-tensor reward terms.

    The rewritten rewards only read ``root_states``/``rigid_body_states``/
    ``contact_forces``, so they can be exercised without a simulation.
    """

    _local_ground_heights = YBTStairsRobot._local_ground_heights
    _reward_base_height = YBTStairsRobot._reward_base_height
    _reward_feet_clearance = YBTStairsRobot._reward_feet_clearance
    _reward_feet_air_time = YBTStairsRobot._reward_feet_air_time
    _past_top_of_stairs = YBTStairsRobot._past_top_of_stairs

    def __init__(self, step_height):
        self.cfg = YBTStairsCfg()
        self.stairs = TiledStairsTerrain(self.cfg.terrain)
        self.feet_indices = torch.arange(4)
        self.env_step_height = torch.tensor([step_height])
        # tile 0 of level 0, i.e. the map origin sits at the run-up spawn point
        self.env_origins = torch.tensor(
            [[self.stairs.spawn_x, self.stairs.spawn_y, 0.0]])
        self.dt = 0.02
        self.commands = torch.tensor([[0.4, 0.0, 0.0]])
        self.root_states = torch.zeros(1, 13)
        self.rigid_body_states = torch.zeros(1, 4, 13)
        self.contact_forces = torch.zeros(1, 4, 3)
        self.last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        self.feet_air_time = torch.zeros(1, 4)

    def local_x_to_world(self, local_x):
        """World x of a point standing local_x into its tile, as the env defines it."""
        return float(self.env_origins[0, 0]) + local_x - self.stairs.spawn_x

    def place(self, local_x):
        """Stand the trunk and all four feet local_x into the tile.

        Returns the ground height there, read back through the same call the
        rewards make -- so a test that compares against this value is comparing
        against the reward's own number rather than a second, differently
        rounded derivation of it.
        """
        world_x = self.local_x_to_world(local_x)
        self.root_states[0, 0] = world_x
        self.rigid_body_states[:, :, 0] = world_x
        return float(self._local_ground_heights(self.root_states[:, 0:1]))


class StairsRewardTests(unittest.TestCase):
    """The reward terms that assumed z = 0 was the ground."""

    def test_local_ground_height_follows_the_staircase(self):
        # the reference profile comes from the level-0 tile, scaled by each
        # level's riser height
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        hardest = stairs.tiles[-1]
        self.assertAlmostEqual(hardest.step_height, stairs.cfg.step_height)
        for step_height in (0.0, 0.05, 0.15):
            stub = _Stub(step_height)
            # offset off the cell boundary so a riser never lands exactly on a
            # sample and the comparison cannot hinge on a float tie
            for local_x in np.arange(0.017, stairs.tile_length, 0.05):
                steps = round(hardest.level_at(local_x) / hardest.step_height)
                self.assertAlmostEqual(
                    stub.place(local_x), steps * step_height, places=6,
                    msg="local_x = {:.2f}".format(local_x))

    def test_base_height_is_measured_from_the_step_under_the_trunk(self):
        # a trunk 0.5 m above the step it stands on is a perfect score, on the
        # run-up, on a mid-stair step, and on the platform of a 15 cm staircase
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        for step_height in (0.0, 0.15):
            for local_x in np.arange(0.017, stairs.tile_length, 0.25):
                stub = _Stub(step_height)
                ground = stub.place(local_x)
                stub.root_states[0, 2] = (ground
                                          + stub.cfg.rewards.base_height_target)
                self.assertAlmostEqual(float(stub._reward_base_height()), 0.0,
                                       places=9,
                                       msg="local_x = {:.2f}".format(local_x))

    def test_absolute_base_height_would_have_punished_the_climb(self):
        # guard the reason for the override: standing on the platform, the
        # inherited version is off by the whole climb height
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        top = stairs.num_steps * stairs.cfg.step_height
        self.assertAlmostEqual(stairs.tiles[-1].top_height, top)
        self.assertGreater(top, 0.5)  # bigger than the 0.5 m target itself

    def test_foot_clearance_is_measured_from_the_step_under_the_foot(self):
        stairs = TiledStairsTerrain(YBTStairsCfg.terrain())
        stub = _Stub(0.15)
        # foot resting on the third step: raised ground, but no lift at all
        ground = stub.place(stairs.x_first_riser + 2.5 * stairs.step_depth)
        self.assertGreater(ground, 0.0)  # it really is on a raised step
        stub.rigid_body_states[:, :, 2] = ground
        self.assertLess(float(stub._reward_feet_clearance()), 1e-9)
        # and a foot lifted clear of that step does score
        stub.rigid_body_states[:, :, 2] = ground + 0.30
        self.assertGreater(float(stub._reward_feet_clearance()), 0.0)

    def test_feet_air_time_rebases_on_a_stair_length_stride(self):
        stub = _Stub(0.15)
        stub.feet_air_time[:] = 0.3       # a step period on 0.30 m treads
        stub.contact_forces[:, 0, 2] = 10.0  # FR just landed
        reward = float(stub._reward_feet_air_time())
        threshold = stub.cfg.rewards.feet_air_time_positive_threshold
        self.assertAlmostEqual(reward, 0.3 + stub.dt - threshold, places=6)
        self.assertGreater(reward, 0.0)   # the inherited 0.5 s would be negative


class StairsEnvContractTests(unittest.TestCase):
    """What the registry, the PPO runner and the flat task rely on."""

    def test_task_is_registered(self):
        self.assertIn("ybt_stairs", task_registry.task_classes)
        self.assertIs(task_registry.get_task_class("ybt_stairs"), YBTStairsRobot)
        _, ppo_cfg = task_registry.get_cfgs("ybt_stairs")
        self.assertIsInstance(ppo_cfg.runner, YBTStairsCfgPPO.runner)

    def test_flat_task_still_refuses_anything_but_a_plane(self):
        cfg = YBTFlatCfg()
        self.assertEqual(cfg.terrain.mesh_type, "plane")
        self.assertEqual(cfg.env.num_envs, 12288)
        # YBTFlatCfg.terrain is a class, so a plain attribute write would mutate
        # it for every later test in this process; subclass it instead
        cfg.terrain = type("terrain", (YBTFlatCfg.terrain,),
                           {"mesh_type": "trimesh"})
        with self.assertRaises(ValueError):
            validate_ybt_asset(cfg)
        self.assertEqual(YBTFlatCfg.terrain.mesh_type, "plane")
        # the staircase task passes the same validation with the flag off
        validate_ybt_asset(YBTStairsCfg(), flat_ground_only=False)

    def test_every_non_zero_reward_scale_names_a_real_function(self):
        # non-zero scales are resolved as getattr(env, "_reward_" + name), so a
        # name with no function is an AttributeError at construction time -- the
        # base cfg ships exactly such a name (`feet_stumble`) at zero scale
        scales = YBTStairsCfg().rewards.scales
        for name, scale in vars(scales).items():
            if name.startswith("_") or not isinstance(scale, (int, float)):
                continue
            if scale == 0:
                continue
            with self.subTest(scale=name):
                self.assertTrue(
                    hasattr(YBTStairsRobot, "_reward_" + name),
                    "reward scale '{}' has no _reward_{} implementation".format(
                        name, name))

    def test_stairs_logs_do_not_collide_with_the_flat_run(self):
        self.assertEqual(YBTStairsCfgPPO.runner.experiment_name, "stairs_ybt")
        self.assertNotEqual(YBTStairsCfgPPO.runner.experiment_name, "flat_ybt")
        self.assertFalse(YBTStairsCfgPPO.runner.resume)
        self.assertNotEqual(
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs",
                         YBTStairsCfgPPO.runner.experiment_name),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "flat_ybt"))

    def test_stairs_asks_for_more_clearance_than_flat(self):
        self.assertGreater(YBTStairsCfg().rewards.feet_clearance_target,
                           YBTFlatCfg().rewards.feet_clearance_target)

    def test_tile_count_bounds_the_environment_count(self):
        cfg = YBTStairsCfg()
        self.assertLessEqual(cfg.env.num_envs,
                             cfg.terrain.num_levels * cfg.terrain.num_lanes)

    def test_termination_flags_a_robot_past_the_last_riser(self):
        stub = _Stub(0.15)
        stairs = stub.stairs
        stub.root_states[0, 0] = stub.local_x_to_world(stairs.x_top - 0.1)
        self.assertFalse(bool(stub._past_top_of_stairs()))
        stub.root_states[0, 0] = stub.local_x_to_world(stairs.x_top + 0.1)
        self.assertTrue(bool(stub._past_top_of_stairs()))

    def test_stairs_robot_extends_the_flat_one(self):
        # the staircase task reuses the validated URDF, safe reset and rigid-body
        # bookkeeping of the flat task rather than reimplementing them
        from legged_gym.envs.ybt.ybt_env import YBTRobot
        self.assertTrue(issubclass(YBTStairsRobot, YBTRobot))
        self.assertTrue(issubclass(YBTStairsRobot, LeggedRobot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
