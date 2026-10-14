"""YBT staircase task: terrain description plus the Isaac Gym task configs.

Two things live here, in this order:

1. **Terrain geometry** (``StairsTerrainCfg`` / ``StairsTerrain`` /
   ``TiledStairsTerrain``) — turns metre-valued parameters into a heightfield.
   Nothing in this half uses Isaac Gym.
2. **Task config** (``YBTStairsCfg`` / ``YBTStairsCfgPPO``) — the values the
   training entry point reads. Deliberately a separate task from the flat
   ``ybt`` one, so neither training run can affect the other.

Heightfield convention (inherited from ``isaacgym.terrain_utils``):

    ``height_field_raw`` has shape ``(rows, cols)``
    rows index the **x** axis, cols index the **y** axis
    cell ``(i, j)`` sits at ``x = i * horizontal_scale``, ``y = j * horizontal_scale``
    its integer value is the height, in units of ``vertical_scale``

One staircase climbs along **+x**: a flat approach, ``num_steps`` risers of
``step_height``, then a platform held at the top height.
"""

import numpy as np

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg


# --------------------------------------------------------------------------- #
# Terrain geometry
# --------------------------------------------------------------------------- #
class StairsTerrainCfg:
    """Metric description of a single staircase tile. All lengths in metres."""

    # --- grid resolution ------------------------------------------------- #
    # 0.05 m horizontal keeps the 0.35 m tread on an exact cell boundary. The
    # tiled training map overrides this to 0.10, which is what makes 2048 tiles
    # affordable; `tile_cfg` passes the scale through, so the two stay separate.
    horizontal_scale = 0.05
    vertical_scale = 0.005

    # --- tile extent ----------------------------------------------------- #
    terrain_length = 12.0  # x, the climbing direction
    terrain_width = 8.0    # y

    # --- staircase ------------------------------------------------------- #
    num_steps = 8
    step_height = 0.15     # 15 cm riser; 0 makes the tile flat
    step_depth = 0.35      # tread depth
    approach_depth = 4.0   # flat run-up before the first riser
    platform_depth = 3.0   # flat area on top of the last step
    # Width of the staircase in y. ``None`` makes it span the whole tile.
    stair_width = 3.0

    @classmethod
    def with_overrides(cls, **overrides):
        """A cfg instance with selected fields replaced; unknown names raise."""
        cfg = cls()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError("unknown StairsTerrainCfg field: {!r}".format(key))
            setattr(cfg, key, value)
        return cfg


class StairsTerrain:
    """Builds one staircase heightfield and reports where things are, in tile coords."""

    def __init__(self, cfg=None):
        self.cfg = cfg or StairsTerrainCfg()
        self._validate()
        self._layout()

    # ------------------------------------------------------------------ #
    def _validate(self):
        cfg = self.cfg
        if cfg.horizontal_scale <= 0 or cfg.vertical_scale <= 0:
            raise ValueError("channel scales must be positive")
        if cfg.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if cfg.step_height < 0:
            raise ValueError("step_height must not be negative")
        for name in ("step_depth", "approach_depth", "platform_depth"):
            if getattr(cfg, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        # a riser or a flat run must survive being quantised to one cell,
        # otherwise the feature silently rounds away
        for name in ("step_depth", "approach_depth", "platform_depth"):
            if getattr(cfg, name) / cfg.horizontal_scale < 1:
                raise ValueError("{} is smaller than one heightfield cell".format(name))
        if cfg.step_height / cfg.vertical_scale < 1 and cfg.step_height != 0:
            raise ValueError("step_height is smaller than one heightfield cell")
        needed = cfg.approach_depth + cfg.num_steps * cfg.step_depth
        if needed + cfg.platform_depth > cfg.terrain_length + 1e-9:
            raise ValueError(
                "approach + steps + platform = {:.2f} m does not fit in the "
                "{:.2f} m tile".format(needed + cfg.platform_depth, cfg.terrain_length))
        if cfg.stair_width is not None and not 0 < cfg.stair_width <= cfg.terrain_width:
            raise ValueError("stair_width must be within the tile width or None")

    def _layout(self):
        cfg = self.cfg
        self.rows = int(round(cfg.terrain_length / cfg.horizontal_scale))
        self.cols = int(round(cfg.terrain_width / cfg.horizontal_scale))
        self.tread_cells = int(round(cfg.step_depth / cfg.horizontal_scale))
        self.riser_units = int(round(cfg.step_height / cfg.vertical_scale))
        self.approach_cells = int(round(cfg.approach_depth / cfg.horizontal_scale))
        # the exact cells per tread only matters because rounding would otherwise
        # make the last riser land somewhere unintended
        self.step_depth = self.tread_cells * cfg.horizontal_scale
        self.step_height = self.riser_units * cfg.vertical_scale

        self.x_first_riser = self.approach_cells * cfg.horizontal_scale
        self.x_top = self.x_first_riser + cfg.num_steps * self.step_depth
        self.top_height = cfg.num_steps * self.step_height
        self.stair_width = (cfg.terrain_width if cfg.stair_width is None
                            else cfg.stair_width)
        # staircase centred in y; a full-width staircase simply covers everything
        self.y_center = cfg.terrain_width / 2.0
        self.y_lo = self.y_center - self.stair_width / 2.0
        self.y_hi = self.y_center + self.stair_width / 2.0

        # spawn on the flat run-up, halfway between the tile edge and the first
        # riser, so there is room to see the approach and the climb in one shot
        self.spawn_x = self.x_first_riser / 2.0
        self.spawn_y = self.y_center

    # ------------------------------------------------------------------ #
    def height_field(self):
        """The raw int16 heightfield, shape (rows, cols)."""
        cfg = self.cfg
        hf = np.zeros((self.rows, self.cols), dtype=np.int16)

        # only the staircase strip is touched; everything else stays at z = 0.
        y_lo = int(round(self.y_lo / cfg.horizontal_scale))
        y_hi = int(round(self.y_hi / cfg.horizontal_scale))
        y_hi = min(y_hi, self.cols)  # a full-width staircase rounds one cell over

        x = self.approach_cells
        height = 0
        for _ in range(cfg.num_steps):
            height += self.riser_units
            hf[x:x + self.tread_cells, y_lo:y_hi] = height
            x += self.tread_cells
        hf[x:, y_lo:y_hi] = height  # platform, held to the far edge

        return hf

    def level_at(self, x):
        """Ground height [m] at tile coordinate x, for a point on the staircase."""
        if x < self.x_first_riser:
            return 0.0
        if x >= self.x_top:
            return self.top_height
        step = int((x - self.x_first_riser) // self.step_depth) + 1
        return step * self.step_height

    def describe(self):
        """Human-readable summary, used by the viewer's --headless mode."""
        cfg = self.cfg
        lines = [
            "staircase: {} steps x {:.0f} cm riser, {:.2f} m tread".format(
                cfg.num_steps, self.step_height * 100, self.step_depth),
            "           top platform at {:.2f} m, {:.2f} m deep".format(
                self.top_height, cfg.terrain_length - self.x_top),
            "           {:.2f} m wide, centred at y = {:.2f} m".format(
                self.stair_width, self.y_center),
            "approach : {:.2f} m flat run-up, stairs start at x = {:.2f} m".format(
                self.x_first_riser, self.x_first_riser),
            "tile     : {:.1f} x {:.1f} m at {:.3f} m/cell ({} x {} cells)".format(
                cfg.terrain_length, cfg.terrain_width, cfg.horizontal_scale,
                self.rows, self.cols),
            "spawn    : on the run-up at x = {:.2f} m, y = {:.2f} m, facing +x".format(
                self.spawn_x, self.spawn_y),
        ]
        return "\n".join(lines)

    def profile(self):
        """Ground height sampled every 0.25 m along x, for a quick textual look."""
        xs = np.arange(0.0, self.cfg.terrain_length + 1e-9, 0.25)
        return xs, np.array([self.level_at(x) for x in xs])


def tile_cfg(terrain_cfg, step_height):
    """A :class:`StairsTerrainCfg` for one tile of a tiled map.

    The map's ``terrain`` config carries the tile geometry directly, so training
    never has to build a second config object by hand. ``step_height`` is what
    makes a level (row of tiles) easier or harder; the staircase spans the full
    lane so neighbouring lanes share one continuous run of steps.
    """
    length = (terrain_cfg.approach_depth
              + terrain_cfg.num_steps * terrain_cfg.step_depth
              + terrain_cfg.platform_depth)
    return StairsTerrainCfg.with_overrides(
        horizontal_scale=terrain_cfg.horizontal_scale,
        vertical_scale=terrain_cfg.vertical_scale,
        terrain_length=length,
        terrain_width=terrain_cfg.lane_width,
        num_steps=terrain_cfg.num_steps,
        step_height=step_height,
        step_depth=terrain_cfg.step_depth,
        approach_depth=terrain_cfg.approach_depth,
        platform_depth=terrain_cfg.platform_depth,
        stair_width=None)


class TiledStairsTerrain:
    """A grid of staircase tiles: ``num_levels`` difficulty rows by ``num_lanes`` lanes.

    Every tile is one environment's private piece of ground, so the tile count
    must be at least the environment count -- two robots in one lane would spawn
    inside each other. That budget is what caps ``num_envs`` for this task: the
    total map area, and therefore the triangle count, grows with it.

    Level 0 is flat and the last level is a full-height ``step_height`` riser, so
    the population trains on every difficulty at once. A per-environment
    curriculum is *not* implemented on purpose: with one tile per robot, the
    number of robots on each level is fixed by how many tiles that level has, so
    promoting one robot would mean evicting another. Changing the difficulty
    distribution would need the terrain rebuilt mid-training, which PhysX does
    not support.
    """

    def __init__(self, terrain_cfg):
        self.cfg = terrain_cfg
        self.num_levels = int(terrain_cfg.num_levels)
        self.num_lanes = int(terrain_cfg.num_lanes)
        if self.num_levels < 2:
            raise ValueError("num_levels must be at least 2 (flat + hardest)")
        if self.num_lanes < 1:
            raise ValueError("num_lanes must be at least 1")

        # spread the riser height evenly from flat to the full 15 cm
        self.step_heights = [terrain_cfg.step_height * i / (self.num_levels - 1)
                             for i in range(self.num_levels)]
        self.tiles = [StairsTerrain(tile_cfg(terrain_cfg, h)) for h in self.step_heights]

        first = self.tiles[0]
        if any(t.rows != first.rows or t.cols != first.cols for t in self.tiles):
            raise ValueError("every tile must have the same heightfield shape")
        self.tile_rows, self.tile_cols = first.rows, first.cols
        self.tile_length = first.cfg.terrain_length
        self.lane_width = first.cfg.terrain_width

        # geometry of the staircase, identical in every tile
        self.spawn_x = first.spawn_x
        self.spawn_y = first.spawn_y
        self.x_first_riser = first.x_first_riser
        self.x_top = first.x_top
        self.step_depth = first.step_depth
        self.num_steps = first.cfg.num_steps

    @property
    def num_tiles(self):
        return self.num_levels * self.num_lanes

    def height_field(self):
        """The whole map as one int16 array, shape (levels*rows, lanes*cols)."""
        hf = np.zeros((self.tile_rows * self.num_levels,
                       self.tile_cols * self.num_lanes), dtype=np.int16)
        for level, tile in enumerate(self.tiles):
            # every lane on a level is identical, so build the row once and tile it
            hf[level * self.tile_rows:(level + 1) * self.tile_rows, :] = \
                np.tile(tile.height_field(), (1, self.num_lanes))
        return hf

    def origins(self):
        """Spawn point of every tile, shape (num_levels, num_lanes, 3).

        ``z`` is 0 everywhere: the run-up of every tile sits on the ground plane
        of the heightfield, and only the stairs rise above it.
        """
        origins = np.zeros((self.num_levels, self.num_lanes, 3), dtype=np.float64)
        origins[:, :, 0] = (np.arange(self.num_levels)[:, None] * self.tile_length
                            + self.spawn_x)
        origins[:, :, 1] = (np.arange(self.num_lanes)[None, :] + 0.5) * self.lane_width
        return origins

    def level_step_heights(self):
        """The riser height of each level, as a numpy array of floats."""
        return np.array(self.step_heights, dtype=np.float64)

    def describe(self):
        hf = self.height_field()
        lines = [
            "map      : {} levels x {} lanes = {} tiles, {} x {:.1f} m".format(
                self.num_levels, self.num_lanes, self.num_tiles,
                self.num_levels * self.tile_length, self.num_lanes * self.lane_width),
            "tile     : {:.1f} m long x {:.2f} m wide ({} x {} cells at {:.2f} m)".format(
                self.tile_length, self.lane_width, self.tile_rows, self.tile_cols,
                self.cfg.horizontal_scale),
            "risers   : {:.0f} cm (level 0, flat) .. {:.0f} cm (level {})".format(
                self.step_heights[0] * 100, self.step_heights[-1] * 100,
                self.num_levels - 1),
            "steps    : {} x {:.2f} m tread, top at {:.2f} m; spawn {:.2f} m before "
            "the first riser".format(self.num_steps, self.step_depth,
                                     self.num_steps * self.step_heights[-1], self.spawn_x),
            "heightfield: {} x {} cells, {:.1f} MB int16".format(
                hf.shape[0], hf.shape[1], hf.size * 2 / 1e6),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Task config
# --------------------------------------------------------------------------- #
class YBTStairsCfg(YBTFlatCfg):
    """Staircase task for YBT. Separate from ``YBTFlatCfg`` in every way that matters."""

    class env(YBTFlatCfg.env):
        # num_levels * num_lanes tiles, one environment each. See TiledStairsTerrain
        # for why this cannot be raised to the flat task's 12288.
        # 4096 measured at 1.07-1.27 s/iteration and 4.5 GB of GPU memory, about
        # 82k samples/s; 2048 is 0.75-0.82 s and about 61k samples/s.
        num_envs = 4096
        episode_length_s = 20.0

    class terrain(YBTFlatCfg.terrain):
        mesh_type = "trimesh"
        horizontal_scale = 0.10
        vertical_scale = 0.005
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # spelled out here rather than reusing the base class's misspelled
        # `slope_treshold`; risers above this slope are snapped to vertical so a
        # step is a step and not a one-cell ramp
        slope_threshold = 1.5

        # --- staircase tile (mirrors StairsTerrainCfg; keep the two in step) ---
        num_steps = 8
        step_height = 0.15   # the hardest level's riser, in metres
        step_depth = 0.30
        approach_depth = 0.9
        platform_depth = 0.9
        lane_width = 1.20    # one robot's private lane

        # --- map grid ---
        # `num_levels` spreads the difficulty; `num_lanes` must be at least
        # `num_envs / num_levels`, because each robot needs a tile of its own.
        # Raising num_envs alone therefore does nothing but raise an error --
        # move the two together. Each extra lane widens the map by `lane_width`
        # and adds roughly 1.5 M triangles.
        num_levels = 16      # risers from 0 to step_height, one tile row each
        num_lanes = 256      # tiles per level; num_levels*num_lanes >= num_envs
        terminate_at_top = True  # end the episode once the last riser is cleared

    class commands(YBTFlatCfg.commands):
        # The task is "walk forward up the stairs", so the command is always
        # forward. A random heading would send the robot off its 1.2 m lane, and
        # a random heading *rate* would fight the heading it is already holding.
        heading_command = True
        curriculum = False

        class ranges(YBTFlatCfg.commands.ranges):
            lin_vel_x = [0.2, 0.6]   # always towards the stairs
            lin_vel_y = [0.0, 0.0]   # no strafing: the lane is only 1.2 m wide
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]     # hold +x

    class init_state(YBTFlatCfg.init_state):
        # Isaac Gym scatters actors by a metre at creation time, which would drop
        # a robot into its neighbour's lane. The stairs env re-places everyone
        # explicitly and uses this much smaller spread instead.
        spawn_noise_xy = 0.15

    class domain_rand(YBTFlatCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = False
        # a 0.5 m/s shove is enough to push a robot off a 1.2 m lane, which
        # teaches "fall off the stairs" rather than "recover"
        push_robots = False

    class rewards(YBTFlatCfg.rewards):
        base_height_target = 0.50
        max_contact_force = 500.0
        soft_dof_pos_limit = 0.9
        # The flat task lifts a swing foot 5 cm clear; a 15 cm riser needs the
        # foot to come up over the step or the robot walks into it.
        feet_clearance_target = 0.17
        feet_clearance_cap = 0.27
        # The inherited 1.0 s target is longer than a step period on 0.30 m
        # treads (~0.25 s of air per foot at 0.5 m/s), so the term would come out
        # negative at every touchdown and pay the robot to slow down. Match it to
        # a stair gait instead.
        feet_air_time_positive_threshold = 0.2

        class scales(YBTFlatCfg.rewards.scales):
            torques = -0.00001
            orientation = -1.0
            base_height = -1.0
            dof_pos_limits = -10.0
            feet_clearance = 0.3
            feet_air_time = 0.5
            # Zero on the flat task: a foot hitting the *side* of a riser rather
            # than the top of it. The base cfg calls this knob `feet_stumble`, but
            # non-zero scales are looked up as `_reward_<name>` and no
            # `_reward_feet_stumble` exists -- the function is `_reward_stumble`,
            # so that is the name that has to carry the weight.
            stumble = -0.2

    class sim(YBTFlatCfg.sim):
        dt = 0.005


class YBTStairsCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "stairs_ybt"
        run_name = ""
        num_steps_per_env = 24
        max_iterations = 2000
        save_interval = 50
        resume = False
