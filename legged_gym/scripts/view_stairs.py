"""Build the YBT staircase terrain and open it in the Isaac Gym viewer.

Two terrains can be shown, both from ``legged_gym/envs/ybt/ybt_stairs_config.py``:

* one 15 cm staircase tile (the default) -- flat run-up, risers, platform, so the
  step height can be judged against the real robot;
* with ``--map``, the tiled training map: ``num_levels`` rows of tiles running
  from flat up to 15 cm, each row ``num_lanes`` robots wide. This is exactly the
  geometry ``ybt_stairs`` trains on, and it is the one to look at when checking
  that a robot's lane is clear and that its tile is the difficulty it should be.

Usage:
    python legged_gym/scripts/view_stairs.py
    python legged_gym/scripts/view_stairs.py --headless              # report only
    python legged_gym/scripts/view_stairs.py --terrain-only          # no robot
    python legged_gym/scripts/view_stairs.py --num-steps=4 --step-height=0.10
    python legged_gym/scripts/view_stairs.py --stair-width=none      # full-width stairs
    python legged_gym/scripts/view_stairs.py --dynamic               # let it fall/settle
    python legged_gym/scripts/view_stairs.py --map --map-level=15    # hardest level
    python legged_gym/scripts/view_stairs.py --map --map-lanes=4     # a narrow slice

Run it from the repo root, with the ``unitree-rl`` environment activated:
``export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`` if the loader
complains about ``libpython3.8.so``.
"""

import argparse
import os
import sys

import numpy as np

import isaacgym  # must be imported before torch or anything that loads its own libs
from isaacgym import gymapi, terrain_utils

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg  # noqa: E402
from legged_gym.envs.ybt.ybt_stairs_config import (  # noqa: E402
    StairsTerrain, StairsTerrainCfg, TiledStairsTerrain, YBTStairsCfg)
from legged_gym.scripts.view_ybt import lowest_point  # noqa: E402

SLOPE_THRESHOLD = 1.5  # risers above this slope are snapped to vertical


def build_mesh(gym, sim, height_field, horizontal_scale, vertical_scale,
               offset=(0.0, 0.0)):
    """Add a heightfield to the sim as one static triangle mesh.

    ``offset`` shifts the mesh in the world; tile coordinates of the trainers'
    map start at the origin, so the map is added unshifted while the standalone
    preview tile is centred.
    """
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        height_field,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=SLOPE_THRESHOLD)

    params = gymapi.TriangleMeshParams()
    params.nb_vertices = vertices.shape[0]
    params.nb_triangles = triangles.shape[0]
    params.static_friction = 1.0
    params.dynamic_friction = 1.0
    params.restitution = 0.0
    params.transform.p.x = offset[0]
    params.transform.p.y = offset[1]
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), params)

    print("terrain mesh: {} vertices, {} triangles".format(
        vertices.shape[0], triangles.shape[0]))
    print("height range: {:.3f} .. {:.3f} m".format(
        vertices[:, 2].min(), vertices[:, 2].max()))
    return vertices


def add_robot(gym, sim, env, args, spawn_pos):
    """Load the YBT URDF onto the run-up and hold it in the trained stand pose."""
    urdf_path = YBTFlatCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    if not os.path.isfile(urdf_path):
        sys.exit("URDF not found: {}".format(urdf_path))

    stance = dict(YBTFlatCfg.init_state.default_joint_angles)
    for item in filter(None, (s.strip() for s in args.pose.split(","))):
        key, _, value = item.partition("=")
        if key.strip() not in stance:
            sys.exit("--pose: no YBT joint named '{}'".format(key.strip()))
        stance[key.strip()] = float(value)

    # place the trunk so the lowest collision point rests exactly on the ground
    ground_z = spawn_pos[2]
    trunk_z = ground_z - lowest_point(urdf_path, stance)
    print("\nstance: trunk at z = {:.3f} m so the feet touch z = {:.3f} m"
          .format(trunk_z, ground_z))

    options = gymapi.AssetOptions()
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    options.collapse_fixed_joints = False  # keep all four foot contact bodies
    options.replace_cylinder_with_capsule = True
    options.flip_visual_attachments = True
    options.fix_base_link = not args.dynamic
    options.density = 0.001
    options.thickness = 0.01

    asset = gym.load_asset(sim, os.path.dirname(urdf_path),
                           os.path.basename(urdf_path), options)
    if asset is None:
        sys.exit("Isaac Gym failed to load the asset — see the log above")

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(spawn_pos[0], spawn_pos[1], trunk_z)
    actor = gym.create_actor(env, asset, pose, "ybt", 0, 0, 0)

    dof_props = gym.get_actor_dof_properties(env, actor)
    for i in range(gym.get_actor_dof_count(env, actor)):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
    p_gains, d_gains = [], []
    for name in gym.get_actor_dof_names(env, actor):
        for role, kp in YBTFlatCfg.control.stiffness.items():
            if role in name:
                p_gains.append(kp)
                d_gains.append(YBTFlatCfg.control.damping[role])
                break
    dof_props["stiffness"] = np.array(p_gains, dtype=np.float32)
    dof_props["damping"] = np.array(d_gains, dtype=np.float32)
    gym.set_actor_dof_properties(env, actor, dof_props)

    targets = [stance.get(n, 0.0) for n in gym.get_actor_dof_names(env, actor)]
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"] = targets
    dof_states["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, targets)

    print("robot welded to the world: {}".format(not args.dynamic))
    return actor


def _plan(height_field, horizontal_scale, vertical_scale, offset, lower, upper,
          spawn, look_at, camera):
    """Everything main() needs to place the mesh, the robot and the camera."""
    return {"height_field": height_field,
            "horizontal_scale": horizontal_scale,
            "vertical_scale": vertical_scale,
            "offset": offset,
            "lower": lower,
            "upper": upper,
            "spawn": spawn,
            "look_at": look_at,
            "camera": camera}


def describe_preview(args):
    """One standalone staircase tile, centred on the world origin."""
    overrides = {"num_steps": args.num_steps,
                 "step_height": args.step_height,
                 "step_depth": args.step_depth,
                 "approach_depth": args.approach_depth,
                 "platform_depth": args.platform_depth}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.stair_width is not None:
        overrides["stair_width"] = (None if args.stair_width.lower() in ("none", "full")
                                    else float(args.stair_width))

    terrain = StairsTerrain(StairsTerrainCfg.with_overrides(**overrides)
                            if overrides else None)
    print(terrain.describe())
    print("\nprofile (ground height every 0.5 m along x):")
    for x in np.arange(0.0, terrain.cfg.terrain_length + 1e-9, 0.5):
        print("   x = {:5.2f} m  z = {:5.2f} m  {}".format(
            x, terrain.level_at(x), "#" * int(round(terrain.level_at(x) * 40))))

    cfg = terrain.cfg
    offset = (-cfg.terrain_length / 2.0, -cfg.terrain_width / 2.0)
    spawn = (offset[0] + terrain.spawn_x, offset[1] + terrain.spawn_y,
             terrain.level_at(terrain.spawn_x))
    print("\nspawn: world ({:.2f}, {:.2f}, {:.2f}), facing +x  (stairs start at x = {:.2f})"
          .format(spawn[0], spawn[1], spawn[2], offset[0] + terrain.x_first_riser))
    return _plan(
        terrain.height_field(), cfg.horizontal_scale, cfg.vertical_scale, offset,
        (offset[0] - 1.0, offset[1] - 1.0, 0.0),
        (-offset[0] + 1.0, -offset[1] + 1.0, 10.0),
        spawn, (0.5, 0.0, 0.6), (-9.0, -7.0, 4.5))


def describe_map(args):
    """The tiled map the ``ybt_stairs`` task trains on, unshifted.

    Tile coordinates and world coordinates are the same thing here, exactly as
    in the environment, so what this shows is what a robot gets.
    """
    # YBTStairsCfg.terrain is a class; a subclass keeps the overrides local
    overrides = {"num_steps": args.num_steps,
                 "step_height": args.step_height,
                 "step_depth": args.step_depth,
                 "approach_depth": args.approach_depth,
                 "platform_depth": args.platform_depth,
                 "lane_width": args.map_lane_width}
    if args.map_levels is not None:
        overrides["num_levels"] = args.map_levels
    if args.map_lanes is not None:
        overrides["num_lanes"] = args.map_lanes
    overrides = {k: v for k, v in overrides.items() if v is not None}
    cfg = type("terrain", (YBTStairsCfg.terrain,), overrides)

    tmap = TiledStairsTerrain(cfg)
    print(tmap.describe())
    print("\nriser height by level ({} envs share each level at num_envs = {}):".format(
        cfg.num_lanes, cfg.num_levels * cfg.num_lanes))
    for level, height in enumerate(tmap.level_step_heights()):
        print("   level {:2d}  {:5.1f} cm  {}".format(
            level, height * 100, "#" * int(round(height * 100))))

    level = cfg.num_levels - 1 if args.map_level is None else args.map_level
    lane = args.map_lane
    if not 0 <= level < cfg.num_levels or not 0 <= lane < cfg.num_lanes:
        sys.exit("--map-level must be 0..{} and --map-lane 0..{}".format(
            cfg.num_levels - 1, cfg.num_lanes - 1))

    spawn = tuple(tmap.origins()[level, lane])
    look_at = (spawn[0] - tmap.spawn_x + tmap.x_top / 2.0, spawn[1],
               tmap.tiles[level].top_height / 2.0)
    print("\nlooking at level {} lane {}: {:.0f} cm risers, spawn at world "
          "({:.2f}, {:.2f}), stairs start at x = {:.2f}"
          .format(level, lane, tmap.tiles[level].step_height * 100,
                  spawn[0], spawn[1], spawn[0] - tmap.spawn_x + tmap.x_first_riser))
    return _plan(
        tmap.height_field(), cfg.horizontal_scale, cfg.vertical_scale, (0.0, 0.0),
        (-1.0, -1.0, 0.0),
        (cfg.num_levels * tmap.tile_length + 1.0,
         cfg.num_lanes * tmap.lane_width + 1.0, 10.0),
        spawn, look_at,
        (look_at[0] - 4.5, look_at[1] - 3.2, look_at[2] + 2.4))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true",
                        help="print the terrain report and exit")
    parser.add_argument("--terrain-only", action="store_true",
                        help="don't load the robot")
    parser.add_argument("--dynamic", action="store_true",
                        help="don't weld the trunk: gravity acts and it can settle or fall")
    parser.add_argument("--pose", type=str, default="",
                        help="comma-separated joint overrides, e.g. 'FL_thigh_joint=-1.0'")
    # terrain overrides, so the geometry can be iterated on without editing the cfg
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--step-height", type=float, default=None,
                        help="riser height in metres (default 0.15 = 15 cm)")
    parser.add_argument("--step-depth", type=float, default=None,
                        help="tread depth in metres (default 0.35)")
    parser.add_argument("--approach-depth", type=float, default=None)
    parser.add_argument("--platform-depth", type=float, default=None)
    parser.add_argument("--stair-width", type=str, default=None,
                        help="staircase width in metres, or 'none' for full tile width")
    parser.add_argument("--map", action="store_true",
                        help="show the tiled training map instead of one preview tile")
    parser.add_argument("--map-level", type=int, default=None,
                        help="difficulty row (0 = flat, num_levels-1 = 15 cm) to look at")
    parser.add_argument("--map-lane", type=int, default=0,
                        help="lane within that row to look at")
    parser.add_argument("--map-levels", type=int, default=None,
                        help="build the map with this many difficulty rows")
    parser.add_argument("--map-lanes", type=int, default=None,
                        help="build the map with this many lanes per row")
    parser.add_argument("--map-lane-width", type=float, default=None,
                        help="width of one robot's lane in metres (default 1.20)")
    args = parser.parse_args()

    print("=" * 78)
    print("STAIRCASE TERRAIN")
    print("=" * 78)
    plan = describe_map(args) if args.map else describe_preview(args)

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 200.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    # the plain actor API is used here, so keep the pipeline on the CPU
    sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False

    gym = gymapi.acquire_gym()
    graphics_id = -1 if args.headless else 0
    sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        sys.exit("failed to create sim (is a GPU/display available?)")

    env = gym.create_env(sim, gymapi.Vec3(*plan["lower"]),
                         gymapi.Vec3(*plan["upper"]), 1)

    build_mesh(gym, sim, plan["height_field"], plan["horizontal_scale"],
               plan["vertical_scale"], plan["offset"])
    # no ground plane: the heightfield's own z = 0 area is the ground, so a
    # second flat collider would overlap it and z-fight

    if not args.terrain_only:
        add_robot(gym, sim, env, args, plan["spawn"])

    if args.headless:
        print("\nheadless: skipping viewer")
        return

    cam_props = gymapi.CameraProperties()
    cam_props.horizontal_fov = 75.0
    cam_props.width = 1280
    cam_props.height = 720
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        sys.exit("failed to create viewer")

    # stand back on the approach side and look across the run-up at the stairs
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*plan["camera"]),
                              gymapi.Vec3(*plan["look_at"]))

    print("\nviewer running — drag to orbit, ESC or close the window to quit")
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
