"""Load a URDF into Isaac Gym, print a diagnostic report, and open the viewer.

Two independent checks run:
  1. a pure-XML audit of the .urdf file (works without a GPU / display), and
  2. an Isaac Gym import, which prints what the simulator actually built.

By default the robot is welded in place, standing on the ground, so it holds
still while you look at it: a resting pose is derived from the joint limits and
the trunk is spawned at whatever height puts the feet on z=0.

Usage:
    python legged_gym/scripts/view_ybt.py
    python legged_gym/scripts/view_ybt.py --urdf resources/robots/ybt/urdf/ybt.urdf
    python legged_gym/scripts/view_ybt.py --headless          # report only, no window
    python legged_gym/scripts/view_ybt.py --keep-fixed-joints # don't collapse fixed joints
    python legged_gym/scripts/view_ybt.py --pose FL_thigh_joint=-1.2,FL_calf_joint=2.0
    python legged_gym/scripts/view_ybt.py --dynamic           # unweld: gravity + PD hold
    python legged_gym/scripts/view_ybt.py --free              # no PD hold, robot just flops
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET

import numpy as np

import isaacgym  # must be imported before torch/anything that loads its own libs
from isaacgym import gymapi

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

DEFAULT_URDF = "resources/robots/ybt/urdf/ybt.urdf"


# --------------------------------------------------------------------------- #
# XML-side audit (no simulator involved)
# --------------------------------------------------------------------------- #
def audit_urdf(path):
    """Parse the URDF as plain XML and report structural problems. Returns (links, joints)."""
    root = ET.parse(path).getroot()
    print("=" * 78)
    print(f"URDF AUDIT  {path}")
    print("=" * 78)
    print(f"robot name attribute : {root.get('name')}")

    links = root.findall("link")
    joints = root.findall("joint")
    print(f"links: {len(links)}   joints: {len(joints)}")

    link_names = [l.get("name") for l in links]
    joint_names = [j.get("name") for j in joints]

    problems = []

    # duplicate names
    for label, names in (("link", link_names), ("joint", joint_names)):
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            problems.append(f"duplicate {label} names: {sorted(dupes)}")

    # joints must reference existing links
    known = set(link_names)
    for j in joints:
        p = j.find("parent")
        c = j.find("child")
        for role, node in (("parent", p), ("child", c)):
            ref = node.get("link") if node is not None else None
            if ref not in known:
                problems.append(f"joint '{j.get('name')}' {role} link '{ref}' does not exist")

    # exactly one root link: every link that is never a child
    children = {j.find("child").get("link") for j in joints if j.find("child") is not None}
    roots = [n for n in link_names if n not in children]
    print(f"root link(s)         : {roots}")
    if len(roots) != 1:
        problems.append(f"expected exactly 1 root link, found {len(roots)}: {roots}")

    # mesh files must exist on disk, relative to the urdf directory
    urdf_dir = os.path.dirname(os.path.abspath(path))
    meshes = {}
    for l in links:
        for geom in l.iter("geometry"):
            mesh = geom.find("mesh")
            if mesh is not None:
                meshes.setdefault(mesh.get("filename"), []).append(l.get("name"))
    print(f"referenced meshes    : {len(meshes)}")
    for fname in sorted(meshes):
        resolved = os.path.normpath(os.path.join(urdf_dir, fname))
        ok = os.path.isfile(resolved)
        print(f"   [{'ok ' if ok else 'MISSING'}] {fname:<28} <- {', '.join(meshes[fname])}")
        if not ok:
            problems.append(f"mesh file not found: {resolved}")

    missing_inertial = [l.get("name") for l in links if l.find("inertial") is None]
    missing_collision = [l.get("name") for l in links if l.find("collision") is None]

    if missing_inertial:
        print(f"links w/o <inertial> : {missing_inertial}")
        print("   (Isaac Gym will fall back to asset_options.density — mass will be a guess)")
    if missing_collision:
        print(f"links w/o <collision>: {missing_collision}")
        print("   (no collision geometry -> these links pass through the ground)")

    # per-joint summary
    print("\njoints:")
    def _num(node, attr):
        """Format a joint limit as a fixed-width float, or '-' when absent."""
        if node is None or node.get(attr) is None:
            return f"{'-':>10}"
        return f"{float(node.get(attr)):>10.4f}"

    print(f"\n   {'name':<18}{'type':<10}{'axis':<10}{'lower':>10}{'upper':>10}{'dont_collapse':>15}")
    for j in joints:
        axis = j.find("axis")
        limit = j.find("limit")
        print(f"   {j.get('name'):<18}{j.get('type'):<10}"
              f"{(axis.get('xyz') if axis is not None else '-'):<10}"
              f"{_num(limit, 'lower')}{_num(limit, 'upper')}"
              f"{str(j.get('dont_collapse')):>15}")

    print("\nstructure problems:", "none" if not problems else "")
    for p in problems:
        print("   !!", p)

    return link_names, joint_names


# --------------------------------------------------------------------------- #
# URDF kinematics — used to pick a valid stance and the height that puts the feet
# exactly on the ground, so the robot stands still instead of flopping over.
# --------------------------------------------------------------------------- #
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _origin_T(node):
    xyz = [float(x) for x in (node.get('xyz') or '0 0 0').split()] if node is not None else [0, 0, 0]
    rpy = [float(x) for x in (node.get('rpy') or '0 0 0').split()] if node is not None else [0, 0, 0]
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def _axis_angle_R(axis, angle):
    a = np.array(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def urdf_kinematics(urdf_path, joint_targets):
    """Forward kinematics. Returns (link -> 4x4 trunk-frame transform, joints dict)."""
    root = ET.parse(urdf_path).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    joints = []
    for j in root.findall('joint'):
        axis = j.find('axis')
        joints.append({
            'name': j.get('name'),
            'type': j.get('type'),
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'T': _origin_T(j.find('origin')),
            'axis': [float(v) for v in (axis.get('xyz').split() if axis is not None else ['1', '0', '0'])],
        })
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j['parent'], []).append(j)
    root_link = next(n for n in links if n not in {j['child'] for j in joints})

    out = {}

    def walk(link, T):
        out[link] = T
        for j in by_parent.get(link, []):
            T_j = j['T'].copy()
            if j['type'] in ('revolute', 'continuous'):
                R = np.eye(4)
                R[:3, :3] = _axis_angle_R(j['axis'], joint_targets.get(j['name'], 0.0))
                T_j = T_j @ R
            walk(j['child'], T @ T_j)

    walk(root_link, np.eye(4))
    return out, links, joints


def _primitive_low_z(geom, T_origin):
    """Lowest z of a collision primitive, given its transform in the link frame."""
    def half_extent(shape, size):
        if shape == 'sphere':
            return size
        if shape == 'box':
            return float(np.abs(T_origin[:3, :3][2]) @ np.array(size) / 2.0)
        if shape == 'cylinder':
            d = T_origin[:3, :3] @ np.array([0.0, 0.0, 1.0])
            L, r = size              # size = (length, radius)
            return abs(L / 2.0 * d[2]) + r * np.sqrt(max(0.0, 1.0 - d[2] ** 2))
        return 0.0

    centre_z = T_origin[2, 3]
    if geom.find('sphere') is not None:
        return centre_z - half_extent('sphere', float(geom.find('sphere').get('radius')))
    if geom.find('box') is not None:
        return centre_z - half_extent('box', [float(v) for v in geom.find('box').get('size').split()])
    if geom.find('cylinder') is not None:
        c = geom.find('cylinder')
        return centre_z - half_extent('cylinder', [float(c.get('length')), float(c.get('radius'))])
    return centre_z


def lowest_point(urdf_path, joint_targets):
    """Lowest z over all collision geometry, in the trunk frame, for a given pose."""
    T_by_link, links, _ = urdf_kinematics(urdf_path, joint_targets)
    low = float('inf')
    for name, link in links.items():
        for col in link.findall('collision'):
            geom = col.find('geometry')
            if geom is None:
                continue
            T_col = _origin_T(col.find('origin')) if col.find('origin') is not None else np.eye(4)
            low = min(low, _primitive_low_z(geom, T_by_link[name] @ T_col))
    return low


def role_of(joint_name):
    """'FL_thigh_joint' -> 'thigh_joint': the leg-independent part of the name."""
    return joint_name.split('_', 1)[1] if '_' in joint_name else joint_name


def default_stance(urdf_path, overrides=None, foot_suffix="_foot"):
    """A resting pose: feet under the hips, legs as straight as the limits allow.

    Searches the last three revolute joints of every chain ending in a
    '<foot_suffix>' link (hip / thigh / calf), then applies the shape found on one
    leg to every leg that shares the same joint roles. Nothing here is
    ybt-specific, but it needs a recognisable foot link — without one it falls back
    to the midpoint of each joint's limits, which is still in range.
    """
    root = ET.parse(urdf_path).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    rev, all_by_child = [], {}
    for j in root.findall('joint'):
        lim = j.find('limit')

        def _lim(which):
            v = lim.get(which) if lim is not None else None
            return float(v) if v is not None else None
        info = {'name': j.get('name'), 'parent': j.find('parent').get('link'),
                'child': j.find('child').get('link'), 'origin': j.find('origin'),
                'lower': _lim('lower'), 'upper': _lim('upper')}
        all_by_child[info['child']] = info
        if j.get('type') in ('revolute', 'continuous'):
            rev.append(info)

    names = [j['name'] for j in rev]
    targets = {j['name']: ((j['lower'] + j['upper']) / 2.0
                           if j['lower'] is not None and j['upper'] is not None else 0.0)
               for j in rev}

    # a leg = the last three revolute joints on the way up from a foot link
    leg = None
    for f in sorted(n for n in links if foot_suffix in n):
        chain, link = [], f
        while link in all_by_child:                     # walk through fixed joints too
            j = all_by_child[link]
            if j in rev:
                chain.append(j)
            link = j['parent']
        chain.reverse()
        if len(chain) >= 3:
            leg = (chain[-3], chain[-2], chain[-1], f)
            break

    if leg is not None:
        hip_j, th_j, ca_j, foot_link = leg
        hip_T = urdf_kinematics(urdf_path, targets)[0]
        hip_origin = hip_T[hip_j['parent']] @ _origin_T(hip_j['origin'])

        def span(j):
            return (j['lower'], j['upper']) if j['lower'] is not None else (0.0, 0.0)

        best = None
        for th in np.linspace(*span(th_j), 81):
            for ca in np.linspace(*span(ca_j), 81):
                p = urdf_kinematics(urdf_path, {**targets, th_j['name']: th,
                                                ca_j['name']: ca})[0][foot_link][:3, 3]
                if abs(p[0] - hip_origin[0, 3]) > 0.004:   # must be under the hip
                    continue
                depth = abs(p[2] - hip_origin[2, 3])
                if best is None or depth > best[0]:        # tie-break: straightest leg
                    best = (depth, th, ca)

        if best is not None:
            _, th, ca = best
            chosen = {role_of(hip_j['name']): targets[hip_j['name']],
                      role_of(th_j['name']): th,
                      role_of(ca_j['name']): ca}
            for j in rev:
                if role_of(j['name']) in chosen:           # same shape on every leg
                    v = chosen[role_of(j['name'])]
                    if j['lower'] is not None:             # legs may have different ranges
                        v = min(max(v, j['lower']), j['upper'])
                    targets[j['name']] = v

    for k, v in (overrides or {}).items():
        if k not in targets:
            sys.exit(f"--pose: no revolute joint named '{k}' in the URDF")
        targets[k] = v
    return targets, names


# --------------------------------------------------------------------------- #
# Isaac Gym import + diagnostics
# --------------------------------------------------------------------------- #
def make_asset_options(args):
    opts = gymapi.AssetOptions()
    # mirror the defaults used for training (legged_robot_config.asset)
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS if args.hold else gymapi.DOF_MODE_EFFORT)
    opts.collapse_fixed_joints = not args.keep_fixed_joints
    opts.replace_cylinder_with_capsule = True
    opts.flip_visual_attachments = True
    opts.fix_base_link = args.fix_base
    opts.density = 0.001
    opts.thickness = 0.01
    return opts


def report_asset(gym, asset, urdf_links, urdf_joints):
    print("\n" + "=" * 78)
    print("ISAAC GYM IMPORT")
    print("=" * 78)

    bodies = gym.get_asset_rigid_body_names(asset)
    joints = gym.get_asset_joint_names(asset)
    dofs = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print(f"rigid bodies: {len(bodies)}  (urdf links {len(urdf_links)})")
    print(f"joints      : {len(joints)}  (urdf joints {len(urdf_joints)})")
    print(f"dofs        : {len(dofs)}")

    dropped_bodies = [n for n in urdf_links if n not in bodies]
    dropped_joints = [n for n in urdf_joints if n not in joints]
    if dropped_bodies:
        print(f"links merged away by collapse_fixed_joints: {dropped_bodies}")
    if dropped_joints:
        print(f"joints merged away by collapse_fixed_joints: {dropped_joints}")

    shape_indices = gym.get_asset_rigid_body_shape_indices(asset)
    print("\nbodies:")
    print(f"   {'name':<18}{'collision shapes':>17}")
    for i, name in enumerate(bodies):
        print(f"   {name:<18}{int(shape_indices[i].count):>17}")

    if dof_props.dtype.names:
        print(f"\ndof property fields: {dof_props.dtype.names}")

    print("\ndofs:")
    print(f"   {'name':<18}{'type':<14}{'lower':>9}{'upper':>9}{'hasLimits':>11}"
          f"{'effort':>9}{'velocity':>10}")
    for i, name in enumerate(dofs):
        p = dof_props[i]
        lower = float(p["lower"]) if "lower" in dof_props.dtype.names else float("nan")
        upper = float(p["upper"]) if "upper" in dof_props.dtype.names else float("nan")
        has_lim = bool(p["hasLimits"]) if "hasLimits" in dof_props.dtype.names else None
        effort = float(p["effort"]) if "effort" in dof_props.dtype.names else float("nan")
        vel = float(p["velocity"]) if "velocity" in dof_props.dtype.names else float("nan")
        dtype = str(gym.get_asset_dof_type(asset, i)).rsplit(".", 1)[-1]
        print(f"   {name:<18}{dtype:<14}{lower:>9.3f}{upper:>9.3f}{str(has_lim):>11}"
              f"{effort:>9.1f}{vel:>10.1f}")

    missing = [n for n in joints if n not in dofs]
    if missing:
        print(f"\nfixed joints (no dof): {missing}")

    extra = [n for n in dofs if n not in urdf_joints]
    if extra:
        print(f"!! dofs with no matching urdf joint: {extra}")

    return bodies, dofs


def report_actor(gym, env, actor):
    props = gym.get_actor_rigid_body_properties(env, actor)
    names = gym.get_actor_rigid_body_names(env, actor)
    total = 0.0
    print("\nbody masses (after import):")
    for name, p in zip(names, props):
        total += p.mass
        print(f"   {name:<18}{p.mass:>10.4f} kg")
    print(f"   {'TOTAL':<18}{total:>10.4f} kg")
    print(f"\ntrunk origin height at spawn: {gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)['pose']['p'][0][2]:.4f} m")
    return total


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF,
                        help="path to the URDF (absolute, or relative to the repo root)")
    parser.add_argument("--headless", action="store_true",
                        help="print the report only, don't open the viewer")
    parser.add_argument("--keep-fixed-joints", action="store_true",
                        help="do not merge bodies joined by fixed joints")
    parser.add_argument("--dynamic", action="store_true",
                        help="don't weld the trunk: gravity acts and the robot can settle/fall")
    parser.add_argument("--free", action="store_true",
                        help="no PD hold: leave drive mode as effort so the robot flops (implies --dynamic)")
    parser.add_argument("--pose", type=str, default="",
                        help="comma-separated joint overrides, e.g. 'FL_thigh_joint=-0.8,FL_calf_joint=1.5' "
                             "(default: midpoint of each joint's limits)")
    parser.add_argument("--height", type=float, default=None,
                        help="spawn height of the trunk [m] (default: computed so the feet touch the ground)")
    parser.add_argument("--physics", type=str, default="physx", help="physx or flex")
    args = parser.parse_args()

    args.hold = not args.free
    args.fix_base = not (args.dynamic or args.free)

    urdf_path = args.urdf
    if not os.path.isabs(urdf_path):
        candidate = os.path.join(LEGGED_GYM_ROOT_DIR, urdf_path)
        urdf_path = candidate if os.path.isfile(candidate) else os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        sys.exit(f"URDF not found: {urdf_path}")

    urdf_links, urdf_joints = audit_urdf(urdf_path)

    overrides = {}
    for item in filter(None, (s.strip() for s in args.pose.split(','))):
        k, _, v = item.partition('=')
        overrides[k.strip()] = float(v)
    stance, stance_names = default_stance(urdf_path, overrides)
    ground_clearance = lowest_point(urdf_path, stance)
    spawn_height = args.height if args.height is not None else -ground_clearance

    print("\n" + "=" * 78)
    print("STANCE")
    print("=" * 78)
    print("pose (feet under the hips, legs as straight as the limits allow;")
    print("      override any joint with --pose name=value):")
    for n in stance_names:
        print(f"   {n:<18}{stance[n]:+.4f} rad")
    print(f"lowest collision point in this pose : {ground_clearance:+.4f} m (trunk frame)")
    print(f"spawn height so the feet rest on z=0: {spawn_height:.4f} m")

    # --- simulator -------------------------------------------------------- #
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 200.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    if args.physics == "physx":
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        # the GPU pipeline needs the dof-state tensor API; this script uses the
        # plain actor API, so keep the pipeline on the CPU
        sim_params.physx.use_gpu = False
    sim_params.use_gpu_pipeline = False

    gym = gymapi.acquire_gym()
    graphics_id = -1 if args.headless else 0
    sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        sys.exit("failed to create sim (is a GPU/display available?)")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)
    asset = gym.load_asset(sim, asset_root, asset_file, make_asset_options(args))
    if asset is None:
        sys.exit("Isaac Gym failed to load the asset — see the log messages above")

    report_asset(gym, asset, urdf_links, urdf_joints)

    # --- one environment -------------------------------------------------- #
    spacing = 2.0
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0.0),
                         gymapi.Vec3(spacing, spacing, 0.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, spawn_height)
    actor = gym.create_actor(env, asset, pose, "ybt", 0, 0, 0)

    # targets, in the order the asset chose for its DOFs
    num_dofs = gym.get_actor_dof_count(env, actor)
    dof_names = gym.get_actor_dof_names(env, actor)
    targets = [stance.get(n, 0.0) for n in dof_names]

    dof_props = gym.get_actor_dof_properties(env, actor)
    if args.hold:
        for i in range(num_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 100.0
            dof_props["damping"][i] = 2.0
    gym.set_actor_dof_properties(env, actor, dof_props)
    gym.set_actor_dof_position_targets(env, actor, targets)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"] = targets
    dof_states["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    print(f"\ntrunk welded to the world: {args.fix_base}"
          f"{'  (use --dynamic to let it move)' if args.fix_base else ''}")

    report_actor(gym, env, actor)

    if args.headless:
        print("\nheadless: skipping viewer")
        return

    # --- viewer ------------------------------------------------------------ #
    cam_props = gymapi.CameraProperties()
    cam_props.horizontal_fov = 75.0
    cam_props.width = 1280
    cam_props.height = 720
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        sys.exit("failed to create viewer")

    cam_pos = gymapi.Vec3(1.4, 1.4, 0.9)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # a 40 cm RGB frame at the world origin, so a floating/mis-scaled mesh is obvious
    verts = np.array([[0, 0, 0], [0.4, 0, 0],
                      [0, 0, 0], [0, 0.4, 0],
                      [0, 0, 0], [0, 0, 0.4]], dtype=np.float32)
    colors = np.array([[1, 0, 0], [1, 0, 0],
                       [0, 1, 0], [0, 1, 0],
                       [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    gym.add_lines(viewer, env, len(verts), verts, colors)

    print("\nviewer running — drag with the mouse to orbit, press ESC or close the window to quit")
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
