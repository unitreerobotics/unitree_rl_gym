"""Asset/config/reset regression checks. Run in the unitree-rl environment."""
import copy
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import isaacgym  # must precede torch
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg, YBTFlatCfgPPO
from legged_gym.envs.ybt.ybt_env import YBTRobot, validate_ybt_asset


def default_link_transforms(root, angles):
    """URDF forward kinematics for the default stand, relative to trunk."""
    transforms = {"trunk": np.eye(4)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find("parent").attrib["link"]
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            rotation = rz @ ry @ rx
            if joint.attrib["type"] != "fixed":
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = angles[joint.attrib["name"]]
                rotation = rotation @ (np.eye(3) + np.sin(angle) * skew
                                       + (1 - np.cos(angle)) * (skew @ skew))
            local = np.eye(4)
            local[:3, :3] = rotation
            local[:3, 3] = xyz
            transforms[joint.find("child").attrib["link"]] = transforms[parent] @ local
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF is not a connected tree rooted at trunk")
    return transforms


class YBTTests(unittest.TestCase):
    def setUp(self):
        self.cfg = YBTFlatCfg()
        self.path = validate_ybt_asset(self.cfg)
        self.root = ET.parse(str(self.path)).getroot()

    def test_registration_preserves_existing_tasks(self):
        self.assertTrue({"go2", "g1", "h1", "h1_2", "ybt"}.issubset(task_registry.task_classes))
        self.assertIs(task_registry.get_task_class("ybt"), YBTRobot)
        self.assertIs(task_registry.get_task_class("go2"), LeggedRobot)
        self.assertEqual(task_registry.get_cfgs("go2")[1].runner.experiment_name, "rough_go2")

    def test_training_defaults(self):
        ppo = YBTFlatCfgPPO()
        self.assertEqual(self.cfg.env.num_envs, 12288)
        self.assertEqual((self.cfg.env.num_actions, self.cfg.env.num_observations), (12, 48))
        self.assertAlmostEqual(self.cfg.sim.dt * self.cfg.control.decimation, 0.02)
        self.assertEqual(ppo.runner.num_steps_per_env, 24)
        self.assertEqual(ppo.runner.max_iterations, 1500)
        self.assertEqual(ppo.runner.experiment_name, "flat_ybt")
        self.assertFalse(ppo.runner.resume)

    def test_meshes_are_nonempty_collada(self):
        paths = {self.path.parent / mesh.attrib["filename"] for mesh in self.root.findall(".//mesh")}
        self.assertEqual(len(paths), 5)
        for path in paths:
            self.assertGreater(path.stat().st_size, 0)
            self.assertTrue(ET.parse(str(path)).getroot().tag.endswith("COLLADA"))

    def test_inertias_and_mass(self):
        mass = 0.0
        for link in self.root.findall("link"):
            inertial = link.find("inertial")
            m = float(inertial.find("mass").attrib["value"])
            self.assertGreater(m, 0.0)
            mass += m
            i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
            matrix = np.array([[i["ixx"], i["ixy"], i["ixz"]],
                               [i["ixy"], i["iyy"], i["iyz"]],
                               [i["ixz"], i["iyz"], i["izz"]]])
            eigenvalues = np.linalg.eigvalsh(matrix)
            self.assertTrue(np.all(eigenvalues > 0), link.attrib["name"])
            self.assertLessEqual(eigenvalues[-1], eigenvalues[:2].sum() + 1e-6)
        self.assertAlmostEqual(mass, 61.669175, places=5)

    def test_stand_has_four_feet_above_ground(self):
        transforms = default_link_transforms(self.root, self.cfg.init_state.default_joint_angles)
        self.assertEqual(len(transforms), len(self.root.findall("link")))
        for leg in ("FR", "FL", "RR", "RL"):
            foot = self.root.find("link[@name='{}_foot']".format(leg))
            radius = float(foot.find("collision/geometry/sphere").attrib["radius"])
            foot_height = self.cfg.init_state.pos[2] + transforms[leg + "_foot"][2, 3]
            self.assertGreater(foot_height - radius, 0.0)
            self.assertLess(foot_height - radius, 0.10)

    def test_feet_clearance_rewards_lifted_feet(self):
        # The clearance term must pay for lifting a swing foot and must never go
        # negative for a dragged foot, otherwise the policy is pushed to drag.
        robot = object.__new__(YBTRobot)
        robot.cfg = self.cfg
        robot.feet_indices = torch.arange(4)
        robot.contact_forces = torch.zeros(2, 6, 3)
        robot.rigid_body_states = torch.zeros(2, 6, 13)
        target = self.cfg.rewards.feet_clearance_target
        cap = self.cfg.rewards.feet_clearance_cap
        self.assertGreater(target, 0.03)  # above the foot collision-sphere radius
        robot.rigid_body_states[:, :4, 2] = target
        robot.contact_forces[:, :4, 2] = 5.0  # resting feet: no reward
        self.assertEqual(robot._reward_feet_clearance().sum().item(), 0.0)
        robot.rigid_body_states[:, :4, 2] = cap  # feet lifted to the cap
        robot.contact_forces[:, :4, 2] = 0.0     # and out of contact
        rew = robot._reward_feet_clearance()
        self.assertTrue(torch.allclose(rew, torch.full_like(rew, 4.0)))
        robot.contact_forces[:, :4, 2] = 5.0     # a high foot still in contact: no reward
        self.assertEqual(robot._reward_feet_clearance().sum().item(), 0.0)

    def test_rejects_invalid_default_angles(self):
        bad = copy.deepcopy(self.cfg)
        bad.init_state.default_joint_angles = dict(self.cfg.init_state.default_joint_angles)
        bad.init_state.default_joint_angles["FR_calf_joint"] = -1.5
        with self.assertRaisesRegex(ValueError, "outside the URDF limits"):
            validate_ybt_asset(bad)

    def test_rejects_missing_asset(self):
        bad = copy.deepcopy(self.cfg)
        bad.asset.file = str(self.path.parent / "missing_ybt.urdf")
        with self.assertRaises(FileNotFoundError):
            validate_ybt_asset(bad)

    def test_rejects_observation_mismatch(self):
        bad = copy.deepcopy(self.cfg)
        bad.env.num_observations = 47
        with self.assertRaisesRegex(ValueError, "48 observations"):
            validate_ybt_asset(bad)

    def test_reset_is_bounded_and_only_changes_selected_envs(self):
        # Exercise the actual override without creating a PhysX simulation.
        robot = object.__new__(YBTRobot)
        robot.cfg = self.cfg
        robot.device = "cpu"
        names = list(reversed(list(self.cfg.init_state.default_joint_angles)))
        robot.num_dof = len(names)
        robot.default_dof_pos = torch.tensor([[self.cfg.init_state.default_joint_angles[n] for n in names]])
        joints = {j.attrib["name"]: j for j in self.root.findall("joint")}
        bounds = [[float(joints[n].find("limit").attrib[key]) for key in ("lower", "upper")] for n in names]
        hard = torch.tensor(bounds)
        mid = hard.mean(dim=1)
        half = (hard[:, 1] - hard[:, 0]) * self.cfg.rewards.soft_dof_pos_limit / 2
        robot.dof_pos_limits = torch.stack((mid - half, mid + half), dim=1)
        robot.dof_state = torch.full((64, 12, 2), -99.0)
        robot.dof_pos = robot.dof_state[..., 0]
        robot.dof_vel = robot.dof_state[..., 1]
        robot.gym = Mock()
        robot.sim = object()
        ids = torch.arange(0, 64, 2)
        with patch("legged_gym.envs.ybt.ybt_env.gymtorch.unwrap_tensor", side_effect=lambda t: t):
            for _ in range(100):
                robot._reset_dofs(ids)
                q = robot.dof_pos[ids]
                self.assertTrue(torch.all(q >= hard[:, 0]))
                self.assertTrue(torch.all(q <= hard[:, 1]))
                self.assertTrue(torch.all(torch.abs(q - robot.default_dof_pos) <= 0.050001))
                self.assertTrue(torch.all(robot.dof_vel[ids] == 0))
        self.assertTrue(torch.all(robot.dof_state[1::2] == -99))
        self.assertEqual(robot.gym.set_dof_state_tensor_indexed.call_count, 100)
        self.assertEqual(robot.gym.set_dof_state_tensor_indexed.call_args[0][2].dtype, torch.int32)
        self.assertFalse(torch.equal(robot.dof_pos[0], robot.dof_pos[2]))


if __name__ == "__main__":
    unittest.main()
