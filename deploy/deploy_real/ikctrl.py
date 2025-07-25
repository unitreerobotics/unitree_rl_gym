import os
from typing import Tuple
from contextlib import contextmanager
from pathlib import Path
import time

import yaml
import numpy as np
import torch as th
# from yourdfpy import URDF

import pinocchio as pin
import pink
from pink.tasks import FrameTask


@contextmanager
def with_dir(d):
    d0 = os.getcwd()
    try:
        os.chdir(d)
        yield
    finally:
        os.chdir(d0)


def xyzw2wxyz(q_xyzw: th.Tensor, dim: int = -1):
    if isinstance(q_xyzw, np.ndarray):
        return np.roll(q_xyzw, 1, axis=dim)
    return th.roll(q_xyzw, 1, dims=dim)


def wxyz2xyzw(q_wxyz: th.Tensor, dim: int = -1):
    if isinstance(q_wxyz, np.ndarray):
        return np.roll(q_wxyz, -1, axis=dim)
    return th.roll(q_wxyz, -1, dims=dim)


def dls_ik(
        dpose: np.ndarray,
        jac: np.ndarray,
        sqlmda: float
):
    """
    Arg:
        dpose: task-space error (A[..., err])
        jac: jacobian (A[..., err, dof])
        sqlmda: DLS damping factor.
    Return:
        joint residual (A[..., dof])
    """
    if isinstance(dpose, tuple):
        dpose = np.concatenate([dpose[0], dpose[1]], axis=-1)
    J = jac
    A = J @ J.T
    # NOTE(ycho): add to view of diagonal
    a = np.einsum('...ii->...i', A)
    a += sqlmda
    dq = (J.T @ np.linalg.solve(A, dpose[..., None]))[..., 0]
    return dq


class IKCtrl:
    def __init__(self,
                 urdf_path: str,
                 act_joints: Tuple[str, ...],
                 frame: str = 'left_hand_palm_link',
                 sqlmda: float = 0.05**2):
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(filename=path.name,
                                                   package_dirs=["."],
                                                   root_joint=None)
            self.robot = robot
            # NOTE(ycho): we skip joint#0(="universe")
            joint_names = list(self.robot.model.names)
            assert (joint_names[0] == 'universe')
            self.joint_names = joint_names[1:]

        # NOTE(ycho): build index map between pin.q and other set(s) of ordered
        # joints.
        act_from_pin = []
        for j in act_joints:
            act_from_pin.append(robot.index(j) - 1)
        self.frame = frame
        self.act_from_pin = np.asarray(act_from_pin, dtype=np.int32)
        self.task = FrameTask(frame, position_cost=1.0, orientation_cost=0.0)
        self.sqlmda = sqlmda
        self.cfg = pink.Configuration(robot.model, robot.data,
                                      np.zeros_like(robot.q0))

    def fk(self, q: np.ndarray):
        robot = self.robot
        return pink.Configuration(
            robot.model, robot.data, q).get_transform_frame_to_world(
            self.frame)

    def __call__(self,
                 q0: np.ndarray,
                 target_pose: np.ndarray,
                 rel: bool = False
                 ):
        """
        Arg:
            q0: Current robot joints; A[..., 43?]
            target_pose:
                Policy output. A[..., 7] formatted as (xyz, q_{wxyz})
                Given as world frame absolute pose, for some reason.
        Return:
            joint residual: A[..., 7]
        """
        robot = self.robot

        # source pose
        self.cfg.update(q0)
        T0 = self.cfg.get_transform_frame_to_world(self.frame)

        # target pose
        dst_xyz = target_pose[..., 0:3]
        dst_quat = pin.Quaternion(wxyz2xyzw(target_pose[..., 3:7]))
        T1 = pin.SE3(dst_quat, dst_xyz)
        if rel:
            TL = pin.SE3.Identity()
            TL.translation = dst_xyz
            TR = pin.SE3.Identity()
            TR.rotation = dst_quat.toRotationMatrix()
            T1 = TL * T0 * TR

        # jacobian
        self.task.set_target(T0)
        jac = self.task.compute_jacobian(self.cfg)
        jac = jac[:, self.act_from_pin]

        # error&ik
        dT = T1.actInv(T0)
        dpose = pin.log(dT).vector
        dq = dls_ik(dpose, jac, self.sqlmda)
        return dq


def main():
    from yourdfpy import URDF
    urdf_path = '../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'

    # == init yourdfpy robot ==
    path = Path(urdf_path)
    with with_dir(path.parent):
        viz = URDF.load(path.name,
                        build_collision_scene_graph=True,
                        load_meshes=False,
                        load_collision_meshes=True)

    # == populate with defaults ==
    with open('./configs/ik.yaml', 'r') as fp:
        data = yaml.safe_load(fp)
    q_mot = np.asarray(data['default_angles'])
    q_viz = np.zeros((len(viz.actuated_joint_names)))

    viz_from_mot = np.zeros(len(data['motor_joint']),
                            dtype=np.int32)
    for i_mot, j in enumerate(data['motor_joint']):
        i_viz = viz.actuated_joint_names.index(j)
        viz_from_mot[i_mot] = i_viz
    q_viz[viz_from_mot] = q_mot

    ctrl = IKCtrl(urdf_path, data['act_joint'])

    q_pin = np.zeros_like(ctrl.cfg.q)
    pin_from_mot = np.zeros(len(data['motor_joint']),
                            dtype=np.int32)
    for i_mot, j in enumerate(data['motor_joint']):
        i_pin = ctrl.joint_names.index(j)
        pin_from_mot[i_mot] = i_pin

    mot_from_act = np.zeros(len(data['act_joint']),
                            dtype=np.int32)
    for i_act, j in enumerate(data['act_joint']):
        i_mot = data['motor_joint'].index(j)
        mot_from_act[i_act] = i_mot

    viz.update_cfg(q_viz)
    # viz.show(collision_geometry=True)

    if True:
        current_pose = viz.get_transform(ctrl.frame)
        print('curpose (viz)', current_pose)
        print('curpose (pin)', ctrl.fk(q_pin).homogeneous)

    def callback(scene):
        if True:
            current_pose = viz.get_transform(ctrl.frame)
            T = pin.SE3()
            T.translation = (current_pose[..., :3, 3]
                             + [0.01, 0.0, 0.0])
            T.rotation = current_pose[..., :3, :3]
            target_pose = pin.SE3ToXYZQUAT(T)
            target_pose[..., 3:7] = xyzw2wxyz(target_pose[..., 3:7])
        q_pin[pin_from_mot] = q_mot
        dq = ctrl(q_pin, target_pose, rel=False)
        q_mot[mot_from_act] += dq

        q_viz[viz_from_mot] = q_mot
        viz.update_cfg(q_viz)

    viz.show(collision_geometry=True,
             callback=callback)


if __name__ == '__main__':
    main()
