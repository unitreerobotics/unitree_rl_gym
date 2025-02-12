import os
from typing import Tuple
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch as th

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
                 joint_names: Tuple[str, ...],
                 sqlmda: float = 0.05**2):
        path = Path(urdf_path)
        with with_dir(path.parent):
            robot = pin.RobotWrapper.BuildFromURDF(filename=urdf_path,
                                                   package_dirs=["."],
                                                   root_joint=None)
            self.robot = robot
        # NOTE(ycho): build index map between pin.q and other set(s) of ordered joints.
        larm_from_pin = []
        for j in joint_names:
            larm_from_pin.append(robot.index(j) - 1)
        self.larm_from_pin = np.asarray(larm_from_pin, dtype=np.int32)
        self.task = FrameTask("left_hand_palm_link",
                              position_cost=1.0,
                              orientation_cost=0.0)
        self.sqlmda = sqlmda
        self.cfg = pink.Configuration(robot.model, robot.data,
                                      np.zeros_like(robot.q0))

    def __call__(self,
                 q0: np.ndarray,
                 target_pose: np.ndarray,
                 rel:bool=False
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
        T0 = self.cfg.get_transform_frame_to_world("left_hand_palm_link")

        # target pose
        dst_xyz = target_pose[..., 0:3]
        dst_quat = pin.Quaternion(wxyz2xyzw(target_pose[..., 3:7]))
        T1 = pin.SE3(dst_quat, dst_xyz)
        if rel:
            T1 = T0 @ T1

        # jacobian
        self.task.set_target(T0)
        jac = self.task.compute_jacobian(self.cfg)
        jac = jac[:, self.larm_from_pin]

        # error&ik
        dT = T1.actInv(T0)
        dpose = pin.log(dT).vector
        dq = dls_ik(dpose, jac, self.sqlmda)
        return dq

