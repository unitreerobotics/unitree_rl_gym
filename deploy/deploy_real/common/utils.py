import numpy as np
from geometry_msgs.msg import Vector3, Quaternion
from typing import Optional, Tuple
def to_array(v):
    if isinstance(v, Vector3):
        return np.array([v.x, v.y, v.z], dtype=np.float32)
    elif isinstance(v, Quaternion):
        return np.array([v.x, v.y, v.z, v.w], dtype=np.float32)


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True).clip(min=eps, max=None)

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.copy().reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(shape)

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), dim=-1).reshape(shape)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize(quat_conjugate(q))

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec + quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray,
    t02: Optional[np.ndarray] = None,
    q02: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

def compute_pose_error(t01: np.ndarray,
                    q01: np.ndarray,
                    t02: np.ndarray,
                    q02: np.ndarray,
                    return_type='axa') -> Tuple[np.ndarray, np.ndarray]:
    q10 = quat_inv(q01)
    quat_error = quat_mul(q02, q10)
    pos_error = t02-t01
    if return_type == 'axa':
        quat_error = axis_angle_from_quat(quat_error)
    return pos_error, quat_error


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = np.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = np.where(
        angle.abs() > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + np.pi) % (2 * np.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return np.where((wrapped_angle == 0) & (angles > 0), np.pi, wrapped_angle - np.pi)