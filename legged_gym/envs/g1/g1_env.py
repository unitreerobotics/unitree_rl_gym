
from legged_gym.envs.base.humanoid_robot import HumanoidRobot


class G1Robot(HumanoidRobot):
    """G1 humanoid robot environment.

    Hip DOF layout (left then right):
      1 - hip roll,  2 - hip pitch,  7 - hip roll,  8 - hip pitch
    """
    hip_indices = [1, 2, 7, 8]
