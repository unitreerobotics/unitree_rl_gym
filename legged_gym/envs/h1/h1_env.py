
from legged_gym.envs.base.humanoid_robot import HumanoidRobot


class H1Robot(HumanoidRobot):
    """H1 humanoid robot environment.

    Hip DOF layout (left then right):
      0 - hip yaw,  1 - hip roll,  5 - hip yaw,  6 - hip roll
    """
    hip_indices = [0, 1, 5, 6]
