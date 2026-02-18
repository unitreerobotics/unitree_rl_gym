
from legged_gym.envs.base.humanoid_robot import HumanoidRobot


class H1_2Robot(HumanoidRobot):
    """H1_2 humanoid robot environment.

    Hip DOF layout (left then right):
      0 - hip yaw,  2 - hip roll,  6 - hip yaw,  8 - hip roll
    """
    hip_indices = [0, 2, 6, 8]
