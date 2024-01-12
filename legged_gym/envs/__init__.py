
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry
task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", LeggedRobot, H1RoughCfg(), H1RoughCfgPPO())


