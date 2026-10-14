import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Optionally hold the velocity command fixed instead of leaving it random.
    # The environment re-randomizes commands every resampling_time and on every
    # reset, so override the one place that writes them rather than setting
    # env.commands once (that would be undone a few hundred steps later).
    if any(c is not None for c in (args.cmd_vx, args.cmd_vy, args.cmd_wz)):
        fixed_command = torch.tensor(
            [0.0 if c is None else c for c in (args.cmd_vx, args.cmd_vy, args.cmd_wz)],
            device=env.device)

        def _resample_commands(env_ids, _cmd=fixed_command):
            if len(env_ids):
                env.commands[env_ids, :3] = _cmd

        env._resample_commands = _resample_commands
        env.commands[:, :3] = fixed_command
        print("Holding the command fixed at vx=%+.2f vy=%+.2f wz=%+.2f" % tuple(fixed_command.tolist()))

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
