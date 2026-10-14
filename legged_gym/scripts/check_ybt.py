"""Headless end-to-end YBT check: physics, PPO, checkpoint, and policy export.

Example: python legged_gym/scripts/check_ybt.py --num_envs=64 --max_iterations=3
Artifacts go to logs/ybt_smoke, not the production flat_ybt experiment.
"""
import sys
from pathlib import Path

import isaacgym  # must precede torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.ybt.ybt_config import YBTFlatCfg, YBTFlatCfgPPO
from legged_gym.utils import export_policy_as_jit, get_args


def check():
    if not any(a == "--task" or a.startswith("--task=") for a in sys.argv[1:]):
        sys.argv.append("--task=ybt")
    args = get_args()
    if args.task != "ybt":
        raise ValueError("This checker is only for --task=ybt")
    if args.resume:
        raise ValueError("Run the checker without --resume; it creates its own test checkpoint")
    args.headless = True
    if args.num_envs is None:
        args.num_envs = 64
    if args.max_iterations is None:
        args.max_iterations = 3
    if args.num_envs < 1 or args.max_iterations < 1:
        raise ValueError("num_envs and max_iterations must be positive")
    cfg = YBTFlatCfg()
    cfg.seed = 1
    env, _ = task_registry.make_env("ybt", args=args, env_cfg=cfg)
    try:
        ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(ids)
        if not torch.all((env.dof_pos >= env.dof_pos_limits[:, 0])
                         & (env.dof_pos <= env.dof_pos_limits[:, 1])):
            raise AssertionError("Reset positions violate joint limits")
        obs, _ = env.reset()
        if obs.shape != (env.num_envs, 48) or env.feet_indices.numel() != 4:
            raise AssertionError("Unexpected observation shape or foot count")
        print("Loaded DOF/action order:", env.dof_names)
        print("Feet:", env.feet_indices.tolist(), "observation shape:", tuple(obs.shape))
        for _ in range(100):
            obs, _, rewards, _, _ = env.step(torch.zeros(env.num_envs, 12, device=env.device))
            for tensor in (obs, rewards, env.root_states, env.dof_state, env.contact_forces):
                if not torch.isfinite(tensor).all():
                    raise AssertionError("Physics produced non-finite state/rewards")
            if not torch.all(env.torques.abs() <= env.torque_limits + 1e-5):
                raise AssertionError("Applied torque exceeds the URDF effort limit")
        train_cfg = YBTFlatCfgPPO()
        train_cfg.runner.experiment_name = "ybt_smoke"
        train_cfg.runner.run_name = "check{}".format(env.num_envs)
        # Ignore experiment/resume CLI overrides: never load/write a production run.
        args.experiment_name = "ybt_smoke"
        args.run_name = train_cfg.runner.run_name
        runner, _ = task_registry.make_alg_runner(
            env, args=args, train_cfg=train_cfg,
            log_root=str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / "ybt_smoke"))
        before = {key: tensor.detach().clone() for key, tensor in runner.alg.actor_critic.actor.state_dict().items()}
        runner.learn(args.max_iterations, init_at_random_ep_len=True)
        actor = runner.alg.actor_critic.actor
        if not any(not torch.equal(before[key], tensor) for key, tensor in actor.state_dict().items()):
            raise AssertionError("PPO did not update actor weights")
        if not all(torch.isfinite(tensor).all() for tensor in runner.alg.actor_critic.parameters()):
            raise AssertionError("PPO produced non-finite weights")
        checkpoint = Path(runner.log_dir) / "model_{}.pt".format(runner.current_learning_iteration)
        runner.load(str(checkpoint))
        # The runner steps the environment under inference_mode. Continue
        # in that mode because its action buffers are inference tensors.
        with torch.inference_mode():
            obs, _ = env.reset()
            actions = runner.get_inference_policy(device=env.device)(obs)
        if actions.shape != (env.num_envs, 12) or not torch.isfinite(actions).all():
            raise AssertionError("Reloaded policy produced invalid actions")
        export_dir = Path(runner.log_dir) / "exported" / "policies"
        export_policy_as_jit(runner.alg.actor_critic, str(export_dir))
        exported = torch.jit.load(str(export_dir / "policy_1.pt"), map_location="cpu")
        with torch.no_grad():
            exported_actions = exported(obs[:4].cpu())
        if not torch.allclose(exported_actions, actions[:4].cpu(), atol=1e-4, rtol=1e-4):
            raise AssertionError("Exported actor differs from checkpoint inference")
        for _ in range(50):
            with torch.inference_mode():
                actions = runner.get_inference_policy(device=env.device)(obs)
                obs, _, rewards, _, _ = env.step(actions)
            if not torch.isfinite(obs).all() or not torch.isfinite(rewards).all():
                raise AssertionError("Reloaded policy rollout produced non-finite values")
        print("PASS: {} environments; physics + {} PPO iterations + save/reload + TorchScript rollout".format(
            env.num_envs, args.max_iterations))
        print("Checkpoint:", checkpoint)
        print("Export:", export_dir / "policy_1.pt")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    check()
