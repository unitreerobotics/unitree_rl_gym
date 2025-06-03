import argparse
import os
import torch # Good practice, though not directly used for config here

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Import helpers for loading environment and rl_games runner
# These paths might differ based on Isaac Lab version / installation structure
# Assuming omni.isaac.lab_tasks is available for these utilities
RL_GAMES_RUNNER_AVAILABLE = False
try:
    # Attempt to import the RlGamesRunner designed for Isaac Lab tasks
    from omni.isaac.lab_tasks.utils.runners import RlGamesRunner
    RL_GAMES_RUNNER_AVAILABLE = True
    print("Using RlGamesRunner from omni.isaac.lab_tasks.utils.runners")
except ImportError:
    print("Warning: Could not import RlGamesRunner from omni.isaac.lab_tasks. "
          "This script might need adjustments for direct rl_games.torch_runner usage, "
          "or ensure omni.isaac.lab_tasks is correctly installed and in PYTHONPATH.")
    # As a fallback, one might try to use the direct rl_games runner,
    # but it requires manual environment wrapping (e.g., with RlGamesGpuEnv, RlGamesVecEnvWrapper).
    # from rl_games.torch_runner import Runner as RlGamesRunner # Direct rl_games runner needs manual env setup

# Environment specific imports
# Assuming h1_isaac_lab_env.py is in the same directory or PYTHONPATH
try:
    # We only need H1IsaacLabEnvCfg for the entry point definition.
    # The actual environment class H1IsaacLabEnv will be imported by the runner utilities.
    from h1_isaac_lab_env import H1IsaacLabEnvCfg
except ImportError:
    print("Error: Could not import H1IsaacLabEnvCfg from h1_isaac_lab_env.py.")
    print("Please ensure 'h1_isaac_lab_env.py' is in the current directory or accessible in PYTHONPATH.")
    exit(1) # Critical import failed

# Default path for the rl_games configuration file
DEFAULT_RL_GAMES_CONFIG = "./h1_ppo_config.yaml" # Assumes it's in the same directory as this script

def main(args):
    # Launch Isaac Sim application
    # The RlGamesRunner or the environment itself might handle simulation_app context,
    # but it's good practice to initialize it early.
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app # Keep a reference, might be passed to runner or used by env

    # --- RL Games Configuration Check ---
    if not os.path.exists(args.rl_games_config):
        print(f"Error: RL Games config file not found at '{args.rl_games_config}'")
        simulation_app.close()
        return

    # The task name should match what's expected in h1_ppo_config.yaml (e.g., params.config.env_config.name)
    # This is often the registered name of the task in Isaac Lab's registry if used,
    # or simply a descriptive name for the experiment.
    task_name = "H1_IsaacLab" # Example task name, should align with YAML if referenced there

    if not RL_GAMES_RUNNER_AVAILABLE:
        print("Error: Isaac Lab specific RlGamesRunner is not available. Cannot proceed with training.")
        print("Please ensure omni.isaac.lab_tasks is installed and accessible.")
        simulation_app.close()
        return

    # --- Initialize RlGamesRunner (from omni.isaac.lab_tasks.utils.runners) ---
    # This runner is specifically designed for Isaac Lab environments.
    # It handles environment creation, wrapping, and rl_games boilerplate.
    try:
        # Configuration for the RlGamesRunner itself
        runner_cfg_params = {
            "train": True,  # Specify that we are training
            "load_run": args.checkpoint if args.checkpoint else -1, # -1 for last run, or specify path/iteration
            "checkpoint": args.checkpoint if args.checkpoint else None, # Path to checkpoint to load
            "sigma": None,  # Initial noise sigma for actions (if applicable, PPO usually learns sigma)
            # The runner will load the algorithm_config from args.rl_games_config internally
        }

        # The RlGamesRunner from lab_tasks.utils will handle environment creation
        # using the provided task_name and env_cfg_entry_point.
        # It also sets up the VecEnvWrapper (RlGamesVecEnvWrapper).
        runner = RlGamesRunner(
            algo_cfg_path=args.rl_games_config, # Path to the YAML config file for rl_games agent
            runner_cfg=runner_cfg_params,      # Runner specific settings (train, load, checkpoint)
            env_cfg_entry_point=f"h1_isaac_lab_env:H1IsaacLabEnvCfg", # Python path to env config class
            task_name=task_name,               # Name for this task/experiment, used for logging/saving
            # Optional: Pass simulation_app if the runner requires it explicitly.
            # Some runner versions might pick it up from the global AppLauncher context.
            # simulation_app=simulation_app
        )
    except Exception as e:
        print(f"Error initializing RlGamesRunner: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # --- Launch Training ---
    try:
        print(f"Launching training for task: {task_name} with RL Games config: {args.rl_games_config}")
        # The run() method of RlGamesRunner starts the training process.
        # It typically handles the entire training loop, including environment steps,
        # agent updates, logging, and checkpointing.
        runner.run()
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Close the simulation application
        print("Closing simulation application.")
        simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train H1 robot in Isaac Lab using rl_games and PPO.")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run Isaac Sim in headless mode (no GUI)."
    )
    parser.add_argument(
        "--rl_games_config", type=str, default=DEFAULT_RL_GAMES_CONFIG,
        help=f"Path to the rl_games YAML configuration file (default: {DEFAULT_RL_GAMES_CONFIG})."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint directory or specific .pth file to resume training from (e.g., 'runs/MyExperiment/nn/MyExperiment')."
    )
    # Note: num_envs, seed, and other rl_games specific parameters are typically handled
    # by the rl_games configuration file (h1_ppo_config.yaml) or overridden via
    # rl_games' own command-line argument parsing if the RlGamesRunner relays them.
    # For example, rl_games might allow `train.py ... num_envs=XYZ seed=S`.
    # The RlGamesRunner in lab_tasks usually forwards extra unknown args to rl_games.

    cli_args = parser.parse_args()
    main(cli_args)
