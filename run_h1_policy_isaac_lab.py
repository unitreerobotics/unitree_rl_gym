import torch
import os
import argparse

# Isaac Lab imports
from omni.isaac.lab.app import AppLauncher

# Environment specific imports
# Assuming h1_isaac_lab_env.py is in the same directory or PYTHONPATH
try:
    from h1_isaac_lab_env import H1IsaacLabEnv, H1IsaacLabEnvCfg
except ImportError:
    print("Error: Could not import H1IsaacLabEnv or H1IsaacLabEnvCfg.")
    print("Ensure 'h1_isaac_lab_env.py' is in the same directory or accessible in PYTHONPATH.")
    exit(1)

# Configuration Variables (can be overridden by command-line arguments)
DEFAULT_POLICY_PATH = "./deploy/pre_train/h1/motion.pt" # Relative to repo root
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_ENABLE_GUI = True
DEFAULT_NUM_ENVS_PLAYBACK = 1 # Running a single environment for policy playback

def main(args):
    """Main function to load and run a pre-trained H1 policy."""

    print(f"Using device: {args.device}")
    print(f"Policy path: {args.policy_path}")
    print(f"GUI enabled: {args.enable_gui}")
    print(f"Number of environments for playback: {args.num_envs}")

    # Initialize Simulation
    # AppLauncher handles the simulation context setup internally for RLTask based envs
    app_launcher = AppLauncher(headless=not args.enable_gui)
    simulation_app = app_launcher.app # Get the simulation app instance

    # Load Policy
    if not os.path.exists(args.policy_path):
        print(f"Error: Policy file not found at '{args.policy_path}'.")
        print("Please ensure the path is correct and the policy file exists.")
        simulation_app.close()
        return

    try:
        policy = torch.jit.load(args.policy_path).to(args.device)
        policy.eval() # Set the policy to evaluation mode
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Error loading the policy: {e}")
        simulation_app.close()
        return

    # Configure and Instantiate Environment
    env_cfg = H1IsaacLabEnvCfg() # Use the default configuration from h1_isaac_lab_env
    env_cfg.num_envs = args.num_envs # Override number of environments for playback

    # If running headless, some rendering-related settings might need adjustment
    # For example, if the environment has specific rendering settings in its config:
    # if not args.enable_gui:
    #     if hasattr(env_cfg.scene, 'viewer'): # Example path to viewer settings
    #         env_cfg.scene.viewer.headless = True
    #     # Disable any other GUI-specific features if necessary

    try:
        # RLTask based environments typically handle their own sim context within their constructor
        # when `simulation_app` is available globally or passed correctly.
        # AppLauncher makes `simulation_app` available.
        env = H1IsaacLabEnv(cfg=env_cfg)
        print("Environment H1IsaacLabEnv instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating the environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # Simulation Loop
    try:
        # Reset the environment to get initial observations
        # RLTask.reset() returns a dictionary of observations
        obs_dict = env.reset()
        # The actual observations tensor is under the key "observations"
        obs = obs_dict["observations"].to(args.device)
        print("Environment reset and initial observations received.")

        step_count = 0
        max_steps = 10000 # Limit simulation steps for testing if needed, or run indefinitely

        while simulation_app.is_running() and step_count < max_steps:
            if args.enable_gui:
                # Update the simulation app, which handles rendering and other background tasks
                simulation_app.update()

            # Get action from the policy
            with torch.no_grad(): # Disable gradient calculations for inference
                action = policy(obs)

            # Step the environment with the computed action
            # RLTask.step() returns: next_obs_dict, rewards, terminated, truncated, info
            next_obs_dict, rewards, terminated, truncated, info = env.step(action)

            # Update current observation for the next iteration
            obs = next_obs_dict["observations"].to(args.device)
            step_count += 1

            # Check for termination or truncation for any environment
            if terminated.any() or truncated.any():
                print(f"Episode terminated or truncated at step {step_count}. Resetting...")
                # Reset only the environments that are done if using >1 envs,
                # but for NUM_ENVS_PLAYBACK=1, this effectively resets the single env.
                # The reset logic within RLTask should handle resetting only terminated envs.
                obs_dict = env.reset() # RLTask's reset should handle which envs to reset
                obs = obs_dict["observations"].to(args.device)
                print("Environment reset.")

            # Optional: Print step count or other info periodically
            # if step_count % 100 == 0:
            #     print(f"Simulation step: {step_count}")

    except Exception as e:
        print(f"An error occurred during the simulation loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Close the simulation application
        print("Closing simulation application.")
        simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H1 pre-trained policy in Isaac Lab.")
    parser.add_argument(
        "--policy_path", type=str, default=DEFAULT_POLICY_PATH,
        help=f"Path to the TorchScript policy file (default: {DEFAULT_POLICY_PATH})"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help=f"Device to use for Torch (e.g., 'cuda:0' or 'cpu') (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--enable_gui", type=bool, default=DEFAULT_ENABLE_GUI,
        help=f"Enable GUI rendering (default: {DEFAULT_ENABLE_GUI})"
    )
    parser.add_argument(
        "--num_envs", type=int, default=DEFAULT_NUM_ENVS_PLAYBACK,
        help=f"Number of environments to run for playback (default: {DEFAULT_NUM_ENVS_PLAYBACK})"
    )

    cli_args = parser.parse_args()

    # Ensure GUI is disabled if running with more than a few environments for performance
    if cli_args.num_envs > 1 and cli_args.enable_gui:
        print(f"Warning: Running {cli_args.num_envs} environments with GUI enabled might be slow. Consider disabling GUI for >1 envs.")

    main(cli_args)
