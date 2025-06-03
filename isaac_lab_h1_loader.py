import os
from omni.isaac.lab.app import AppLauncher

# Launch Omniverse app
# Headless mode is True for server-side execution, set to False for GUI
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.lab.sim import SimulationContext, schemas
from omni.isaac.lab.assets import AssetCfg, RobotCfg, Robot
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.lab.utils.assets import NVIDIA_ASSETS_DIR

# Define the main function to encapsulate the simulation logic
def main():
    """Main function to set up and run the H1 robot simulation."""

    # Setup simulation context
    # Using modern way to configure simulation settings
    sim = SimulationContext(
        sim_params=schemas.SimCfg(
            dt=0.01,
            physics_engine="physx",
            physics_prim_path="/physicsScene",
            device="cpu", # or "cuda:0" if GPU is available
            gravity=(0.0, 0.0, -9.81),
        ),
        simulation_app=simulation_app
    )

    # Add a default ground plane to the scene
    # The modern way is to add it to the scene configuration
    scene_cfg = InteractiveSceneCfg(num_envs=1) # Default scene configuration
    scene_cfg.ground_plane.enabled = True # Enable ground plane

    # Define AssetCfg for the H1 robot
    # Assuming LEGGED_GYM_ROOT_DIR is the current directory where the script is run.
    # If LEGGED_GYM_ROOT_DIR is not set, default to a placeholder.
    # For this subtask, we assume the script is in the root of the legged_gym repo.
    legged_gym_root_dir = os.getenv("LEGGED_GYM_ROOT_DIR", ".")
    h1_urdf_path = f"{legged_gym_root_dir}/resources/robots/h1/urdf/h1.urdf"

    # Check if the URDF file exists
    if not os.path.exists(h1_urdf_path):
        # Fallback to a known path if the resources are not in the current directory
        # This might happen if LEGGED_GYM_ROOT_DIR is not set correctly or the script is not in the root
        # For robustness, let's try a common Isaac Lab assets path as a fallback for H1 if available
        # This part is speculative as H1 is not a standard Nvidia asset.
        # If the primary path fails, this will likely also fail without correct setup.
        # For now, we will rely on the primary path and let it fail if not found.
        print(f"Error: H1 URDF not found at {h1_urdf_path}. "
              "Please ensure LEGGED_GYM_ROOT_DIR is set correctly or the script is run from the repository root.")
        # As a placeholder, one might use a known NVIDIA asset for testing if H1 is unavailable:
        # h1_urdf_path = f"{NVIDIA_ASSETS_DIR}/Robots/Franka/franka_alt_fingers.urdf"
        # print(f"Warning: H1 URDF not found. Attempting to use default Franka robot: {h1_urdf_path}")
        # if not os.path.exists(h1_urdf_path):
        #     print(f"Error: Default Franka URDF also not found at {h1_urdf_path}.")
        #     simulation_app.close()
        #     return

    # Robot configuration
    h1_robot_cfg = RobotCfg(
        prim_path="/World/H1", # Specifies the prim path for the robot in the USD stage
        asset_cfg=AssetCfg(
            prim_path="{PRIM_PATH}/Asset", # Nested prim path for the asset under the robot's prim path
            asset_path=h1_urdf_path,
            collision_group=-1 # Default collision group
        ),
        init_state=RobotCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), # Initial position (x, y, z)
            rot=(1.0, 0.0, 0.0, 0.0) # Initial orientation as a quaternion (w, x, y, z)
        )
    )

    # Create the interactive scene which handles the environment setup
    scene = InteractiveScene(scene_cfg)
    # Add the robot to the scene
    # The robot will be spawned by the scene based on its configuration
    scene.add_asset("h1_robot", h1_robot_cfg)

    # After adding assets, the scene needs to be initialized
    # This is typically done once after all assets are added.
    # However, for a single robot and ground plane, direct simulation context usage might be simpler.
    # Let's stick to SimulationContext stepping for now if InteractiveScene isn't strictly needed for this simple case,
    # but usually, assets are managed by the scene.
    # For Isaac Lab 0.6+, the scene manages assets and simulation stepping.
    # `sim.reset()` is important to initialize the scene and assets.
    sim.reset()

    # Simulate for a short duration
    num_steps = 100
    for i in range(num_steps):
        # Perform a simulation step
        sim.step()
        # Print progress (optional)
        # if (i + 1) % 10 == 0:
        #     print(f"Simulation step: {i + 1}/{num_steps}")

    print(f"Successfully loaded H1 robot in Isaac Lab and simulated for {num_steps} steps.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
    finally:
        # Close the simulation application
        if simulation_app is not None:
            simulation_app.close()
