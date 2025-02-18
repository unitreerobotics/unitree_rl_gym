import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

from icecream import ic
import deploy.deploy_real.math_utils as math_utils


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    

    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        isaaclab_joint_order = config["lab_joint"]

        raw_joint_order = config["motor_joint"]
        mapping_tensor = torch.zeros((len(raw_joint_order), len(isaaclab_joint_order)))

        # Fill the mapping tensor
        for b_idx, b_joint in enumerate(raw_joint_order):
            if b_joint in isaaclab_joint_order:
                a_idx = isaaclab_joint_order.index(b_joint)
                mapping_tensor[a_idx, b_idx] = 1.0

        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        target_cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # create observation
                
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                position = d.qpos[:3]
                # position is given on world frame.
                # ic(position)
                # ic(cmd)
                cmd[0] = target_cmd[0] - position[0]
                cmd[1] = target_cmd[1] - position[1]
                # cmd[2] -= position[2]
                cmd[:3] = math_utils.quat_rotate_inverse(
                    math_utils.yaw_quat(torch.as_tensor(quat).float().to("cpu")),
                    torch.as_tensor(cmd[:3]).float().to("cpu")
                ).float().numpy()


                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:10] = cmd * cmd_scale
                obs[10 : 10 + num_actions] = qj
                obs[10 + num_actions : 10 + 2 * num_actions] = dqj
                obs[10 + 2 * num_actions : 10 + 3 * num_actions] = action
                # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                obs_tensor[..., 10:39] = obs_tensor[..., 10:39] @ mapping_tensor.transpose(0, 1)
                obs_tensor[..., 39:68] = obs_tensor[..., 39:68] @ mapping_tensor.transpose(0, 1)
                obs_tensor[..., 68:97] = obs_tensor[..., 68:97] @ mapping_tensor.transpose(0, 1)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # reordered_actions = action @ mapping_tensor.detach().cpu().numpy()
                # applied_action = action @ mapping_tensor.detach().cpu().numpy()
                action = action @ mapping_tensor.detach().cpu().numpy()
                # ic(
                #     action
                # )
                # action = 0.

                # transform action to target_dof_pos
                # target_dof_pos = action * action_scale + default_angles
                target_dof_pos = action * action_scale + default_angles
                # raise NotImplementedError

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
