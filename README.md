# Unitree RL GYM

This is a simple example of using Unitree Robots for reinforcement learning, including Unitree Go2, H1, H1_2, G1

## 1. Installation

1. Create a new python virtual env with python 3.8

2. Install pytorch 2.3.1 with cuda-12.1:

   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install Isaac Gym

   - Download and install Isaac Gym Preview 4 from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs isaacgym/docs/index.html
4. Install rsl_rl (PPO implementation)

   - Clone [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
   - `cd rsl_rl && git checkout v1.0.2 && pip install -e .`

5. Install unitree_rl_gym

   - Navigate to the folder `unitree_rl_gym`
   - `pip install -e .`

6. Install unitree_sdk2py (Optional for depoly on real robot)

   - Clone [https://github.com/unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
   - `cd unitree_sdk2_python & pip install -e .`

## 2. Train in Isaac Gym

1. Train:
   `python legged_gym/scripts/train.py --task=go2`

   * To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
   * To run headless (no rendering) add `--headless`.
   * **Important** : To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
   * The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
   * The following command line arguments override the values set in the config files:
   * --task TASK: Task name.
   * --resume: Resume training from a checkpoint
   * --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
   * --run_name RUN_NAME: Name of the run.
   * --load_run LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
   * --checkpoint CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
   * --num_envs NUM_ENVS: Number of environments to create.
   * --seed SEED: Random seed.
   * --max_iterations MAX_ITERATIONS: Maximum number of training iterations.
2. Play:`python legged_gym/scripts/play.py --task=go2`

   * By default, the loaded policy is the last model of the last run of the experiment folder.
   * Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### 2.1 Play Demo


| Go2 | G1 | H1 | H1_2 |
|--- | --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/98395d82-d3f6-4548-b6ee-8edfce70ac3e" controls="controls" width="400px"></video>  |  <video src="https://github.com/user-attachments/assets/2d2c2ea9-7816-4754-a3f0-d6d23912c569" controls="controls" width="400px"></video>  |  <video src="https://github.com/user-attachments/assets/622c8e3f-e82d-413a-b0ff-b0aca0fa22e5" controls="controls" width="400px"></video> | <video src="https://github.com/user-attachments/assets/1ec25467-4101-49c1-8c35-5c109e49e81e" controls="controls" width="400px"></video> |

## 3. Sim in Mujoco

### 3.1 Mujoco Usage

To execute sim2sim in mujoco, execute the following command:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py {config_name}
```

`config_name`: The file name of the configuration file. The configuration file will be found under `deploy/deploy_mujoco/configs/`, for example `g1.yaml`, `h1.yaml`, `h1_2.yaml`.

**example**:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### 3.2 Mujoco Demo

| G1 | H1 | H1_2 |
|--- | --- | --- |
| <video src="https://github.com/user-attachments/assets/9455b595-791e-4715-b280-0e70b9d45c53" controls="controls" width="400px"></video>  |  <video src="https://github.com/user-attachments/assets/2ff75b99-0186-4195-9ec8-69901e7b6700" controls="controls" width="400px"></video>  |  <video src="https://github.com/user-attachments/assets/8e3476ff-3d5c-45d4-a227-7565bf885d93" controls="controls" width="400px"></video> |

## 4. Depoly on Physical Robot

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/c5667475f51844628911cf032509d80a_1920x1080.png" width="400px">](https://oss-global-cdn.unitree.com/static/621806fb837c4f869e5c59efd1d93105.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/42d2332dc3004097896f33d0db027039_1920x1080.png" width="400px">](https://oss-global-cdn.unitree.com/static/9c61509fc4f74d21bb707a5fe3ae11aa.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/c49a03fa297a4d178ec3a5b01b9c0bbf_1920x1080.png" width="400px">](https://oss-global-cdn.unitree.com/static/e60a0fcd829e417f92a88e78463a695d.mp4) |

reference to [Deploy on Physical Robot(English)](deploy/deploy_real/README.md) | [实物部署（简体中文）](deploy/deploy_real/README.zh.md)
