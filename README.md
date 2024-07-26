# Unitree RL GYM

This is a simple example of using Unitree Robots for reinforcement learning, including Unitree Go2, H1, H1_2, G1

### Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:

   ```
   pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

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

### Usage

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

### Robots Demo

1. Go2

https://github.com/user-attachments/assets/98395d82-d3f6-4548-b6ee-8edfce70ac3e

2. H1

https://github.com/user-attachments/assets/a9475a63-ea06-4327-bfa6-6a0f8065fa1c

3. H1-2

https://github.com/user-attachments/assets/d6cdee70-8f8a-4a50-b219-df31b269b083

4. G1

https://github.com/user-attachments/assets/0b554137-76bc-43f9-97e1-dd704a33d6a9
