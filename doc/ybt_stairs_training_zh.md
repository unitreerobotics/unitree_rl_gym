# YBT 四足：15 cm 楼梯任务（`ybt_stairs`）

平地任务（`ybt`）见 [ybt_training_zh.md](ybt_training_zh.md)。本文只讲楼梯任务。
**平地任务一行代码都没有改**，两个任务可以并行跑、互不影响，见第 9 节。

## 1. 这一步加了什么

新增文件：

| 文件 | 用途 |
| --- | --- |
| `legged_gym/envs/ybt/ybt_stairs_config.py` | 地形几何（单块 + 整张图）与任务配置 `YBTStairsCfg` / `YBTStairsCfgPPO` |
| `legged_gym/envs/ybt/ybt_stairs_env.py` | 楼梯环境 `YBTStairsRobot`，只重写被"地面变高"这件事打破的部分 |
| `legged_gym/scripts/view_stairs.py` | 预览工具：单块楼梯（原样保留）或整张训练地图（`--map`） |
| `tests/test_ybt_stairs.py` | 地形几何、量化精度、奖励改写、注册与配置契约的回归测试（25 项） |
| `doc/ybt_stairs_training_zh.md` | 本说明 |

修改文件（都是**向后兼容**的加法）：

- `legged_gym/envs/ybt/ybt_env.py`：`validate_ybt_asset(cfg, flat_ground_only=True)`。
  平地任务调用时用默认值，行为逐位不变——它仍然会拒绝 `mesh_type != "plane"`。
  楼梯任务传 `flat_ground_only=False`，因为它已经把那两个假设 `z = 0` 的奖励换掉了。
- `legged_gym/envs/__init__.py`：导入并注册 `ybt_stairs`，其余任务不动。

## 2. 概念：楼梯是怎么"放"进仿真里的

Isaac Gym 里没有"楼梯"这种物体，只有两种地形表达方式：

- **heightfield（高度场）**：一个整数二维数组，每个元素就是那一点的地面高度。
  便宜，但台阶的立面只能是一格宽的斜坡，做不出真正的直角。
- **trimesh（三角网）**：由顶点和三角形组成的静态网格，立面可以是垂直的。

楼梯必须用 trimesh，否则 15 cm 的立面会被压成一格（10 cm）宽的斜坡，脚踩上去会打滑。

本仓库原有的 `legged_gym/utils/terrain.py` 是一个上游遗留的 `Terrain` 类，**在本 fork 里是死代码**
（`create_sim()` 只加了地面平面，从没调用过它）。所以楼梯地形是从零写的。

高度场数组的约定（沿用 `isaacgym.terrain_utils`）：

```text
height_field_raw 形状是 (rows, cols)
  rows 沿世界 x 轴  →  楼梯往上爬的方向
  cols 沿世界 y 轴  →  机器人左右的方向
格子 (i, j) 的位置是 x = i * horizontal_scale, y = j * horizontal_scale
格子的整数值 × vertical_scale = 该处地面高度（米）
```

单块楼梯沿 **+x** 爬升：平的助跑段 → `num_steps` 级台阶 → 顶部平台。

```text
z
1.20 m                                  ┌────────────┐  平台
                                        │            │
        ┌────┬────┬────┬────┬────┬────┬─┘            │
        │    │    │    │    │    │    │              │
0.15 m  │    │    │    │    │    │    │              │
   ─────┴────┴────┴────┴────┴────┴────┴──────────────┴─── x
        ↑                                          ↑
   第一级立面 x_first_riser                  最后一级立面 x_top
        └────────── 助跑段 approach_depth ─────────┘
```

## 3. 地图布局：16 层 × 256 道 = 4096 块地形

训练用的不是一块楼梯，而是一张铺满楼梯的"地图"：

- **x 方向**切成 `num_levels = 16` 层（row），一层一个难度。
  第 0 层完全平，第 15 层 15 cm，中间按 1 cm 递增：
  `step_height_i = 0.15 * i / 15`，实测 `[0, 1, 2, ..., 14, 15] cm`。
- **y 方向**切成 `num_lanes = 256` 道（column），一道一个机器人。
  每道宽 1.20 m，整条道铺满台阶（`stair_width = None`），所以相邻道的地面是连续的。

```text
        lane 0   lane 1                     lane 255
      ┌────────┬────────┬────  ...  ────────┬────────┐
层 0  │ 平地   │ 平地   │                 │ 平地   │   <- 0 cm
      ├────────┼────────┼────  ...  ────────┼────────┤
层 1  │ 1cm台阶│ 1cm台阶│                 │ 1cm台阶│
      ├────────┼────────┼────  ...  ────────┼────────┤
 ...  │  ...   │  ...   │                 │  ...   │
      ├────────┼────────┼────  ...  ────────┼────────┤
层 15 │15cm台阶│15cm台阶│                 │15cm台阶│
      └────────┴────────┴────  ...  ────────┴────────┘
```

实测：整张图 **67.2 m × 153.6 m**，高度场 672 × 1536 格，转成 trimesh 后
**1 032 192 个顶点 / 2 059 970 个三角形**。

### 3.1 为什么 `num_envs` 不能像平地那样随便加

**一块地形只能站一个机器人。** 两个机器人分到同一块地形 = 出生时互相穿模，
所以"地形块数 ≥ 机器人"是硬约束，而地形块数又是面积（也就是三角形数）的来源：

| 环境数 | 需 `num_lanes` | 地图宽 | 三角形数 | 实测迭代耗时 | 实测峰值显存 |
| --- | --- | --- | --- | --- | --- |
| **4096（当前默认，已实测）** | **256** | **307.2 m** | **4.12 M** | **1.07–1.23 s** | **4.5 GB** |
| 2048（要更快地调试时退回） | 128 | 153.6 m | 2.06 M | 0.75–0.82 s | — |
| 12288（平地任务的值） | 768 | 921.6 m | 12.4 M | 未测 | 未测 |

4096 环境的实测细节：4096 块地形**互不重复**，每层正好 256 个环境，每个出生点脚下高度 = 0.000 m，
单迭代 294 912 步。每步样本吞吐反而从 2048 环境时的约 61 k/s 升到约 82 k/s，
显存峰值 5.2 GB / 12.3 GB（RTX 4070 SUPER，其中本次训练占 4.5 GB），还有充足余量。

这 4.5 GB 并不是"地形吃掉的"——拆开看，地形只占其中约 0.7 GB，而且是**一次性**的：

| 同样 4096 个机器人 | 峰值显存（减去空闲） |
| --- | --- |
| 平地任务（plane 地面） | 3796 MiB |
| 楼梯任务（4.12 M 三角形） | 4536 MiB |
| **差值 = 地形本身** | **+740 MiB**（约 0.18 KB/三角形） |
| **差值 = 每个机器人** | **约 +0.29 MiB/环境**（平地）；楼梯斜率同量级 |

也就是说：**地形是一次性固定成本，机器人数量才是线性成本。** 2048 环境时地形同样只多出
669 MiB，说明它基本不随机器人数量变化。真正的耦合在第二步：环境数翻倍 → `num_lanes` 翻倍
→ 地图宽翻倍 → 三角形翻倍 → 地形显存也跟着翻倍（约 0.35 GB → 0.7 GB）。

平地任务用 12288 个环境是免费的，因为地面平面本身没有面积代价；楼梯不是。
想加大并行数就**同时**调大 `num_lanes`（保持 `num_levels * num_lanes >= num_envs`），
三角形数按宽度线性增长。作为参照：上游 legged_gym 的 rough terrain 在 4096 环境时约 2.5 M 三角形。

> **只改 `--num_envs` 会直接报错**，因为地图格数没跟着变。错误信息里会算好该填多少：
>
> ```text
> ValueError: ybt_stairs needs one terrain tile per environment, got 8192 envs for
> 4096 tiles (num_levels 16 x num_lanes 256). Raise YBTStairsCfg.terrain.num_lanes
> to at least 512 to match.
> ```

> 顺带说明 `horizontal_scale`：整张图用 **0.10 m/格**（装下 4096 块的折中值），
> 而单块预览地形用 0.05 m/格。两个值是通过 `tile_cfg()` 显式传递的，互不牵连。
> 训练地形里 `step_depth = 0.30` 正好是 3 格、`approach_depth = 0.90` 正好是 9 格，
> **没有任何取整误差**——这很重要，因为奖励里的"脚下第几级台阶"是按位置解析算出来的。

### 3.2 难度是同时训练的，没有 per-env curriculum

4096 个环境**均匀分布**在 16 个难度上（每层 256 个），从第一次迭代起就在同时练最简单的平地
和最难的 15 cm。这是刻意的：

- 一块地形一个机器人 → **每层的机器人数量被该层的地形块数锁死**。想把某个机器人升到更高层，
  就必须把那一层的另一个机器人踢下来，难度分布本身不会变，只是在换位。
- 真正的 curriculum 需要"每层多备一些没人用的地形块"（面积代价成倍）或者中途重建地形
  （PhysX 在 `prepare_sim` 之后不支持换 trimesh）。

所以 `TiledStairsTerrain` 的文档字符串里把这一点写成了**已知限制**，而不是留一个半成品的
`_update_terrain_curriculum`。实际效果上，PPO 会从低难度的显式梯度里学会走路，再逐渐在高层拿到分数，
这也是 quadruped 任务里常用的做法。

## 4. 与平地任务实质不同的地方

`YBTStairsRobot` 继承 `YBTRobot`，只重写"地面不再是 z = 0 的平面"所打破的部分。
其余（URDF 校验、PD 力矩、观测、复位加噪…）全部复用。

### 4.1 出生点（`_get_env_origins` / `_reset_root_states`）

基类 `_create_envs` 会把每个 actor 在 x/y 上再随机撒 **±1 m**。平地无所谓，
但这里一条道只有 1.20 m 宽，撒完就掉进隔壁道了。而且基类在创建时把机器人放在 **z = 0**
（机身埋进地面里），要等第一次 `reset_idx` 才提到 0.55 m。

楼梯环境做两件事：

- `_get_env_origins`：从 4096 块地形里**不重复地**抽 `num_envs` 块，一个环境一块，
  记录 `terrain_levels` / `terrain_types` 和每块的出生点。
- `_reset_root_states`：只加 `spawn_noise_xy = 0.15 m` 的抖动，并在 `__init__` 末尾主动调用一次
  `reset_idx(全部环境)`，把所有机器人一次性摆正到各自的助跑段上，再开始第一步。

实测：256 个环境 → 256 块互不相同的道，16 个难度层全部出现，每个出生点脚下的地面高度都是 **0.000 m**。

### 4.2 两个奖励改成"相对脚下这一级"

地面变高以后，"离地多高"和"机身多高"都不能再用绝对 z 了：

| 奖励 | 平地版本 | 问题 | 楼梯版本 |
| --- | --- | --- | --- |
| `_reward_feet_clearance` | 脚在世界系 z 超过 0.05 m 给分 | 站在升高了的台阶上就等于"抬得很高"，白送分；而且永远不会要求脚跨过下一级立面 | 减去**脚所在位置**那一级的地面高度 |
| `_reward_base_height` | `root_states[:,2]` 与 0.50 比 | 爬到 1.20 m 的平台上时，误差是整整 1.2 m，等于**惩罚爬楼梯** | 减去**机身所在位置**那一级的地面高度 |

两级地面高度都用 `_local_ground_heights(world_x)` 解析求出（不 raycast）：
地形是解析生成的，同一个 x 落在第几级台阶是个闭式函数，末端用 `clamp` 同时兜住
"助跑段之前"和"最后一级之后"两种情况。

第 0 层完全平坦时，两个改写版本与平地原版**数值完全一致**，所以两个任务的曲线可以直接对比
（唯一差别是跳跃处的一级台阶）。

### 4.3 摆动腿抬多高：0.05 → 0.17 m

平地上抬 5 cm 就够（脚部碰撞球半径 0.03 m）；15 cm 的台阶必须把脚抬过立面，否则就是撞上去。
所以 `feet_clearance_target = 0.17 m`、`feet_clearance_cap = 0.27 m`（比一级台阶高一点点）。
这个奖励全程钳制在 ≥ 0，抬和不抬之间没有"索性拖地更划算"的空子。

### 4.4 `_reward_feet_air_time` 的阈值

基类把"这次落地前腾空了多久"与**固定的 0.5 s** 比较。平地上 stride 长，这个参考没问题；
楼梯上 `step_depth = 0.30 m`、指令速度 0.2–0.6 m/s，一只脚的腾空时间大约 0.25 s，
于是**每一次落地都是负分**，等于在花钱让机器人走慢、跳高。

楼梯配置新增 `feet_air_time_positive_threshold = 0.2`，并把权重从继承来的 1.0 降到 0.5。
这是本次改动里**依据最间接的一项**（平地文档里有"该项为负不等于拖沓"的实测记录，
但那是对 go2 平地收敛状态的观察，不是楼梯上的证据）。如果训练后步态显得"跳"或者"拖"，
**第一个应该动的旋钮就是它**。

### 4.5 指令：只往前

| 参数 | 平地 | 楼梯 | 原因 |
| --- | --- | --- | --- |
| `lin_vel_x` | [-0.8, 0.8] | **[0.2, 0.6]** | 任务就是往上爬，倒退/停住没必要 |
| `lin_vel_y` | [-0.4, 0.4] | **[0, 0]** | 一条道只有 1.20 m 宽，横移就掉出去 |
| `ang_vel_yaw` | [-0.8, 0.8] | **[0, 0]** | 同上，原地转就掉出去 |
| `heading` | 随机 | **[0, 0]** | 配合 `heading_command = True` 保持朝 +x |
| `push_robots` | True, 0.5 m/s | **False** | 0.5 m/s 的一推足以把机器人推下 1.2 m 宽的道，学到的是"掉下去"而不是"稳住" |

`heading_command` 仍然开着：偏航由"朝向误差"闭环生成，比只追踪偏航速率稳得多（平地文档里的同一个教训）。

### 4.6 爬完就结束

`terminate_at_top = True`：机身越过最后一级立面（`x > x_top = 3.30 m`，即已经站上平台）就终止该回合。
不终止的话，机器人会继续往前走出自家地形块，掉进下一层地形的助跑段。

### 4.7 相机

`set_camera` 被重写成"看向 0 号环境那块楼梯"。地图横跨 60 多米、机器人的地形块是随机抽的，
基类那个固定机位会对着空气。

## 5. 奖励配置一览

`YBTStairsCfg.rewards.scales`（只列与平地不同的项，其余继承平地）：

| 项 | 平地 | 楼梯 | 说明 |
| --- | --- | --- | --- |
| `tracking_lin_vel` | 1.0 | 1.0 | 主项，不变 |
| `feet_clearance` | 0.3 | **0.3** | 从属项，阈值从 0.05 提到 0.17 m（4.3） |
| `feet_air_time` | 1.0 | **0.5** | 阈值 0.5 s → 0.2 s（4.4） |
| `stumble` | 0（基类里那个 0 权重的 `feet_stumble`） | **-0.2** | 见下 |
| `base_height` | -1.0 | -1.0 | 目标仍是 0.50 m，但改成相对本地台阶（4.2） |
| `collision` | -1.0（继承） | -1.0 | **最该盯的一项**，见第 8 节 |

关于 `stumble` 这个名字有个坑：基类配置里的键叫 `feet_stumble`，但权重非零时环境会去找
`_reward_feet_stumble` —— 而真正存在的函数叫 `_reward_stumble`。基类把它写成 0，
所以从来没有触发过 `AttributeError`。楼梯任务想启用这个奖励，就必须用 `stumble` 这个键名。
`test_every_non_zero_reward_scale_names_a_real_function` 现在会替你把这类错名字挡下来。

## 6. 怎么跑

```bash
conda activate unitree-rl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd ~/unitree_rl_gym
```

**先看地形**（强烈建议，改任何参数前先肉眼确认）：

```bash
# 单块预览楼梯（12 m × 8 m 的样板，就是之前那个）
python legged_gym/scripts/view_stairs.py

# 真正训练用的整张地图：16 层 × 256 道，看最难的 15 cm 那一层
python legged_gym/scripts/view_stairs.py --map

# 只铺 4 道，看某几层的对照
python legged_gym/scripts/view_stairs.py --map --map-lanes=4 --map-level=8

# 不开窗口，只打印参数（CI / 远程用）
python legged_gym/scripts/view_stairs.py --headless --map
```

`--map` 模式下窗口里看到的坐标系和训练**完全一致**（地图不居中），所以别处的
"世界坐标 x = 63.45"就是训练里那个机器人的位置。

**训练**（日志写到 `logs/stairs_ybt/`，和平地的 `logs/flat_ybt/` 完全分开）：

```bash
python legged_gym/scripts/train.py --task=ybt_stairs --headless     # 默认 4096 环境
```

（想改成别的并行数，看 3.1 的 `num_envs` / `num_lanes` 耦合说明。）

**播放 / 导出**（`play.py` 走任务注册表，不需要为楼梯写新脚本）：

```bash
python legged_gym/scripts/play.py --task=ybt_stairs --cmd_vx=0.4
```

**测试**：

```bash
python tests/test_ybt_stairs.py   # 新增的 25 项
python tests/test_ybt.py          # 平地回归，确认没被碰坏
```

## 7. 本次实测结果

| 项目 | 结果 |
| --- | --- |
| 整张地图（默认 4096） | 67.2 m × 307.2 m，高度场 672 × 3072 格（4.1 MB int16） |
| trimesh（默认 4096） | 2 064 384 顶点 / 4 121 282 三角形 |
| 整张地图（2048，作为对照） | 67.2 m × 153.6 m，高度场 672 × 1536 格（2.1 MB int16） |
| trimesh（2048，作为对照） | 1 032 192 顶点 / 2 059 970 三角形，构建 **0.04 s**，约 37 MB |
| 出生点 | 4096 块地形互不重复，每层正好 256 个环境，每个出生点脚下地面高度 = 0.000 m |
| 出生高度 | 全部 z = 0.550 m（平地任务的同一个值） |
| **4096 环境（默认）** | **1.07–1.23 s / 迭代**（98 304 步），2000 次迭代约 **40 分钟**；峰值显存 4.5 GB |
| 2048 环境 | **0.75–0.82 s / 迭代**（49 152 步），2000 次迭代约 **25–27 分钟** |
| 冒烟训练 | 2 次迭代跑通 PPO + 日志；`rew_stumble` 等 14 个非零奖励项全部生效 |
| 回归测试 | `tests/test_ybt_stairs.py` 25 项全过；`tests/test_ybt.py` 10 项全过 |
| 预览窗口 | 单块和 `--map` 两种模式都能开窗并稳定运行 |

冒烟训练里（128 环境；还没学、原地不动）各奖励项的量级：

```text
rew_tracking_lin_vel  +0.008      rew_collision        -0.026   <- 最大负项
rew_tracking_ang_vel  +0.008      rew_lin_vel_z        -0.023
rew_feet_clearance    +0.004      rew_ang_vel_xy       -0.013
rew_stumble           -0.001      rew_torques          -0.008
```

## 8. 已知限制与下一步该动什么

1. **没有 per-env curriculum**（原因见 3.2）。想要难度分布动起来，就得换地形，代价是面积成倍。
2. **`collision = -1.0` 是继承来的值，没有为楼梯调过。** 楼梯上大腿/小腿蹭到立面是常事，
   如果训练后曲线一起不来、机器人贴着台阶磨，先把这一项放松（例如 -0.5）。
3. **`feet_air_time_positive_threshold` 依据最间接**（见 4.4），步态怪就先回到 `0.5`。
4. **`num_envs = 4096` 已实测通过（4.5 GB 显存），但还不是上限。** 要再大就按 3.1 的表
   同时加大 `num_envs` 和 `num_lanes`；只改一个会报错并提示该填多少。
5. **重启训练注意 `resume`**：`YBTStairsCfgPPO.runner.resume = False`，默认从零开始。
   接着上次练要手动把它改成 `True`，不会误接平地的检查点。

## 9. 平地任务没有被改动的证据

- `YBTFlatCfg` / `YBTFlatCfgPPO` / `tests/test_ybt.py`：一个字都没改。
- `ybt_env.py` 只有一处改动，是给 `validate_ybt_asset` 加了个默认 `True` 的参数；
  `YBTRobot` 调用时仍用默认值，`mesh_type != "plane"` 依旧被拒绝。
  测试 `test_flat_task_still_refuses_anything_but_a_plane` 盯着这一点。
- 新增奖励项都写在 `YBTStairsCfg` / `YBTStairsRobot` 里，没有回到基类或平地的类上。
- 日志目录不同（`logs/stairs_ybt/` vs `logs/flat_ybt/`），互不覆盖。
- 跑 `tests/test_ybt.py` 10 项全过。
