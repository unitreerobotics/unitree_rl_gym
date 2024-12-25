# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 18.04 或更高版本  
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 525 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree-rl python=3.8
```

### 1.3 激活虚拟环境

```bash
conda activate unitree-rl
```

---

## 2. 安装依赖

### 2.1 安装 PyTorch

PyTorch 是一个神经网络计算框架，用于模型训练和推理。使用以下命令安装：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 安装 Isaac Gym

Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

#### 2.2.1 下载

从 Nvidia 官网下载 [Isaac Gym](https://developer.nvidia.com/isaac-gym)。

#### 2.2.2 安装

解压后进入 `isaacgym/python` 文件夹，执行以下命令安装：

```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 验证安装

运行以下命令，若弹出窗口并显示 1080 个球下落，则安装成功：

```bash
cd examples
python 1080_balls_of_solitude.py
```

如有问题，可参考 `isaacgym/docs/index.html` 中的官方文档。

### 2.3 安装 rsl_rl

`rsl_rl` 是一个强化学习算法库。

#### 2.3.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
```

#### 2.3.2 切换分支

切换到 v1.0.2 分支：

```bash
cd rsl_rl
git checkout v1.0.2
```

#### 2.3.3 安装

```bash
pip install -e .
```

### 2.4 安装 unitree_rl_gym

#### 2.4.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
```

#### 2.4.2 安装

进入目录并安装：

```bash
cd unitree_rl_gym
pip install -e .
```

### 2.5 安装 unitree_sdk2py（可选）

`unitree_sdk2py` 是用于与真实机器人通信的库。如果需要将训练的模型部署到物理机器人上运行，可以安装此库。

#### 2.5.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
```

#### 2.5.2 安装

进入目录并安装：

```bash
cd unitree_sdk2_python
pip install -e .
```

---

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

