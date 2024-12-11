# 实物部署

本代码可以在实物部署训练的网络。目前支持的机器人包括 Unitree G1, H1, H1_2。

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

## 启动用法

```bash
python deploy_real.py {net_interface} {config_name}
```

- `net_interface`: 为连接机器人的网卡的名字，例如`enp3s0`
- `config_name`: 配置文件的文件名。配置文件会在 `deploy/deploy_real/configs/` 下查找， 例如`g1.yaml`, `h1.yaml`, `h1_2.yaml`。

## 启动过程

### 1. 启动机器人

将机器人在吊装状态下启动，并等待机器人进入 `零力矩模式`

### 2. 进入调试模式

确保机器人处于 `零力矩模式` 的情况下，按下遥控器的 `L2+R2`组合键；此时机器人会进入`调试模式`, `调试模式`下机器人关节处于阻尼状态。

### 3. 连接机器人

使用网线连接自己的电脑和机器人上的网口。修改网络配置如下

<img src="https://doc-cdn.unitree.com/static/2023/9/6/0f51cb9b12f94f0cb75070d05118c00a_980x816.jpg" width="400px">

然后使用 `ifconfig` 命令查看与机器人连接的网卡的名称。网卡名称记录下来，后面会作为启动命令的参数

<img src="https://oss-global-cdn.unitree.com/static/b84485f386994ef08b0ccfa928ab3830_825x484.png" width="400px">

### 4. 启动程序

假设目前与实物机器人连接的网卡名为`enp3s0`.以G1机器人为例，执行下面的命令启动

```bash
python deploy_real.py enp3s0 g1.yaml
```

#### 4.1 零力矩状态

启动之后，机器人关节会处于零力矩状态，可以用手晃动机器人的关节感受并确认一下。

#### 4.2 默认位置状态

在零力矩状态时，按下遥控器上的`start`按键，机器人会运动到默认关节位置状态。

在机器人运动到默认关节位置之后，可以缓慢的下放吊装机构，让机器人的脚与地面接触。

#### 4.3 运动控制模式

准备工作完成，按下遥控器上`A`键，机器人此时会原地踏步，在机器人状态稳定之后，可以逐渐降低吊装绳，给机器人一定的活动空间。

此时使用遥控器上的摇杆就可以控制机器人的运动了。
左摇杆的前后，控制机器人的x方向的运动速度
左摇杆的左右，控制机器人的y方向的运动速度
右摇杆的左右，控制机器人的偏航角yaw的运动速度

#### 4.4 退出控制

在运动控制模式下，按下遥控器上 `select` 按键，机器人会进入阻尼模式倒下，程序退出。或者在终端中 使用 `ctrl+c` 关闭程序。

> 注意：
> 
> 由于本示例部署并非稳定的控制程序，仅用于示例作用，请控制过程尽量不要给机器人施加扰动，如果控制过程中出现任何意外情况，请及时退出控制，以免发生危险。

## 视频教程

部署在G1上：[https://oss-global-cdn.unitree.com/static/ff70a257ddf34adbb67733d8a90b24d4.mp4](https://oss-global-cdn.unitree.com/static/ff70a257ddf34adbb67733d8a90b24d4.mp4)

部署在H1上：[https://oss-global-cdn.unitree.com/static/6dae1756f4214409a5ced7386ca011ae.mp4](https://oss-global-cdn.unitree.com/static/6dae1756f4214409a5ced7386ca011ae.mp4)

部署在H1_2上：[https://oss-global-cdn.unitree.com/static/9d61a1470d3e4b9a9af6d131324fcb94.mp4](https://oss-global-cdn.unitree.com/static/9d61a1470d3e4b9a9af6d131324fcb94.mp4)