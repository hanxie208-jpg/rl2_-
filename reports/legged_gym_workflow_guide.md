# legged_gym 四足机器人训练与 Sim2Sim 工作流程指南

## 0. 项目背景与目标

- 框架：`legged_gym`，基于 NVIDIA Isaac Gym。
- 当前工作区：`/home/xiehan/下载/workspace_summer`
- 当前仓库：`/home/xiehan/下载/workspace_summer/rl_stack/legged_gym`
- 当前任务名：`dog_flat`
- 当前机器人：自研 URDF 狗。
- 当前 URDF：`/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/resources/robots/dog/urdf/dog.urdf`
- 训练环境：Isaac Gym。
- 验证目标：训练平地行走策略，并把导出的策略部署到 MuJoCo 中做 Sim2Sim 验证视频。

建议先把目标拆成三步：

1. Isaac Gym 中能稳定站立和前进。
2. Isaac Gym 中播放策略，确认步态、方向、姿态和足端接触合理。
3. 导出 Actor 策略，在 MuJoCo 中复现同样的控制周期、PD 参数、默认关节角和动作缩放。

当前项目的 Python 前缀建议统一使用：

```bash
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python
```

## 1. 训练前准备：修改配置文件

### 1.1 核心配置文件

`legged_gym` 的基础配置在：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/base/legged_robot_config.py
```

基础环境逻辑在：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/base/legged_robot.py
```

已有机器人示例目录：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/a1/
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/anymal_c/
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/cassie/
```

当前自研狗目录：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/dog/
```

主要文件：

```text
legged_gym/envs/dog/dog.py
legged_gym/envs/dog/dog_config.py
```

### 1.2 为自研 URDF 创建配置

通用做法是复制一个已有机器人配置作为模板，例如 A1：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
mkdir -p legged_gym/envs/dog
cp legged_gym/envs/a1/a1_config.py legged_gym/envs/dog/dog_config.py
```

如果只需要基础四足逻辑，可以先使用 `LeggedRobot`；如果要自定义奖励、观测、终止条件、日志指标，就创建自己的环境类，例如当前项目的：

```text
legged_gym/envs/dog/dog.py
```

当前注册入口在：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/legged_gym/envs/__init__.py
```

当前已经注册：

```python
from .dog.dog import DogRobot
from .dog.dog_config import DogFlatCfg, DogFlatCfgPPO

task_registry.register("dog_flat", DogRobot, DogFlatCfg(), DogFlatCfgPPO())
```

这里的 `"dog_flat"` 就是训练和播放命令里的 `--task=dog_flat`。

### 1.3 asset 部分

位置：

```text
legged_gym/envs/dog/dog_config.py
```

当前项目示例：

```python
class asset(LeggedRobotCfg.asset):
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
    name = 'dog'
    foot_name = 'foot'
    penalize_contacts_on = ['hip', 'thigh', 'calf']#跨关节,大腿,小腿
    terminate_after_contacts_on = ['base', 'hip']#机体,跨关节
    collapse_fixed_joints = False
    flip_visual_attachments = False
    self_collisions = 1
```

关键含义：

- `file`：URDF 路径。必须指向真实存在的 URDF。
- `foot_name`：足端 body 名称关键词。`legged_gym` 会用它找足端刚体，用于接触检测、足端奖励、步态统计。
- `penalize_contacts_on`：这些 body 发生接触会扣分。
- `terminate_after_contacts_on`：这些 body 发生接触会终止 episode。
- `collapse_fixed_joints`：如果足端是 fixed joint 连接，建议先设为 `False`，否则 foot body 可能被折叠消失，接触检测会出问题。
- `flip_visual_attachments`：如果模型显示方向怪，检查这个参数。

### 1.4 init_state 部分

当前项目示例：

```python
class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.35]
    rot = [0.0, 0.0, 0.0, 1.0]
    lin_vel = [0.0, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
    default_joint_angles = {
        'FL_hip_joint': 0.1,
        'FL_thigh_joint': 0.9,
        'FL_calf_joint': -1.55,
        'FR_hip_joint': -0.1,
        'FR_thigh_joint': 0.9,
        'FR_calf_joint': -1.55,
        'RL_hip_joint': 0.1,
        'RL_thigh_joint': 0.9,
        'RL_calf_joint': -1.55,
        'RR_hip_joint': -0.1,
        'RR_thigh_joint': 0.9,
        'RR_calf_joint': -1.55,
    }
```

关键含义：

- `pos[2]` 是出生高度，太低会一出生就接触机身，太高会自由落体后摔倒。
- `default_joint_angles` 是默认站姿，也是 PD 控制目标的基准。
- 这些关节名必须和 URDF 里的 joint name 完全一致。
- MuJoCo Sim2Sim 时必须使用同一套默认关节角，否则策略输入和动作含义会变。

### 1.5 control 部分

当前项目示例：

```python
class control(LeggedRobotCfg.control):
    control_type = 'P'
    stiffness = {'joint': 32.0}
    damping = {'joint': 2.4}
    action_scale = 0.22
    decimation = 4
```

关键含义：

- `control_type='P'`：策略输出动作会变成目标关节角偏移，底层用 PD 控制。
- `control_type='T'`：策略直接输出力矩，难度更高。
- `stiffness`：P 增益，越大越硬。
- `damping`：D 增益，越大越抑制抖动。
- `action_scale`：动作幅度。太小抬不起腿，太大容易乱踢或跳。
- `decimation`：一个动作保持多少个仿真步。会影响控制频率，也必须在 MuJoCo 里对齐。

### 1.6 commands 部分

当前项目示例：

```python
class commands(LeggedRobotCfg.commands):
    heading_command = False
    resampling_time = 4.0

    class ranges(LeggedRobotCfg.commands.ranges):
        lin_vel_x = [0.22, 0.28]
        lin_vel_y = [0.0, 0.0]
        ang_vel_yaw = [0.0, 0.0]
```

关键含义：

- `lin_vel_x`：前进速度命令范围。
- `lin_vel_y`：横向速度命令。第一次平地直走建议设为 `[0.0, 0.0]`。
- `ang_vel_yaw`：转向速度命令。第一次直走建议设为 `[0.0, 0.0]`。
- `resampling_time`：多久重新采样一次命令。

注意：基础代码里小于等于约 `0.2 m/s` 的速度命令可能会被置零，所以平地直走命令不要设得太低。

### 1.7 rewards 部分

当前项目重点：

```python
class rewards(LeggedRobotCfg.rewards):
    only_positive_rewards = True
    base_height_target = 0.35
    tracking_sigma = 0.12

    class scales(LeggedRobotCfg.rewards.scales):
        termination = -300.0
        world_velocity_tracking = 2.2
        stable_command_speed = 2.0
        commanded_forward_motion = 1.8
        step_forward_progress = 2.2
        low_base_height = -8.0
        all_feet_contact = -8.0
        diagonal_clock_gait = 0.75
        trot_joint_pose = 4.0
        front_swing_pose = 2.2
        lateral_drift = -1.0
        yaw_rate = -1.3
        yaw_drift = -1.1
```

关键含义：

- `tracking_lin_vel` / `world_velocity_tracking`：速度跟踪奖励。
- `step_forward_progress`：每一步真实向世界 X 正方向前进的奖励。
- `low_base_height`：机身过低惩罚，防止趴地拱。
- `all_feet_contact`：四脚长期贴地惩罚，防止小碎步蹭地。
- `diagonal_clock_gait`：对角步态接触节奏奖励。
- `trot_joint_pose` / `front_swing_pose`：当前项目自定义的对角小跑关节姿态先验。
- `lateral_drift` / `yaw_rate` / `yaw_drift`：压制横漂和转圈。
- `termination`：失败终止惩罚。注意 `legged_gym` 会把 reward scale 乘以 `dt`，所以 `-300` 在 `dt=0.02` 时约等价于一次失败扣 `-6`。

不要只看总 reward。四足训练里最容易出现的问题是策略通过奇怪动作刷分，例如趴地拱、后腿猛推、转圈前进、四脚贴地蹭。

### 1.8 PPO 训练配置

位置：

```text
legged_gym/envs/dog/dog_config.py
```

当前项目示例：

```python
class DogFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.7
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.0008

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'step34_moderate_height_trot'
        experiment_name = 'flat_dog'
        load_run = -1
        max_iterations = 600
        save_interval = 25
```

常见关注项：

- `init_noise_std`：初始动作探索噪声。
- `entropy_coef`：探索强度，太大可能一直乱动，太小可能过早卡住。
- `learning_rate`：学习率，在基类或算法配置中设置。
- `num_learning_epochs`：每轮采样后更新几轮。
- `num_mini_batches`：mini-batch 数量。
- `run_name`：实验名称后缀。
- `experiment_name`：日志大目录。
- `max_iterations`：默认训练轮数。
- `save_interval`：每隔多少轮保存一次模型。

## 2. 启动训练

### 2.1 基本训练命令

进入仓库：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
```

启动训练：

```bash
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py --task=dog_flat --headless --max_iterations=300
```

推荐把终端输出保存下来：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py --task=dog_flat --headless --max_iterations=300 2>&1 | tee /home/xiehan/下载/workspace_summer/rl_stack/legged_gym/logs/flat_dog_step34_moderate_height_trot_console.log
```

### 2.2 参数解释

- `--task=dog_flat`：任务名，必须和 `legged_gym/envs/__init__.py` 中注册的名称一致。
- `--headless`：不打开 viewer，训练更快，更适合长时间训练。
- `--num_envs=384`：覆盖配置中的并行环境数量。数量越大采样越快，但显存占用越高。
- `--max_iterations=300`：覆盖配置中的最大迭代轮数。
- `--sim_device=cuda:0`：仿真设备。
- `--rl_device=cuda:0`：神经网络训练设备。
- `--resume`：从已有 run 恢复训练。
- `--experiment_name=flat_dog`：实验大目录。
- `--run_name=xxx`：本次训练名称。
- `--load_run=Aug14_17-25-01_step34_moderate_height_trot`：指定从哪个 run 加载。
- `--checkpoint=200`：指定加载 `model_200.pt`。

恢复训练示例：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py --task=dog_flat --headless --resume --load_run=Aug14_17-25-01_step34_moderate_height_trot --checkpoint=200 --max_iterations=600
```

### 2.3 日志和模型保存位置

当前项目日志目录：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/logs/flat_dog/
```

一次训练会生成类似：

```text
logs/flat_dog/Aug14_17-25-01_step34_moderate_height_trot/
```

里面包含：

```text
events.out.tfevents...
model_0.pt
model_25.pt
model_50.pt
...
model_300.pt
```

## 3. 解读终端输出

训练终端会反复打印每一轮的统计信息。需要重点看下面几类。

### 3.1 Iteration

示例：

```text
Learning iteration 200/300
```

表示当前第 200 轮，总共计划 300 轮。

### 3.2 Mean reward

示例：

```text
Mean reward: 176.49
```

一般希望总体上升，但不能只看它。因为策略可能通过异常动作刷 reward。

### 3.3 Mean episode length

示例：

```text
Mean episode length: 926.5
```

越接近最大 episode 长度，通常说明机器人越不容易摔倒。当前项目中长度接近 `1000` 时，说明很多 episode 能撑到超时结束。

### 3.4 PPO 损失

常见字段：

```text
Value function loss
Surrogate loss
Mean action noise std
```

观察方式：

- `Value function loss` 太大且持续爆炸，可能训练不稳定。
- `Surrogate loss` 不需要单独追求越小越好，主要看是否异常震荡。
- `Mean action noise std` 从 `0.7` 慢慢降到 `0.6`、`0.5` 一般是正常的。一直很大说明还在强探索，太快接近 0 可能过早收敛。

### 3.5 当前项目自定义指标

当前 `dog.py` 里额外记录了这些指标：

```text
success_rate
distance_success_rate
straight_success_rate
base_contact
time_out
forward_distance
lateral_drift
alive_ratio
front_clearance
support_count
low_base_rate
no_progress_rate
```

判断标准：

- `forward_distance`：越大说明真实向前走得越远。
- `lateral_drift`：越小越直，太大说明斜走或绕圈。
- `time_out`：越接近 1，说明 episode 多数不是摔倒结束。
- `base_contact`：越低越好，说明机身/髋部少碰地。
- `low_base_rate`：越低越好，说明不是趴地拱。
- `support_count`：四足步态通常在 2 到 3 左右波动，长期接近 4 说明四脚贴地蹭。
- `straight_success_rate`：最适合判断“能不能直着走”。

当前 `step34` 的候选结果：

```text
iter 200: reward=176.49, forward=5.44, drift=1.13, success=0.88, straight=0.42, low=0.00
iter 298: reward=193.83, forward=5.01, drift=0.94, success=0.78, straight=0.70, low=0.00
iter 299: reward=192.28, forward=4.17, drift=1.69, success=0.54, straight=0.33, low=0.00
```

结论：最后一轮不一定最好。播放时优先看 `model_200.pt`、`model_298.pt` 附近，但因为只保存了每 25 轮和最终模型，实际可直接播放 `model_200.pt`、`model_300.pt`。

## 4. 检查训练结果：Play

### 4.1 基本播放命令

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --num_envs=1
```

### 4.2 加载指定模型

播放 `step34` 的 200 轮：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug14_17-25-01_step34_moderate_height_trot --checkpoint=200 --num_envs=1
```

播放 `step34` 的 300 轮：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug14_17-25-01_step34_moderate_height_trot --checkpoint=300 --num_envs=1
```

### 4.3 播放时重点观察

不要只看“有没有动”。要看：

- 是否向世界 X 正方向前进。
- 是否明显横漂或绕圈。
- 前脚是否只是蹭地。
- 后腿是否同时猛蹬。
- 机身是否长期很低。
- 是否能撑完整个 episode。
- 是否四脚同时小碎步，而不是形成支撑和摆动。

### 4.4 Viewer 报错处理

如果播放时报：

```text
GLFW initialization failed
GLFW window creation failed
Failed to create Window in CreateGymViewerInternal
```

先检查：

```bash
xdpyinfo >/dev/null && echo "X11 ok"
```

如果出现：

```text
Maximum number of clients reached
```

说明桌面 X11 连接数满了。当前机器曾经是 QQ 占用大量 X11 连接，可以先退出 QQ，或执行：

```bash
pkill -f /opt/QQ/qq
```

再测试：

```bash
xdpyinfo >/dev/null && echo "X11 ok"
```

如果还不行，注销当前桌面后重新登录最稳。

### 4.5 策略导出

`play.py` 通常会导出 Actor 策略到：

```text
/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/logs/flat_dog/exported/policies/
```

常见导出文件类似：

```text
policy_1.pt
```

这个导出的 JIT 策略文件就是后续 MuJoCo Sim2Sim 更适合加载的文件。

注意：如果你切换 checkpoint 播放，导出的 policy 可能被覆盖。做 Sim2Sim 前，建议把满意的策略复制一份并写清楚来源，例如：

```bash
mkdir -p /home/xiehan/下载/workspace_summer/policies
cp /home/xiehan/下载/workspace_summer/rl_stack/legged_gym/logs/flat_dog/exported/policies/policy_1.pt /home/xiehan/下载/workspace_summer/policies/dog_flat_step34_ckpt300_policy.pt
```

## 5. Sim2Sim 验证：MuJoCo

### 5.1 当前仓库状态

当前 `/home/xiehan/下载/workspace_summer/rl_stack/legged_gym` 没有自带：

```text
deploy/deploy_mujoco/
```

所以 MuJoCo 验证需要你新增部署脚本，或接入学长/同学已有的 `deploy_mujoco` 工程。常见结构如下：

```text
deploy/
  deploy_mujoco/
    deploy_mujoco.py
    configs/
      dog_flat.yaml
```

### 5.2 MuJoCo 部署命令

如果你已经有上述部署目录，常见命令是：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python deploy/deploy_mujoco/deploy_mujoco.py deploy/deploy_mujoco/configs/dog_flat.yaml
```

如果部署工程在工作区其他位置，则进入对应工程执行。

### 5.3 MuJoCo 配置文件需要对齐的内容

典型配置文件：

```text
deploy/deploy_mujoco/configs/dog_flat.yaml
```

需要重点检查：

```yaml
policy_path: /home/xiehan/下载/workspace_summer/policies/dog_flat_step34_ckpt300_policy.pt
xml_path: /home/xiehan/下载/workspace_summer/mujoco_models/dog/dog.xml

num_actions: 12
num_observations: 60

default_joint_angles:
  FL_hip_joint: 0.1
  FL_thigh_joint: 0.9
  FL_calf_joint: -1.55
  FR_hip_joint: -0.1
  FR_thigh_joint: 0.9
  FR_calf_joint: -1.55
  RL_hip_joint: 0.1
  RL_thigh_joint: 0.9
  RL_calf_joint: -1.55
  RR_hip_joint: -0.1
  RR_thigh_joint: 0.9
  RR_calf_joint: -1.55

control:
  control_type: P
  stiffness: 32.0
  damping: 2.4
  action_scale: 0.22
  decimation: 4
```

必须对齐的内容：

- `policy_path`：Isaac Gym 播放导出的 JIT 策略。
- `xml_path`：MuJoCo 使用的 XML 模型。
- 关节顺序：MuJoCo 的 qpos/qvel 顺序必须和策略动作顺序一致。
- `default_joint_angles`：必须和 `dog_config.py` 一致。
- `action_scale`：必须和 Isaac Gym 一致。
- `stiffness`、`damping`：必须和 Isaac Gym 一致或经过明确调参。
- `decimation` 和控制周期：必须和 Isaac Gym 尽量一致。
- 观测归一化和拼接顺序：必须和 `legged_gym` 的 `compute_observations()` 一致。
- 命令输入：训练时命令范围是 `lin_vel_x=[0.22, 0.28]`，MuJoCo 验证时不要突然给很大的速度命令。

### 5.4 当前 dog_flat 的观测提醒

当前 `DogFlatCfg.env.num_observations = 60`。

它不是原始基础 48 维，而是：

```text
基础观测 48 维 + gait clock 8 维 + desired contact states 4 维 = 60 维
```

所以 MuJoCo 部署时也必须构造同样的 60 维观测，否则 policy 输入维度或语义会错。

当前 `dog.py` 中有：

```python
self.obs_buf = torch.cat((self.obs_buf, self.gait_clock_inputs, self.desired_contact_states), dim=-1)
```

这意味着 MuJoCo 里也要按同样频率生成：

- 4 个足端相位的 `sin`
- 4 个足端相位的 `cos`
- 4 个期望接触状态

### 5.5 Sim2Sim 常见问题

如果 Isaac Gym 能走，MuJoCo 不能走，优先检查：

1. 关节顺序是否一致。
2. 关节正方向是否一致。
3. 默认关节角是否一致。
4. PD 控制频率是否一致。
5. `action_scale` 是否一致。
6. 观测顺序是否一致。
7. 重力方向、base 坐标系、四元数顺序是否一致。
8. 足端接触和摩擦参数是否过于不同。
9. MuJoCo XML 的质量、惯量、碰撞体是否和 URDF 接近。

## 6. 推荐日常工作流

### 6.1 每次调参前

记录本轮目标，例如：

```text
step35_straighter_walk: 在 step34 基础上继续压横漂和偏航，目标是减少绕圈。
```

修改：

```text
legged_gym/envs/dog/dog.py
legged_gym/envs/dog/dog_config.py
```

更新：

```python
run_name = 'step35_straighter_walk'
```

### 6.2 先短跑验证

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py --task=dog_flat --headless --max_iterations=3
```

确认无报错、reward 项注册成功。

### 6.3 再跑正式训练

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py --task=dog_flat --headless --max_iterations=300 2>&1 | tee /home/xiehan/下载/workspace_summer/rl_stack/legged_gym/logs/flat_dog_step35_console.log
```

### 6.4 不要盲目跑满

如果 100 到 150 轮时仍然出现：

```text
forward_distance < 0
success_rate = 0
base_contact = 1
low_base_rate 很高
support_count 长期接近 4
```

就不建议继续跑满，应停止并检查奖励、默认站姿、动作幅度、PD 参数和接触配置。

### 6.5 播放对比

优先对比多个 checkpoint：

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug14_17-25-01_step34_moderate_height_trot --checkpoint=200 --num_envs=1
```

```bash
cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug14_17-25-01_step34_moderate_height_trot --checkpoint=300 --num_envs=1
```

最终选视频效果最好的模型，不一定选 reward 最高的模型。

Aug16_23-54-23_step43_slow_low_stable_from900
Aug16_22-07-44_step14_straight_gate
Aug11_15-07-07_step1
Aug11_15-07-07_step1
Aug11_23-48-25_step2_easy_forward
Aug12_00-11-48_step4_low_spawn_visual
Aug12_00-21-54_step5_walk_reward
Aug12_00-25-48_step6_forward_vel
//

//跪姿:Aug12_00-47-38_step7_straight_walk 800

抬起来一直脚抬起来滑动:Aug12_10-29-35_step8_from450_gait 600/500转弯太严重
Aug12_10-48-00_step9_soft_gait

后腿一起动:Aug12_11-13-35_step10_from450_soft_gait   450

Aug12_13-02-33_step12_success_forward
Aug12_13-15-06_step13_straight_success

有一只脚桥起来但是走的还可以PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug12_13-15-06_step13_straight_success --checkpoint=850 --num_envs=1

Aug12_13-19-28_step14_straight_gate
 PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH /home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/play.py --task=dog_flat --load_run=Aug12_13-19-28_step14_straight_gate --checkpoint=900 --num_envs=1
走的卡ui还是只有三条腿和好与上一个

source /home/xiehan/anaconda3/etc/profile.d/conda.sh
conda activate leggedgym
export LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH

cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym

PLAY_COMMAND_X=0.22 PLAY_DISABLE_PLOT=1 python legged_gym/scripts/play.py \
  --task dog_flat \
  --num_envs 1 \
  --load_run Aug16_22-07-44_step14_straight_gate \
  --checkpoint 800走直线三条腿比较稳定48

  source /home/xiehan/anaconda3/etc/profile.d/conda.sh
conda activate leggedgym
export LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH

cd /home/xiehan/下载/workspace_summer/rl_stack/legged_gym

PLAY_COMMAND_X=0.22 PLAY_DISABLE_PLOT=1 python legged_gym/scripts/play.py \
  --task dog_flat \
  --num_envs 1 \
  --load_run Aug16_23-54-23_step43_slow_low_stable_from900 \
  --checkpoint 1300
这个好像会议点四只脚但是卡顿




             FL_hip_joint': 0.10,
            'FL_thigh_joint': 0.80,
            'FL_calf_joint': -1.50,
            'FR_hip_joint': -0.10,
            'FR_thigh_joint': 0.80,
            'FR_calf_joint': -1.50,
            'RL_hip_joint': 0.10,
            'RL_thigh_joint': 0.80,
            'RL_calf_joint': -1.50,
            'RR_hip_joint': -0.10,
            'RR_thigh_joint': 0.80,
            'RR_calf_joint': -1.50,