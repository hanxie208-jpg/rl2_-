# dog URDF 与 sim2sim 审计报告

审计时间：2026-09-02

本次只读检查并新增本报告；没有修改 URDF、训练配置、部署配置或代码，因此没有对原文件做备份。

## 1. 对比对象

- 原始 URDF：`/home/xiehan/下载/workspace_summer/dog最初版本.urdf`
- 当前训练 URDF：`/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/resources/robots/dog/urdf/dog.urdf`
- 当前训练配置：`rl_stack/legged_gym/legged_gym/envs/dog/dog_config.py`
- MuJoCo 部署配置：`rl_stack/legged_gym/deploy/deploy_mujoco/configs/dog_step50_60d.yaml`
- MuJoCo 日志：`rl_stack/legged_gym/logs/mujoco_step50_60d_test.log`

## 2. URDF 实际修改内容

### 2.1 质量和惯性没有被你改高

两份 URDF 的 `<mass>` 与 `<inertia>` 完全一致。

总质量：

- 原始 URDF：21.71528 kg
- 当前 URDF：21.71528 kg

质量分布：

- base：13.555 kg
- 每条腿：
  - hip：0.33292 kg
  - thigh：1.5467 kg
  - calf：0.1266 kg
  - foot：0.03385 kg
- 四条腿合计后总质量约 21.715 kg

结论：学长说“重量过高”值得检查，但从这两份文件的差异看，不能证明是你把重量改高了。模型从最初版本开始就是约 21.7 kg。是否过重，需要和真实机器狗实测质量、CAD 质量或电机可输出力矩一起判断。

### 2.2 8 个髋/大腿关节 effort 从 17 改成 28

原始值：`effort="17"`

当前值：`effort="28"`

涉及关节：

- `FL_hip_joint`
- `FL_thigh_joint`
- `FR_hip_joint`
- `FR_thigh_joint`
- `RR_hip_joint`
- `RR_thigh_joint`
- `RL_hip_joint`
- `RL_thigh_joint`

4 个 calf 关节仍是 `effort="25"`，没有变化。

影响判断：这是会影响训练的。legged_gym 会从 URDF 的 effort 读 torque limit，训练中的策略等于被允许使用更大的髋/大腿扭矩。导出到 MuJoCo 时如果也按 28 Nm 限幅，仿真内部是一致的；但如果真实电机或原始设计只有 17 Nm，这会导致策略学到实体或其他仿真器难以复现的动作。

### 2.3 足端碰撞体被统一成半径 0.02 m 的球

当前 URDF 中四个足端碰撞体均为：

```xml
<sphere radius="0.02"/>
```

更精确地说：

- 原始 `FL_foot`、`FR_foot`、`RR_foot` 碰撞体是对应 STL mesh，当前改成 sphere。
- 原始 `RL_foot` 碰撞体本来已经是 sphere，当前仍是 sphere。

STL 包围盒检查：

- `FL_foot.STL` / `FR_foot.STL` / `RR_foot.STL` 尺寸约 `0.042 x 0.036 x 0.036 m`，局部 z 范围约 `[-0.0166, 0.0195]`。
- `RL_foot.STL` 尺寸相同，但整体在 z 方向低约 `0.228 m`，局部 z 范围约 `[-0.2450, -0.2090]`。

影响判断：这是 URDF 里最高风险的物理修改之一。足端接触形状会直接影响：

- 接触时机和足端最低点
- 摩擦和滑移
- 接触力大小
- 支撑脚数量
- `feet_air_time`、接触力、脚部高度、前后腿支撑类奖励

不过，把脚掌碰撞简化成球本身不是一定错误。对 RL 训练来说，足端球碰撞反而常见，也能避免复杂 STL 碰撞不稳定。关键是 Isaac Gym 和 MuJoCo 必须用同一套足端碰撞、同一半径、相近 contact margin 和 friction。

### 2.4 RL_foot 视觉模型修正

原始 `RL_foot` visual 使用的是：

```xml
<mesh filename="../meshes/RR_foot.STL"/>
```

当前改成：

```xml
<origin xyz="0 0 0.22849" rpy="0 0 0"/>
<mesh filename="../meshes/RL_foot.STL"/>
```

结合 STL 包围盒看，`RL_foot.STL` 的顶点本身整体低约 0.228 m，所以当前 visual 加 `+0.22849` 是合理的视觉补偿。这个修改不改变质量、惯性、碰撞体，主要影响 viewer 里看到的脚是否在正确位置。

## 3. 学长提到的问题是否会影响后面训练

### 3.1 “重量过高”

会影响，但当前证据不是“你改高了”。

如果真实机器狗明显小于 21.7 kg，而 URDF 是 21.7 kg，那么训练会出现以下问题：

- 需要更大地面反力和电机扭矩才能站立、迈步。
- reward 可能鼓励策略用很大的动作和扭矩顶住身体。
- sim2sim 或上实物时更容易出现趴下、抖动、关节饱和。

但如果真实机器狗也接近 20 kg 级，并且电机扭矩确实能到当前限制，则质量不是首要问题。

### 3.2 “碰撞体是否修改过”

会影响，而且这次确实修改过。当前最应该让学长重点看的是足端碰撞球是否符合他们希望的训练模型。

建议结论：

- 如果目标是先稳定训练：保留四足端 sphere 是合理路线。
- 如果目标是严格复现实体脚掌形状：需要重新设计足端 collision，最好用简单 capsule/box/sphere 组合，而不是直接用复杂 STL。
- 不建议在 Isaac 用一种碰撞，在 MuJoCo 又用另一种碰撞。

## 4. 当前 sim2sim 差的更大原因

仅从工作区证据看，sim2sim 差不应只归因于 URDF 重量。更直接的问题有三个。

### 4.1 训练控制参数和 MuJoCo 部署控制参数不一致

step50 训练配置：

- stiffness = 32.0
- damping = 3.5
- action_scale = 0.14

MuJoCo `dog_step50_60d.yaml`：

- stiffness = 32.0
- damping = 5.0
- action_scale = 0.30

`action_scale` 从 0.14 放大到 0.30，约为训练时的 2.14 倍。这会让同一个 policy 输出在 MuJoCo 中变成更大的目标关节偏移。

MuJoCo 日志也支持这个判断：

- action_rms/max_abs：1.8237 / 12.3724
- torque_rms/max_abs：22.6974 / 28.0000 Nm

说明动作很大，且扭矩经常接近或打满 28 Nm 限幅。

### 4.2 MuJoCo/URDF 关节顺序与配置顺序存在警告

MuJoCo 日志明确输出：

- MuJoCo/URDF 顺序：FL, FR, RR, RL
- 配置顺序：FL, FR, RL, RR

部署代码当前用 joint name 地址读写，所以不一定已经把动作直接打错腿；但这是必须消除的风险。policy 的输入/输出顺序、Isaac `dof_names` 顺序、MuJoCo `joint_order` 顺序必须完全按同一语义锁死。否则会出现策略以为自己在控制 RL，实际影响 RR 的情况。

### 4.3 step50 在 Isaac 里成功，但 gait 迁移性不稳

训练日志显示 step50 后期 Isaac 指标可以到：

- success_rate：0.75 到 1.0
- forward_distance：约 5.2 到 5.55 m
- alive_ratio：约 0.96 到 1.00
- lateral_drift：约 0.46 到 0.87 m

但 MuJoCo step50 结果：

- 20 秒位移 x：0.5248 m
- 平均 vx：0.0262 m/s
- 最低高度：0.0635 m
- 最大倾角：约 89.85 deg
- 平均足端接触数：1.17
- likely_fall：True

CSV 里首次达到 `z < 0.22` 或 `tilt > 45 deg` 的时间约 1.12 s。

这说明 policy 在 Isaac 中学到的行为对 MuJoCo 接触/控制差异非常敏感。奖励里同时存在前进、直行、速度追踪、接触、支撑、前腿摆动、后腿抑制、姿态、高度等多目标，容易学出能刷 Isaac reward 但不够物理稳健的步态。

## 5. 是否因为 Isaac Gym 导出 policy 到 MuJoCo 天然特别差

不是“天然特别差”，但四足 locomotion 的 sim2sim 对以下东西非常敏感：

- URDF/MJCF 的 mass、inertia、joint axis、joint order
- foot collision 形状、radius、contact margin
- friction、solref/solimp/contact 参数
- PD stiffness/damping/action_scale/torque limit
- policy 观测顺序、动作顺序、默认关节角
- Isaac reset 时机和 MuJoCo policy engage 时机

你这里已经看到几个足够解释差异的因素，尤其是 action_scale/damping 不一致、关节顺序警告、足端碰撞简化后的接触模型差异。因此不能简单说“Isaac 导到 MuJoCo 就一定差”，更像是训练-部署物理接口还没有完全对齐。

## 6. 建议修复顺序

1. 先不要改质量，先确认真实机器狗总质量和各部件质量。如果没有实测，不建议凭感觉把 URDF 改轻。

2. 保留当前 URDF 前先统一声明：四足端 collision 使用 `sphere radius=0.02` 是训练假设。之后 Isaac 和 MuJoCo 都必须用这一假设。

3. 先把 MuJoCo step50 部署控制参数改回训练一致：

```yaml
stiffness: 32.0
damping: 3.5
action_scale: 0.14
```

4. 把 `dog_step50_60d.yaml` 的 `joint_order` 调整到与 MuJoCo/URDF 顺序一致，或在日志里完全消除 warning。随后再确认 policy 输入/输出仍然对应 Isaac 训练时的 `dof_names`。

5. 做三个短测试：

- stand only：应该 5 到 20 秒不倒，四脚接触稳定。
- zero action policy 接管前后：接管瞬间不应有大跳变。
- policy 5 秒测试：看首次摔倒时间、平均接触数、扭矩饱和比例。

6. 如果控制和顺序对齐后仍差，再回到奖励函数：优先减少同时启用的 shaping 项，先训一个低速、稳定、少摔、少饱和的基础步态，再逐渐加入直线/速度/步态相位约束。

## 7. 本次结论

- 你没有把 URDF 的重量改高；两份 URDF 总质量都约 21.715 kg。
- 你确实改了足端碰撞体：FL/FR/RR 从 STL 改为 0.02 m 球，RL 原本已经是球。
- 你把 8 个 hip/thigh 的 effort 从 17 改成 28，这会改变训练可用扭矩上限。
- 当前 sim2sim 差，更直接的证据指向控制参数不一致、关节顺序警告、接触模型敏感和奖励目标复杂，而不是单一的 URDF 重量问题。
