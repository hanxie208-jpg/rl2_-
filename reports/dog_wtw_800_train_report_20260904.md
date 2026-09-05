# dog_wtw_flat 800 轮训练报告

时间：2026-09-04

## 本次新增内容

新增独立实验文件夹：

- `legged_gym/envs/dog_wtw/`
- `legged_gym/envs/dog_wtw/dog_wtw_config.py`

注册新任务：

- `dog_wtw_flat`

注册前已备份：

- `rollback_backups/20260904_130538_before_register_dog_wtw/__init__.py`

该任务复用现有 `DogRobot`，读取当前最新版 URDF：

- `resources/robots/dog/urdf/dog.urdf`

当前 URDF 已是上一轮修改后的版本，12 个关节 `effort` 均为 `22`。

## 训练配置

- 环境数量：`256`
- 观测维度：`60`
- gait clock：开启
- 地形：plane
- 训练轮数：`800`
- 初始站姿：`base z = 0.35`
- PD：
  - stiffness = `32.0`
  - damping = `3.5`
  - action_scale = `0.14`

奖励参考 walk-these-ways-go2 的简化结构：

- 速度跟踪：`tracking_lin_vel`、`tracking_ang_vel`
- 姿态稳定：`lin_vel_z`、`ang_vel_xy`、`orientation`、`base_height`
- 代价项：`torques`、`dof_acc`、`action_rate`、`collision`、`dof_pos_limits`
- 足端/步态：`feet_air_time`、`gait_balance`、`tracking_contacts_force`、`tracking_contacts_vel`、`foot_clearance_clock`、`support_stability`、`foot_impact`

这版刻意没有使用原先大量 `forward_progress`、`straight_forward_vel`、`front_support`、`rear_push_without_front_ready` 等修漏洞项，目的是先观察“瘦奖励”能不能稳定站住和形成基础步态。

## 执行命令

```bash
LD_LIBRARY_PATH=/home/xiehan/anaconda3/envs/leggedgym/lib:$LD_LIBRARY_PATH \
PATH=/home/xiehan/anaconda3/envs/leggedgym/bin:$PATH \
/home/xiehan/anaconda3/envs/leggedgym/bin/python legged_gym/scripts/train.py \
  --task dog_wtw_flat \
  --headless \
  --num_envs 256 \
  --max_iterations 800
```

控制台日志：

- `logs/flat_dog_wtw_style_800_env256_effort22.log`

训练输出目录：

- `logs/flat_dog_wtw/Sep04_13-08-48_wtw_style_800_env256_effort22/`

最终 checkpoint：

- `logs/flat_dog_wtw/Sep04_13-08-48_wtw_style_800_env256_effort22/model_800.pt`

JIT policy 已导出：

- `logs/flat_dog_wtw/exported/policies/policy_1.pt`

## 关键训练结果

| 轮数 | Mean reward | Episode length | time_out | base_contact | forward_distance | lateral_drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.44 | 15.07 | 0.7708 | 0.2292 | -0.0320 | 0.0687 |
| 100 | 21.50 | 985.32 | 1.0000 | 0.0000 | -0.0266 | 0.0201 |
| 200 | 10.89 | 617.46 | 0.6250 | 0.3750 | 0.0165 | 0.0839 |
| 400 | 16.13 | 794.24 | 0.5417 | 0.4583 | 0.0137 | 0.0246 |
| 600 | 22.43 | 1001.80 | 1.0000 | 0.0000 | -0.0302 | 0.0061 |
| 799 | 20.45 | 960.55 | 0.9583 | 0.0417 | 0.0042 | 0.0108 |

## 结论

这版训练没有崩，环境和最新版 URDF 可以正常训练。策略基本学会了稳定存活：

- 后期 episode length 接近满长度
- `time_out` 后期经常接近 `1.0`
- `base_contact` 后期大幅下降
- `lateral_drift` 很小

但它还没有真正学会向前走：

- `success_rate` 始终为 `0`
- `forward_distance` 后期仍接近 `0`
- `tracking_lin_vel` 分数较高，但没有变成真实有效位移

我的判断：WTW 风格瘦奖励对“站稳”有效，但对你这台偏重 base 的自定义狗来说，单靠局部速度 tracking 不足以打破原地稳定的局部最优。

## 下一步建议

下一版不要回到大量奖励堆叠，建议只补两个前进信号：

1. 加一个小权重 `step_forward_progress`
   - 建议 scale：`0.6` 到 `1.0`
   - 作用：奖励每个控制步的世界 X 正位移

2. 加一个小权重 `commanded_forward_motion`
   - 建议 scale：`0.4` 到 `0.8`
   - 作用：让速度从 0 附近有连续梯度，不只是靠 tracking 硬拉

同时建议把 success 判定中的 `forward_distance > 3.0` 暂时降到 `0.8-1.2m` 作为早期调试指标，否则 800 轮内一直显示 0，不利于判断中间进步。

质量方面仍建议另开 URDF 质量实验分支，不要在这个 WTW 基线上继续乱调奖励掩盖物理问题。
