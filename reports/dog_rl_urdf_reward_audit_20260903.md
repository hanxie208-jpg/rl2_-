# 机器狗训练与 sim2sim 审计报告

审计时间：2026-09-03

## 本次做了什么

1. 将训练 URDF `resources/robots/dog/urdf/dog.urdf` 的 12 个执行关节 `effort` 统一改为 `22`。
2. 先备份原文件到：
   - `rollback_backups/20260903_173341_before_urdf_effort22/dog.urdf`
3. 将 MuJoCo 部署配置 `deploy/deploy_mujoco/configs/dog_step50_60d.yaml` 的三类力矩上限也统一改为 `22.0`，并先备份到：
   - `rollback_backups/20260903_175124_before_deploy_yaml_torque22/dog_step50_60d.yaml`

## 当前已确认的结果

- `dog.urdf` 当前 12 个关节 effort 全部是 `22`
- `dog_step50_60d.yaml` 当前 torque_limits 为：
  - hip = 22.0
  - thigh = 22.0
  - calf = 22.0
- 训练 URDF 总质量仍是 `21.71528 kg`
- 这次没有改质量、惯量、关节角度上下限，也没有改碰撞几何

## 质量分布判断

当前质量分布不是“腿总重离谱”，而是“base 偏重、远端过轻”：

- base = `13.555 kg`，约占总质量 `62.4%`
- 每条腿 = `2.04007 kg`
- 每条腿内部：
  - hip = `0.33292 kg`
  - thigh = `1.5467 kg`
  - calf = `0.1266 kg`
  - foot = `0.03385 kg`

问题点：

- thigh 占单腿质量约 `75.8%`
- calf + foot 很轻，导致腿端惯性偏小
- base 占比明显高于 Go2 一类参考机型，策略更容易学成“拖着重机身挪腿”

## MuJoCo 里“calf 变重”的原因

不是反向加重，而是 fixed foot 被合并到了 calf body：

- URDF 中 `calf + foot = 0.16045 kg`
- MuJoCo generated XML 里 calf body 质量也会变成 `0.16045 kg`

所以你看到的 calf 质量对不上，主要是 body 合并方式变化，不是质量被凭空加大。

## walk-these-ways-go2 可借鉴的奖励思路

我读到的核心 reward 结构比较清晰，主要是这几类：

- 速度跟踪：`tracking_lin_vel`、`tracking_ang_vel`
- 身体稳定：`lin_vel_z`、`ang_vel_xy`、`orientation`、`base_height`
- 代价项：`torques`、`dof_acc`、`action_rate`、`collision`、`dof_pos_limits`
- 足端相关：`feet_air_time`、`feet_slip`、`feet_contact_forces`、`feet_clearance_cmd_linear`、`feet_impact_vel`
- 步态相关：`tracking_contacts_shaped_force`、`tracking_contacts_shaped_vel`、`raibert_heuristic`

它的特点不是“奖励越多越好”，而是：

- 主目标少而明确
- 接触/步态项是结构化的
- 课程学习和 gait 参数 curriculum 很明显

## 对你当前任务的建议

盲狗也可以学 walk-these-ways 的结构，但不要照搬全部项。更适合分三层：

1. 第一层只保留
   - `tracking_lin_vel`
   - `tracking_ang_vel`
   - `orientation` / `body_upright`
   - `lin_vel_z`
   - `ang_vel_xy`
   - `torques`
   - `action_rate`
   - `collision`

2. 第二层再加
   - `feet_air_time`
   - `feet_slip`
   - `foot_clearance`
   - 轻量接触节律项

3. 第三层才考虑
   - gait clock
   - Raibert 风格足落点
   - 轨迹/接触形状奖励

## 我认为最可能影响 sim2sim 的点

1. 质量分布和惯量表达不够像真实机器人，尤其是 base 偏重、脚端过轻。
2. 训练侧和部署侧力矩上限曾不一致，这会让策略在 Isaac Gym 里学到的动作，在 MuJoCo 里更容易被截断。
3. 奖励项偏多，且多个项在重复推动“前进、抬脚、别横漂”，容易把策略推到局部最优。

## 下一步建议

- 先用当前 `effort=22` 和部署侧 `torque_limits=22` 跑一轮，对比是否改善“关节一打开就过猛 / 一站就炸”的问题
- 如果还是站不稳，优先改质量，不要先乱调奖励
- 推荐试两个质量版本：
  - 版本 A：全身质量按比例缩到约 `15-17 kg`
  - 版本 B：base 再降一些，把一部分质量转给腿部，尤其是 calf/foot 的惯性不要太薄

## 备注

当前 `deploy/deploy_mujoco/generated/dog_step14_900_mujoco.xml` 还是旧快照，必须重新生成后才能真正反映最新的 `22` 力矩上限。
