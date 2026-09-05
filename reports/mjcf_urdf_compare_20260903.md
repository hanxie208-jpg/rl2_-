# dog.urdf 与 dog_step14_900_mujoco.xml 对比报告

审计时间：2026-09-03

本次只读取并比较文件，只新增本报告；没有修改 URDF、MJCF、训练配置或部署配置。

## 1. 对比对象

- 训练 URDF：`/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/resources/robots/dog/urdf/dog.urdf`
- MuJoCo generated XML：`/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/deploy/deploy_mujoco/generated/dog_step14_900_mujoco.xml`

注意：`dog_step14_900_mujoco.xml` 是 `deploy_mujoco.py` 在 checker floor 模式下固定写出的文件名，后续运行其他配置也可能覆盖它。因此它更像“最近一次生成的 MuJoCo 模型快照”，不一定严格等于 step14 原始模型。

## 2. 质量对比

总质量没有变：

- URDF 总质量：21.71528 kg
- MJCF body 总质量：21.71528 kg

所以当前不是 MuJoCo generated XML 把整机额外加重了。

## 3. calf 看起来变重的原因

URDF 中每条小腿和脚是两个 link：

- `*_calf` mass = 0.12660 kg
- `*_foot` mass = 0.03385 kg
- 合计 = 0.16045 kg

MuJoCo XML 中没有单独的 `*_foot` body，fixed foot 被合并进对应的 `*_calf` body，所以：

- `FL_calf` mass = 0.16045 kg
- `FR_calf` mass = 0.16045 kg
- `RR_calf` mass = 0.16045 kg
- `RL_calf` mass = 0.16045 kg

这正好等于：

```text
0.12660 + 0.03385 = 0.16045
```

结论：你看到的 calf 变重，不是反向加重，而是 MuJoCo 导入 URDF 时把 fixed joint 连接的 foot 质量合并到了 calf 上。

## 4. 合并后惯性和质心发生了变化

URDF 的 calf COM：

```text
(0.012446, 约 0, -0.11689)
```

MJCF 合并 calf+foot 后 COM：

```text
约 (0.00982, 约 0, -0.140434)
```

也就是说，小腿合并脚之后，质心沿 z 方向更靠近脚端。这是物理上合理的，因为 foot 在 calf 下方。

但这会带来一个 sim2sim 差异风险：

- Isaac Gym 配置里 `collapse_fixed_joints = False`，训练时可能保留 foot rigid body。
- MuJoCo generated XML 里 foot body 被合并进 calf，只保留 foot sphere geom。

总质量一致，但 body 划分、刚体质心和惯性表达方式不同，接触奖励和日志里记录的 foot body 高度也可能不完全对应。

## 5. 碰撞体对比

URDF：

- base/hip/thigh/calf 使用 mesh collision
- 四个 foot 使用 `sphere radius="0.02"`

MJCF：

- base/hip/thigh/calf 仍使用 mesh geom
- 四个 foot collision sphere 被挂到对应 calf body 下面：
  - `FL_foot_collision_0` pos = `0 0 -0.22849`
  - `FR_foot_collision_0` pos = `0 0 -0.22849`
  - `RR_foot_collision_0` pos = `0 0 -0.228486`
  - `RL_foot_collision_0` pos = `0 0 -0.22849`

结论：足端球碰撞的半径和相对位置基本对齐，没有发现 foot sphere 在 MJCF 中丢失。

## 6. 关节力矩限制对比

URDF 与 MJCF 当前一致：

- hip/thigh：`effort=28`，MJCF 为 `actuatorfrcrange="-28 28"`
- calf：`effort=25`，MJCF 为 `actuatorfrcrange="-25 25"`

这里没有发现 MuJoCo 把 calf 力矩限制改大。真正与“减轻体重”方向相反的地方，仍然是整机 URDF 自身总质量约 21.7 kg，以及 hip/thigh effort 从原始 17 改到 28。

## 7. 其它 MuJoCo 参数变化

MJCF 额外加入了：

- free joint：`floating_base`
- floor plane
- checker texture/material
- floor friction：`1 0.005 0.0001`

MJCF 文件本身没有写入：

- damping
- armature
- solref/solimp
- contact margin

这些通常是在 `deploy_mujoco.py` 运行时从 yaml 配置写进 `model`，不一定体现在 generated XML 文件中。

## 8. 结论

1. 你发现 calf 对不上是对的：URDF calf 是 0.1266 kg，MJCF calf 是 0.16045 kg。
2. 但这不是 MuJoCo 反向加重，而是 foot fixed joint 被合并后，`calf + foot` 变成一个 body。
3. 整机总质量没有变，仍是 21.71528 kg。
4. 如果学长说“重量给重了”，应优先怀疑当前 URDF 的原始 mass/inertia 本身过大，而不是 MJCF 转换过程额外加重。
5. 真正需要减重时，应该按比例修改 URDF 的 mass 和 inertia，再重新生成 MuJoCo XML；不要只手改 generated XML，否则训练和部署会更不一致。
