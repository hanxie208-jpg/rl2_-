import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


# 这个脚本的目标：
# 1. 让你看到“RL 环境”其实就是一个不断更新状态、接收动作、返回奖励的循环
# 2. 先不用真正训练神经网络，而是先用一个简单的控制策略，让机器人动起来
# 3. 通过 reward 的定义，理解为什么 RL 训练时要设计好奖励函数


def build_parser():
    parser = argparse.ArgumentParser(description='A simple PyBullet environment with reward tracking')
    parser.add_argument('--mode', choices=['gui', 'direct'], default='direct')
    return parser


def connect(mode):
    if mode == 'gui':
        return p.connect(p.GUI)
    return p.connect(p.DIRECT)


def load_robot():
    # 设置资源路径，PyBullet 才能找到 plane.urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 先加载地面
    p.loadURDF('plane.urdf')

    # 再加载狗模型
    project_dir = Path(__file__).resolve().parent
    urdf_path = project_dir / 'dog' / 'urdf' / 'dog.urdf'
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # 识别所有可驱动的关节（revolute joints）
    # 这里的思路是：RL 的 action 通常就是对这些关节做控制
    joint_indices = []
    for joint_index in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            joint_indices.append(joint_index)
            print(f'Joint {joint_index}: {joint_name}')

    return robot_id, joint_indices, urdf_path


def compute_reward(robot_id, joint_indices):
    # 这里的 reward 直接告诉机器人：
    # 1. 想要向前走，就给正向速度奖励
    # 2. 想要站得稳，就给高度和姿态奖励
    # 3. 想要动作不要太激烈，就给关节惩罚
    base_pos, base_ori = p.getBasePositionAndOrientation(robot_id)
    base_lin_vel, _ = p.getBaseVelocity(robot_id)
    forward_speed = base_lin_vel[0]
    height = base_pos[2]
    roll, pitch, _ = p.getEulerFromQuaternion(base_ori)

    energy_penalty = 0.0
    for joint_idx in joint_indices:
        joint_state = p.getJointState(robot_id, joint_idx)
        joint_angle = joint_state[0]
        energy_penalty += abs(joint_angle)

    # reward 的含义是：
    # - forward_speed 越大越好
    # - height 太低说明机器人可能摔倒，应该被惩罚
    # - roll/pitch 过大说明姿态不稳，也应该惩罚
    # - energy_penalty 过大说明动作太激烈，也应该惩罚
    reward = 1.2 * max(forward_speed, 0.0) + 0.6 * height - 0.6 * abs(roll) - 0.6 * abs(pitch) - 0.01 * energy_penalty

    if height < 0.2:
        reward -= 2.0
    if forward_speed < 0.0:
        reward -= 0.4 * abs(forward_speed)

    return reward, forward_speed, height, roll, pitch


def apply_action(robot_id, joint_indices, step_count):
    # 这里先用一个“稳定姿态 + 小幅节奏变化”的动作策略
    # 它不是完整的 RL 策略，但能让你直观看到：
    # - action 由什么决定
    # - action 进入仿真后，状态会如何变化
    # - reward 如何被影响
    phase = step_count / 40.0
    targets = []

    for index, joint_idx in enumerate(joint_indices):
        # 用不同的相位，让关节产生轻微的节奏变化，而不是完全固定
        if index % 3 == 0:
            target = 0.10 * math.sin(phase + index * 0.6)
        elif index % 3 == 1:
            target = 0.25 + 0.08 * math.sin(phase + index * 0.4)
        else:
            target = -0.40 + 0.06 * math.sin(phase + index * 0.3)
        targets.append(target)

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=targets,
        positionGains=[1.0] * len(joint_indices),
        velocityGains=[0.1] * len(joint_indices),
    )


def main():
    args = build_parser().parse_args()
    connect(args.mode)

    robot_id, joint_indices, urdf_path = load_robot()
    print(f'Loaded robot from: {urdf_path}')

    # 这是一个非常简化的 RL 环境循环：
    # 1. 根据当前状态，给出一个 action
    # 2. 仿真一步
    # 3. 计算 reward
    # 4. 记录结果
    total_reward = 0.0
    for step in range(400):
        apply_action(robot_id, joint_indices, step)
        p.stepSimulation()

        reward, forward_speed, height, roll, pitch = compute_reward(robot_id, joint_indices)
        total_reward += reward

        if step % 50 == 0:
            print(f'step={step:03d} reward={reward:7.3f} speed={forward_speed:6.3f} height={height:6.3f} roll={roll:6.3f} pitch={pitch:6.3f}')

    print(f'Final total reward: {total_reward:.3f}')
    p.disconnect()


if __name__ == '__main__':
    main()
