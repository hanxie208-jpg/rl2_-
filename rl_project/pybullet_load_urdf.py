import argparse
import time
from pathlib import Path

import pybullet as p
import pybullet_data


# 这个脚本的目标：
# 1. 让你看到怎样把 URDF 模型真正加载到物理仿真里
# 2. 让你理解 PyBullet 的基本流程：连接 -> 加地面 -> 加机器人 -> 进行仿真
# 3. 为后面的 RL 训练打基础：环境、状态、动作，都是从这里开始


def build_parser():
    parser = argparse.ArgumentParser(description='Load the dog URDF in PyBullet')
    parser.add_argument('--mode', choices=['gui', 'direct'], default='direct',
                        help='Use GUI for visualization or DIRECT for headless runs')
    return parser


def main():
    args = build_parser().parse_args()

    # 连接物理引擎。
    # GUI 模式适合你手动看机器人；DIRECT 模式适合在服务器或无界面环境里运行。
    if args.mode == 'gui':
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)

    # 让 PyBullet 能找到它自带的示例资源（比如 plane.urdf）
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 设置重力，模拟真实世界。
    p.setGravity(0, 0, -9.81)

    # 载入地面，避免机器人直接掉到虚空里。
    plane_id = p.loadURDF('plane.urdf')

    # 载入你的狗模型。
    # useFixedBase=False 表示机器人不是固定在地面上，而是可自由运动。
    # URDF_USE_INERTIA_FROM_FILE 会使用 URDF 中的惯性信息，提高模型真实感。
    robot_dir = Path(__file__).resolve().parent / 'dog' / 'urdf'
    urdf_path = robot_dir / 'dog.urdf'
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)

    print(f'Loaded robot from: {urdf_path}')
    print('Physics client ID:', physics_client)

    # 让仿真运行一段时间，观察机器人是否稳定地落到地面上。
    for step in range(300):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 这里不做训练，只是让你看“环境已经起来了”。
    print('Simulation finished. If you used GUI mode, you should have seen the robot in the scene.')

    p.disconnect()


if __name__ == '__main__':
    main()
