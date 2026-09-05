# RL Project Setup

## 当前状态
- 已提取 URDF 模型 `dog/urdf/dog.urdf`
- 已修正 `RL_foot` 的视觉网格引用错误
- 已创建项目目录 `rl_project`

## 下一步
1. 进入项目目录
   ```bash
   cd /home/xiehan/下载/workspace_summer/rl_project
   . .venv/bin/activate
   ```
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
3. 运行 URDF 检查脚本
   ```bash
   python inspect_urdf.py
   ```

## 建议的 URDF 调整方向
- 检查所有脚部 `visual` 与 `collision` 是否一致
- 如果要在模拟中稳定接触，可以将脚部 `collision` 换成简单几何体（如 `box` 或 `sphere`）
- 确保所有关节的 `limit`、`axis`、`origin` 设置合理
- 如果使用 Isaac Gym / PyBullet，先在仿真里加载模型并测试静态稳态

## 你现在可以做的事
- 如果你有 `URDF` 需要进一步修改，可以先把关键问题告诉我，我来给你一条条改
- 我也可以继续帮你写一个 `pybullet` 加载示例脚本，直接把这个模型放到仿真里看效果

