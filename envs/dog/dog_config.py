# from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


# class DogFlatCfg(LeggedRobotCfg):
#     class env(LeggedRobotCfg.env):
#         # step14 checkpoint 是 48 维观测；不能拼 gait clock，否则旧模型加载会维度不匹配。
#         num_observations = 48
#         use_gait_observations = False
#         # 8GB 显存下 384 只仍然比较稳，采样比 256 更充分；播放时用 --num_envs=1。
#         num_envs = 384

#     class terrain(LeggedRobotCfg.terrain):
#         # 第一次训练先用平地，先把“能站、能走”跑通
#         mesh_type = 'plane'
#         measure_heights = False
#         curriculum = False

#     class commands(LeggedRobotCfg.commands):
#         # 第一次先做平地速度跟踪，不启用 heading curriculum
#         heading_command = False
#         resampling_time = 4.0
#         class ranges(LeggedRobotCfg.commands.ranges):
#             # step44：继续压低速度，避免为了追速度奖励前冲、扭屁股后摔倒。
#             # 注意 legged_gym 会把 <=0.2 的小速度命令置零，所以最低值保持略高于 0.2。
#             lin_vel_x = [0.28, 0.40]#0.21 0.27
#             lin_vel_y = [0.0, 0.0]
#             ang_vel_yaw = [0.0, 0.0]

#     class init_state(LeggedRobotCfg.init_state):
#         # 按当前默认关节角计算，足端球心在 base 下方约 0.31~0.32m；
#         # base z 设为 0.34m，可让脚接近地面，避免出生后先自由落体再摔倒。
#         pos = [0.0, 0.0, 0.34]#0.32
#         rot = [0.0, 0.0, 0.0, 1.0]
#         lin_vel = [0.0, 0.0, 0.0]
#         ang_vel = [0.0, 0.0, 0.0]
#         # 让四条腿更对称一点，避免前后腿初始受力差太大，策略只学会动后腿。
#         default_joint_angles = {
#             'FL_hip_joint': 0.1,
#             'FL_thigh_joint': 0.95,
#             'FL_calf_joint': -1.65,
#             'FR_hip_joint': -0.1,
#             'FR_thigh_joint': 0.95,
#             'FR_calf_joint': -1.65,
#             'RL_hip_joint': 0.1,
#             'RL_thigh_joint': 0.95,
#             'RL_calf_joint': -1.65,
#             'RR_hip_joint': -0.1,
#             'RR_thigh_joint': 0.95,
#             'RR_calf_joint': -1.65,
#         }

#     class control(LeggedRobotCfg.control):
#         # 给小幅摆腿留出控制余量；阻尼同步提高，避免动作幅度增大后变成后腿猛蹬。
#         control_type = 'P'
#         stiffness = {'joint': 32.0}
#         damping = {'joint': 3.5}
#         action_scale = 0.14
#         decimation = 4

#     class asset(LeggedRobotCfg.asset):
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
#         name = 'dog'
#         foot_name = 'foot'
#         penalize_contacts_on = ['hip', 'thigh', 'calf']
#         terminate_after_contacts_on = ['base', 'hip']
#         # 真实 dog URDF 的 foot 是 fixed joint 连接；不能折叠，否则 *_foot body 会消失，
#         # feet_air_time/contact 判断会找不到脚，足端零件在 viewer 里也容易看起来缺失。
#         collapse_fixed_joints = False
#         # STL 已按 URDF 坐标导出，关闭视觉翻转，避免 viewer 中腿部显示方向异常。
#         flip_visual_attachments = False
#         self_collisions = 1

#     class domain_rand(LeggedRobotCfg.domain_rand):
#         # 第一次训练先少做扰动，便于看清参数是否正确
#         randomize_friction = False
#         push_robots = False

#     class rewards(LeggedRobotCfg.rewards):
#         # 机器人先要稳定站住，再学会前进
#         # 保留正奖励裁剪，避免 PPO 学到“早点摔倒就少吃负奖励”。
#         # 真正需要疼的失败情况交给 termination，一次性扣足。
#         only_positive_rewards = True
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.31
#         max_contact_force = 350.0
#         # 速度命令降低后，sigma 稍收紧，让策略更在意速度误差。
#         tracking_sigma = 0.25

#         class scales(LeggedRobotCfg.rewards.scales):
#             # legged_gym 会把 scale 乘以 dt=0.02；这里 -300 才等价于每次失败约 -6。
#             termination = -500.0
#             tracking_lin_vel = 0.8
#             tracking_ang_vel = 0.4
#             # step44：速度只作为辅助，不能让策略靠前冲拿分。
#             forward_vel = 0.15
#             world_forward_vel = 0.2
#             forward_progress = 0.3
#             step_forward_progress = 1.2
#             straight_forward_progress = 0.8
#             backward_progress = -2.0
#             low_speed = -1.0
#             straight_forward_vel = 1.4
#             world_velocity_tracking = 1.5
#             stable_command_speed = 0.8
#             commanded_forward_motion = 0.8
#             forward_speed_floor = -2.0
#             no_forward_motion = -2.0
#             backward_vel = -4.0
#             world_backward_motion = -3.0
#             stand_motion = -2.0
#             lateral_vel = -4.0
#             world_lateral_vel = -5.0
#             lateral_drift = -6.0
#             yaw_rate = -5.0
#             yaw_drift = -5.0
#             orientation = -4.0
#             lin_vel_z = -3.0
#             ang_vel_xy = -0.2
#             base_height = -3.0
#             low_base_height = -4.0
#             body_upright = 0.4
#             torques = -0.00012
#             dof_acc = -1.0e-6
#             feet_air_time = 0.08
#             collision = -1.0
#             stumble = -0.6
#             gait_balance = 0.03
#             diagonal_clock_gait = 0.0
#             trot_joint_pose = 0.0
#             front_swing_pose = 0.0
#             tracking_contacts_force = 0.0
#             tracking_contacts_vel = 0.0
#             foot_clearance_clock = 0.0
#             vertical_motion = 0.0
#             airborne = 0.0
#             support_stability = 0.0
#             all_feet_contact = 0.0
#             foot_impact = 0.0
#             rear_push_jump = 0.0
#             front_support = 0.0
#             front_leg_activity = 0.0
#             foot_clearance = 0.0
#             front_foot_clearance = 0.0
#             rear_overstride = 0.0
#             rear_pair_swing = 0.0
#             front_slip = 0.0
#             action_rate = -0.02
#             dof_pos_limits = -5.0

#     class normalization(LeggedRobotCfg.normalization):
#         # 观测尺度先不动，沿用基础四足设置
#         pass

#     class noise(LeggedRobotCfg.noise):
#         # 第一次训练先关噪声，减少不稳定因素
#         add_noise = False

#     class viewer(LeggedRobotCfg.viewer):#视角调试直接改播放相机
#        pos = [2.0, -3.0, 1.5]
#        lookat = [0.0, 0.0, 0.35]

# class DogFlatCfgPPO(LeggedRobotCfgPPO):
#     class policy(LeggedRobotCfgPPO.policy):
#         # 小一点的网络更适合第一版验证，也更省显存
#         init_noise_std = 0.7
#         actor_hidden_dims = [128, 64, 32]
#         critic_hidden_dims = [128, 64, 32]
#         activation = 'elu'

#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         # 降低探索奖励，让 Mean action noise std 更快从 1 往 0.x 收，减少后腿乱蹬。
#         entropy_coef = 0.0002
#         # step43 后期出现横漂奖励振荡，继续训练时降低 PPO 更新幅度。
#         learning_rate = 5.e-4

#     class runner(LeggedRobotCfgPPO.runner):
#         # step44：从 step43 的 1400 继续，目标是压住前冲、横漂和摔倒。
#         run_name = 'step44_lateral_guard_from1300'
#         experiment_name = 'flat_dog'
#         load_run = -1
#         max_iterations = 1800
#         save_interval = 50


# class DogFlatClockCfg(DogFlatCfg):
#     class env(DogFlatCfg.env):
#         # step21_clock_forward was trained with 48 base observations plus
#         # 8 gait-clock and 4 desired-contact observations.
#         num_observations = 60
#         use_gait_observations = True

#     class commands(DogFlatCfg.commands):
#         class ranges(DogFlatCfg.commands.ranges):
#             lin_vel_x = [0.20, 0.25]
#             lin_vel_y = [0.0, 0.0]
#             ang_vel_yaw = [0.0, 0.0]

#     class init_state(DogFlatCfg.init_state):
#         pos = [0.0, 0.0, 0.35]
#         default_joint_angles = {
#             'FL_hip_joint': 0.1,
#             'FL_thigh_joint': 0.9,
#             'FL_calf_joint': -1.55,
#             'FR_hip_joint': -0.1,
#             'FR_thigh_joint': 0.9,
#             'FR_calf_joint': -1.55,
#             'RL_hip_joint': 0.1,
#             'RL_thigh_joint': 0.9,
#             'RL_calf_joint': -1.55,
#             'RR_hip_joint': -0.1,
#             'RR_thigh_joint': 0.9,
#             'RR_calf_joint': -1.55,
#         }

#     class control(DogFlatCfg.control):
#         stiffness = {'joint': 32.0}
#         damping = {'joint': 2.8}
#         action_scale = 0.18


# class DogFlatClockCfgPPO(DogFlatCfgPPO):
#     class runner(DogFlatCfgPPO.runner):
#         run_name = 'step21_clock_forward'
#         experiment_name = 'flat_dog'


# class DogFlatSim2SimCfg(DogFlatCfg):
#     """Clock-conditioned deterministic gait task used before domain randomization."""

#     class env(DogFlatCfg.env):
#         num_observations = 60
#         use_gait_observations = True
#         num_envs = 384

#     class normalization(DogFlatCfg.normalization):
#         # Stage 1 retains the source policy's action range. A later low-noise
#         # robustness stage will narrow the learned distribution if needed.
#         clip_actions = 100.0

#     class rewards(DogFlatCfg.rewards):
#         only_positive_rewards = False
#         tracking_sigma = 0.04
#         world_tracking_sigma = 0.04
#         stable_speed_sigma = 0.025
#         command_motion_slope = 25.0
#         command_motion_threshold = 0.08
#         phase_motion_min_speed = 0.02
#         phase_motion_full_speed = 0.16

#         class scales(DogFlatCfg.rewards.scales):
#             # Make the deterministic actor use the phase observation instead of
#             # depending on sampled PPO action noise to start each step.
#             tracking_lin_vel = 0.6
#             tracking_ang_vel = 0.2
#             forward_vel = 0.8
#             world_forward_vel = 0.8
#             forward_progress = 0.5
#             step_forward_progress = 3.0
#             straight_forward_progress = 1.2
#             straight_forward_vel = 2.0
#             world_velocity_tracking = 1.0
#             stable_command_speed = 1.0
#             commanded_forward_motion = 1.0
#             forward_speed_floor = -4.0
#             no_forward_motion = -4.0
#             low_speed = -2.0
#             stand_motion = -3.0
#             gait_balance = 0.03
#             diagonal_clock_gait = 0.35
#             trot_joint_pose = 0.15
#             front_swing_pose = 0.08
#             tracking_contacts_force = -0.12
#             tracking_contacts_vel = -0.08
#             foot_clearance_clock = 0.12
#             support_stability = 0.10
#             vertical_motion = -1.0
#             airborne = -0.8
#             foot_impact = -0.15
#             rear_push_jump = -0.6
#             front_support = 0.12
#             front_leg_activity = 0.08
#             front_slip = -0.20
#             action_rate = -0.04
#             lateral_vel = -8.0
#             world_lateral_vel = -10.0
#             lateral_drift = -10.0
#             yaw_rate = -7.0
#             yaw_drift = -8.0

#     class noise(DogFlatCfg.noise):
#         # A small amount of observation noise avoids a policy that only works
#         # on exact Isaac Gym state values while keeping phase learning stable.
#         add_noise = True
#         noise_level = 0.15


# class DogFlatSim2SimCfgPPO(DogFlatCfgPPO):
#     class policy(DogFlatCfgPPO.policy):
#         init_noise_std = 0.06

#     class algorithm(DogFlatCfgPPO.algorithm):
#         entropy_coef = 0.0
#         learning_rate = 2.0e-5
#         schedule = 'fixed'

#     class runner(DogFlatCfgPPO.runner):
#         run_name = 'step50_lateral_distill_from_step49_250'
#         experiment_name = 'flat_dog'
#         max_iterations = 500
#         save_interval = 50


# class DogFlatSim2SimFrontFirstCfg(DogFlatSim2SimCfg):
#     """Fine-tune the nominal 60D gait with front-foot-first shaping."""

#     class commands(DogFlatSim2SimCfg.commands):
#         class ranges(DogFlatSim2SimCfg.commands.ranges):
#             lin_vel_x = [0.24, 0.34]
#             lin_vel_y = [0.0, 0.0]
#             ang_vel_yaw = [0.0, 0.0]

#     class rewards(DogFlatSim2SimCfg.rewards):
#         class scales(DogFlatSim2SimCfg.rewards.scales):
#             step_forward_progress = 2.0
#             forward_vel = 0.6
#             world_forward_vel = 0.6
#             straight_forward_vel = 1.4
#             front_swing_pose = 0.20
#             front_swing_motion = 0.50
#             front_support = 0.45
#             front_leg_activity = 0.05
#             rear_push_without_front_ready = -2.80
#             base_ahead_of_support = -3.0
#             rear_push_jump = -0.80
#             front_slip = -0.30
#             foot_impact = -0.20


# class DogFlatSim2SimFrontFirstCfgPPO(DogFlatSim2SimCfgPPO):
#     class algorithm(DogFlatSim2SimCfgPPO.algorithm):
#         learning_rate = 2.0e-5
#         entropy_coef = 0.0

#     class policy(DogFlatSim2SimCfgPPO.policy):
#         init_noise_std = 0.06

#     class runner(DogFlatSim2SimCfgPPO.runner):
#         run_name = 'step53_front_first_from_step50_50'
#         max_iterations = 1200
#         save_interval = 100


# class DogFlatSim2SimDeployMatchCfg(DogFlatSim2SimFrontFirstCfg):
#     """Train under the PD gains and action range validated in MuJoCo."""

#     class control(DogFlatSim2SimFrontFirstCfg.control):
#         # Start with the step50 controller and move to the MuJoCo-validated
#         # controller over several thousand control steps.
#         stiffness = {'joint': 32.0}
#         damping = {'joint': 3.5}
#         action_scale = 0.14
#         deploy_match_transition_steps = 5000
#         deploy_match_source_stiffness = 32.0
#         deploy_match_source_damping = 3.5
#         deploy_match_source_action_scale = 0.14
#         deploy_match_target_stiffness = 50.0
#         deploy_match_target_damping = 5.0
#         deploy_match_target_action_scale = 0.10


# class DogFlatSim2SimDeployMatchCfgPPO(DogFlatSim2SimFrontFirstCfgPPO):
#     class algorithm(DogFlatSim2SimFrontFirstCfgPPO.algorithm):
#         learning_rate = 2.0e-5

#     class runner(DogFlatSim2SimFrontFirstCfgPPO.runner):
#         run_name = 'step54_deploy_pd_curriculum_from_step50_50'
#         max_iterations = 250
#         save_interval = 25


# class DogFlatSim2SimRobustCfg(DogFlatSim2SimCfg):
#     """Domain-randomized fine tuning for Isaac Gym to MuJoCo transfer."""

#     class domain_rand(DogFlatSim2SimCfg.domain_rand):
#         randomize_friction = True
#         friction_range = [0.65, 1.35]
#         randomize_base_mass = True
#         added_mass_range = [-0.6, 0.6]
#         randomize_motor_strength = True
#         motor_strength_range = [0.82, 1.18]
#         push_robots = False

#     class noise(DogFlatSim2SimCfg.noise):
#         add_noise = True
#         noise_level = 0.20

#     class asset(DogFlatSim2SimCfg.asset):
#         # Match the small numerical joint inertia required by the MuJoCo
#         # low-inertia model while retaining Isaac Gym contact dynamics.
#         armature = 0.002


# class DogFlatSim2SimRobustCfgPPO(DogFlatSim2SimCfgPPO):
#     class policy(DogFlatSim2SimCfgPPO.policy):
#         init_noise_std = 0.08

#     class algorithm(DogFlatSim2SimCfgPPO.algorithm):
#         learning_rate = 3.0e-5

#     class runner(DogFlatSim2SimCfgPPO.runner):
#         run_name = 'step52_armature_domain_rand_from_step50_50'
#         max_iterations = 300
#         save_interval = 50


# class DogFriendCfg(LeggedRobotCfg):
#     class env(LeggedRobotCfg.env):
#         # Compatibility task for a friend's checkpoint with 48-dim observations.
#         # It intentionally uses the base LeggedRobot observation, not DogRobot's
#         # extra gait-clock observations.
#         num_observations = 48
#         num_envs = 1

#     class terrain(LeggedRobotCfg.terrain):
#         mesh_type = 'plane'
#         measure_heights = False
#         curriculum = False

#     class commands(LeggedRobotCfg.commands):
#         heading_command = False
#         resampling_time = 4.0

#         class ranges(LeggedRobotCfg.commands.ranges):
#             lin_vel_x = [0.20, 0.25]
#             lin_vel_y = [0.0, 0.0]
#             ang_vel_yaw = [0.0, 0.0]

#     class init_state(LeggedRobotCfg.init_state):
#         rot = [0.0, 0.0, 0.0, 1.0]
#         lin_vel = [0.0, 0.0, 0.0]
#         ang_vel = [0.0, 0.0, 0.0]
#         default_joint_angles = {
#             'FL_hip_joint': 0.1,
#             'FL_thigh_joint': 0.9,
#             'FL_calf_joint': -1.55,
#             'FR_hip_joint': -0.1,
#             'FR_thigh_joint': 0.9,
#             'FR_calf_joint': -1.55,
#             'RL_hip_joint': 0.1,
#             'RL_thigh_joint': 0.9,
#             'RL_calf_joint': -1.55,
#             'RR_hip_joint': -0.1,
#             'RR_thigh_joint': 0.9,
#             'RR_calf_joint': -1.55,
#         }

#     class control(LeggedRobotCfg.control):
#         control_type = 'P'
#         stiffness = {'joint': 32.0}
#         damping = {'joint': 2.8}
#         action_scale = 0.18
#         decimation = 4

#     class asset(LeggedRobotCfg.asset):
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
#         name = 'dog_friend'
#         foot_name = 'foot'
#         penalize_contacts_on = ['hip', 'thigh', 'calf']
#         terminate_after_contacts_on = ['base', 'hip']
#         collapse_fixed_joints = False
#         flip_visual_attachments = False
#         self_collisions = 1

#     class domain_rand(LeggedRobotCfg.domain_rand):
#         randomize_friction = False
#         push_robots = False

#     class rewards(LeggedRobotCfg.rewards):
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.35

#         class scales(LeggedRobotCfg.rewards.scales):
#             tracking_lin_vel = 1.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -2.0
#             torques = -0.00008
#             dof_acc = -1.0e-6
#             collision = -1.0
#             action_rate = -0.01
#             dof_pos_limits = -5.0

#     class noise(LeggedRobotCfg.noise):
#         add_noise = False

#     class viewer(LeggedRobotCfg.viewer):
#         pos = [2.0, -3.0, 1.5]
#         lookat = [0.0, 0.0, 0.35]


# class DogFriendCfgPPO(LeggedRobotCfgPPO):
#     class policy(LeggedRobotCfgPPO.policy):
#         init_noise_std = 0.7
#         actor_hidden_dims = [128, 128, 128]
#         critic_hidden_dims = [128, 128, 128]
#         activation = 'elu'

#     class runner(LeggedRobotCfgPPO.runner):
#         run_name = 'friend_model_2000_compat'
#         experiment_name = 'flat_dog_friend'
#         load_run = -1
#         max_iterations = 1
#         save_interval = 1


# class DogSimpleCfg(LeggedRobotCfg):
#     class env(LeggedRobotCfg.env):
#         # Clean Go2/A1-style baseline: base 48-dim observations only.
#         num_observations = 48
#         num_envs = 384

#     class terrain(LeggedRobotCfg.terrain):
#         mesh_type = 'plane'
#         measure_heights = False
#         curriculum = False

#     class commands(LeggedRobotCfg.commands):
#         heading_command = False
#         resampling_time = 4.0

#         class ranges(LeggedRobotCfg.commands.ranges):
#             lin_vel_x = [0.20, 0.25]
#             lin_vel_y = [0.0, 0.0]
#             ang_vel_yaw = [0.0, 0.0]

#     class init_state(LeggedRobotCfg.init_state):
#         pos = [0.0, 0.0, 0.35]
#         rot = [0.0, 0.0, 0.0, 1.0]
#         lin_vel = [0.0, 0.0, 0.0]
#         ang_vel = [0.0, 0.0, 0.0]
#         default_joint_angles = {
#             'FL_hip_joint': 0.1,
#             'FL_thigh_joint': 0.9,
#             'FL_calf_joint': -1.55,
#             'FR_hip_joint': -0.1,
#             'FR_thigh_joint': 0.9,
#             'FR_calf_joint': -1.55,
#             'RL_hip_joint': 0.1,
#             'RL_thigh_joint': 0.9,
#             'RL_calf_joint': -1.55,
#             'RR_hip_joint': -0.1,
#             'RR_thigh_joint': 0.9,
#             'RR_calf_joint': -1.55,
#         }

#     class control(LeggedRobotCfg.control):
#         control_type = 'P'
#         stiffness = {'joint': 32.0}
#         damping = {'joint': 2.8}
#         action_scale = 0.18
#         decimation = 4

#     class asset(LeggedRobotCfg.asset):
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
#         name = 'dog_simple'
#         foot_name = 'foot'
#         penalize_contacts_on = ['hip', 'thigh', 'calf']
#         terminate_after_contacts_on = ['base', 'hip']
#         collapse_fixed_joints = False
#         flip_visual_attachments = False
#         self_collisions = 1

#     class domain_rand(LeggedRobotCfg.domain_rand):
#         randomize_friction = False
#         push_robots = False

#     class rewards(LeggedRobotCfg.rewards):
#         only_positive_rewards = True
#         tracking_sigma = 0.04
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.35
#         max_contact_force = 350.0

#         class scales(LeggedRobotCfg.rewards.scales):
#             termination = -2.0
#             tracking_lin_vel = 2.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -1.0
#             base_height = -1.0
#             torques = -0.000025
#             dof_acc = -2.5e-7
#             action_rate = -0.01
#             dof_pos_limits = -10.0
#             feet_air_time = 0.0
#             collision = -1.0
#             stumble = -0.0
#             front_rear_action_balance = 0.6
#             rear_overwork = -1.0
#             front_foot_clearance = 0.5
#             front_foot_drag = -2.0

#     class noise(LeggedRobotCfg.noise):
#         add_noise = False

#     class viewer(LeggedRobotCfg.viewer):
#         pos = [2.0, -3.0, 1.5]
#         lookat = [0.0, 0.0, 0.35]


# class DogSimpleCfgPPO(LeggedRobotCfgPPO):
#     class policy(LeggedRobotCfgPPO.policy):
#         init_noise_std = 0.5
#         actor_hidden_dims = [128, 128, 128]
#         critic_hidden_dims = [128, 128, 128]
#         activation = 'elu'

#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         entropy_coef = 0.003

#     class runner(LeggedRobotCfgPPO.runner):
#         run_name = 'step42_contact_guard'
#         experiment_name = 'flat_dog_simple'
#         load_run = -1
#         max_iterations = 1000
#         save_interval = 50


from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ============================================================================
# 唯一生效的配置类（所有继承已展平，无隐藏父类干扰）
# 对应 task: dog_flat_sim2sim_front_first
# ============================================================================
class DogCleanCfg(LeggedRobotCfg):
    class env:
        num_observations = 60                # 48基础观测 + 8 gait时钟 + 4期望接触
        use_gait_observations = True
        num_envs = 384

    class terrain:
        mesh_type = 'plane'
        measure_heights = False
        curriculum = False

    class commands:
        heading_command = False
        resampling_time = 4.0
        class ranges:
            lin_vel_x = [0.24, 0.34]         # 前进命令范围（m/s）
            lin_vel_y = [0.0, 0.0]           # 侧向命令（禁止横移）
            ang_vel_yaw = [0.0, 0.0]         # 转向命令（禁止转弯）

    class init_state:
        pos = [0.0, 0.0, 0.32]               # 初始 base 高度
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {             # 初始站姿（略蹲）
            'FL_hip_joint': 0.1,
            'FL_thigh_joint': 0.95,
            'FL_calf_joint': -1.65,
            'FR_hip_joint': -0.1,
            'FR_thigh_joint': 0.95,
            'FR_calf_joint': -1.65,
            'RL_hip_joint': 0.1,
            'RL_thigh_joint': 0.95,
            'RL_calf_joint': -1.65,
            'RR_hip_joint': -0.1,
            'RR_thigh_joint': 0.95,
            'RR_calf_joint': -1.65,
        }

    class control:
        control_type = 'P'
        stiffness = {'joint': 32.0}
        damping = {'joint': 3.5}
        action_scale = 0.14
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
        name = 'dog'
        foot_name = 'foot'
        penalize_contacts_on = ['hip', 'thigh', 'calf']
        terminate_after_contacts_on = ['base', 'hip']
        collapse_fixed_joints = False
        flip_visual_attachments = False
        self_collisions = 1

    class domain_rand:
        randomize_friction = False
        push_robots = False

    class rewards:
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.31
        max_contact_force = 350.0
        tracking_sigma = 0.04
        world_tracking_sigma = 0.04
        stable_speed_sigma = 0.025
        command_motion_slope = 25.0
        command_motion_threshold = 0.08
        phase_motion_min_speed = 0.02
        phase_motion_full_speed = 0.16

        class scales:
            # ================================================================
            # 1. 生存与终止惩罚（保命底线）
            # ================================================================
            termination = -500.0   # 摔倒/触发终止时一次性扣分，绝对值极大

            # ================================================================
            # 2. 速度命令跟踪（主任务）
            # ================================================================
            tracking_lin_vel = 0.6          # 本体坐标系下跟踪前进速度
            tracking_ang_vel = 0.2          # 跟踪偏航角速度（命令=0，故惩罚转动）
            world_velocity_tracking = 1.0   # 世界坐标系下综合速度跟踪（含直行门控）
            stable_command_speed = 1.0      # 稳定门控速度奖励：匹配×直行×不跳×支撑×直立

            # ================================================================
            # 3. 前进激励（防止原地刷分、逼着往前走）
            # ================================================================
            forward_vel = 0.6               # 本体 x 方向速度正反馈
            world_forward_vel = 0.6         # 世界 x 方向速度正反馈（防转歪）
            forward_progress = 0.5          # 累计 X 位移奖励（封顶 2m）
            step_forward_progress = 2.0     # 每个控制步的净 X 位移奖励
            straight_forward_progress = 1.2 # 只奖励“朝世界 X、不横漂、不偏航”的位移
            straight_forward_vel = 1.4      # 同上，基于速度版本
            commanded_forward_motion = 1.0  # Sigmoid 型：速度>阈值就奖励前进

            # ================================================================
            # 4. 运动不足惩罚（逼着动起来，避免站着刷分）
            # ================================================================
            forward_speed_floor = -4.0      # 速度低于命令值时连续惩罚
            no_forward_motion = -4.0        # 站立但速度<0.12 时惩罚
            low_speed = -2.0                # 速度<0.16 时惩罚（有命令前提下）
            stand_motion = -3.0             # 有命令但几乎不动时惩罚

            # ================================================================
            # 5. 步态与足端轨迹（让走姿像狗，接近 trot）
            # ================================================================
            gait_balance = 0.03             # 奖励对角支撑（FL+RR 或 FR+RL）
            diagonal_clock_gait = 0.35      # 用时钟相位匹配期望接触状态（核心节拍）
            trot_joint_pose = 0.15          # 密集姿态先验：摆动相抬腿，支撑相站姿
            front_swing_pose = 0.20         # 前腿摆动相目标姿态（抬 thigh、收 calf）
            front_swing_motion = 0.50       # 实际前脚抬起+向前运动的多维度奖励

            # ================================================================
            # 6. 足端接触与支撑质量（接触管理）
            # ================================================================
            tracking_contacts_force = -0.12 # 惩罚摆动相脚触地
            tracking_contacts_vel = -0.08   # 惩罚支撑相脚滑动
            foot_clearance_clock = 0.12     # 摆动相足端抬到目标高度奖励
            support_stability = 0.10        # 奖励 2~3 足稳定支撑（前后左右都有人撑）
            airborne = -0.8                 # 惩罚四足同时离地或少于 2 足支撑
            foot_impact = -0.20             # 惩罚落地瞬间垂直冲击过大

            # ================================================================
            # 7. 身体姿态与稳定性（不歪、不倒、不漂）
            # ================================================================
            lateral_vel = -8.0              # 惩罚本体侧向速度（平方）
            world_lateral_vel = -10.0       # 惩罚世界侧向速度（更重）
            lateral_drift = -10.0           # 惩罚相对出生点的横向漂移（平方，截断 2m）
            yaw_rate = -7.0                 # 惩罚偏航角速度（平方）
            yaw_drift = -8.0                # 惩罚朝向偏离世界 X 方向（平方）
            vertical_motion = -1.0          # 惩罚身体上下弹跳（z 方向速度平方）

            # ================================================================
            # 8. 前后腿协调（专门针对“后腿猛推、前腿拖地”的漏洞）
            # ================================================================
            rear_push_jump = -0.80          # 惩罚后腿同时猛蹬 + 身体向上弹
            front_support = 0.45            # 奖励前后腿支撑力平衡（前腿不能太弱）
            front_leg_activity = 0.05       # 奖励前后腿动作幅度接近
            front_slip = -0.30              # 惩罚前足着地时滑动
            rear_push_without_front_ready = -2.80  # 【核心】前腿没准备好时后腿猛蹬的重罚
            base_ahead_of_support = -3.0    # 惩罚重心跑到支撑足前方太多（狗刨式跑法）

            # ================================================================
            # 9. 硬件友好性（平滑性）
            # ================================================================
            action_rate = -0.04             # 惩罚相邻控制步动作变化率（平方）

    class normalization:
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 0.15

    class viewer:
        pos = [2.0, -3.0, 1.5]
        lookat = [0.0, 0.0, 0.35]


# ============================================================================
# PPO 训练超参数（同样展平）
# ============================================================================
class DogCleanCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.06
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm:
        entropy_coef = 0.0
        learning_rate = 2.0e-5
        schedule = 'fixed'

    class runner:
        run_name = 'step53_front_first_from_step50_50'
        experiment_name = 'flat_dog'
        load_run = -1
        max_iterations = 1200
        save_interval = 100