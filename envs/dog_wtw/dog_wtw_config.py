from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DogWtwFlatCfg(LeggedRobotCfg):
    """Small WTW-style flat-walk experiment for the custom dog URDF."""

    class env(LeggedRobotCfg.env):
        num_observations = 60
        use_gait_observations = True
        num_envs = 256
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        curriculum = False

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(LeggedRobotCfg.commands.ranges):
            # Match the step53 source policy during reward fine-tuning.
            lin_vel_x = [0.24, 0.34]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.10,
            'FL_thigh_joint': 0.95,
            'FL_calf_joint': -1.65,
            'FR_hip_joint': -0.10,
            'FR_thigh_joint': 0.95,
            'FR_calf_joint': -1.65,
            'RL_hip_joint': 0.10,
            'RL_thigh_joint': 0.95,
            'RL_calf_joint': -1.65,
            'RR_hip_joint': -0.10,
            'RR_thigh_joint': 0.95,
            'RR_calf_joint': -1.65,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 32.0}
        damping = {'joint': 3.5}
        action_scale = 0.14
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog.urdf'
        name = 'dog_wtw'
        foot_name = 'foot'
        penalize_contacts_on = ['hip', 'thigh', 'calf']
        terminate_after_contacts_on = ['base', 'hip']
        collapse_fixed_joints = False
        flip_visual_attachments = False
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.05
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        max_contact_force = 350.0
        phase_motion_min_speed = 0.02
        phase_motion_full_speed = 0.14

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0

            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.4

            lin_vel_z = -2.0
            ang_vel_xy = -0.08
            orientation = -2.5
            base_height = -1.2

            torques = -0.00008
            dof_acc = -5.0e-7
            action_rate = -0.05
            dof_pos_limits = -5.0
            collision = -1.0

            feet_air_time = 0.20
            gait_balance = 0.01
            tracking_contacts_force = -0.08
            tracking_contacts_vel = -0.06
            foot_clearance_clock = 0.10
            support_stability = 0.01
            foot_impact = -0.05

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.10

    class viewer(LeggedRobotCfg.viewer):
        pos = [2.0, -3.0, 1.5]
        lookat = [0.0, 0.0, 0.33]


class DogWtwFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        # Exact network layout of flat_dog step53/model_1200.
        init_noise_std = 0.06
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.0
        learning_rate = 2.0e-5
        schedule = 'fixed'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'WTW_WARMSTART_STEP53_1200_V1'
        experiment_name = 'flat_dog_wtw'
        load_run = -1
        max_iterations = 200
        save_interval = 25
