from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


DOG_WTW_URDF = (
    "/home/xiehan/下载/workspace_summer/wtw/walk-these-ways/"
    "resources/robots/dog/urdf/dog.urdf"
)
DOG_WTW_MASS137_URDF = (
    "/home/xiehan/下载/workspace_summer/wtw/walk-these-ways/"
    "resources/robots/dog/urdf/dog_mass137.urdf"
)
DOG_ORIGINAL17_URDF = (
    "/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/"
    "resources/robots/dog/urdf/dog_original17.urdf"
)
DOG_ORIGINAL23_URDF = (
    "/home/xiehan/下载/workspace_summer/rl_stack/legged_gym/"
    "resources/robots/dog/urdf/dog_original23.urdf"
)


class DogWTWLightACfg(LeggedRobotCfg):
    """Experiment A: fixed Dog dynamics and WTW base locomotion rewards."""

    class env(LeggedRobotCfg.env):
        num_envs = 256
        num_observations = 63
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 14
        resampling_time = 20.0
        gait_frequency = 1.5
        gait_phase = 0.5
        gait_offset = 0.0
        gait_bound = 0.0
        gait_duration = 0.5
        footswing_height = 0.07
        body_height_cmd = 0.0
        body_pitch = 0.0
        body_roll = 0.0
        stance_width = 0.276
        stance_length = 0.429

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.25, 0.25]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "FL_hip_joint": 0.10,
            "FL_thigh_joint": 0.75,
            "FL_calf_joint": -1.20,
            "FR_hip_joint": -0.10,
            "FR_thigh_joint": 0.75,
            "FR_calf_joint": -1.20,
            "RL_hip_joint": 0.10,
            "RL_thigh_joint": 0.75,
            "RL_calf_joint": -1.20,
            "RR_hip_joint": -0.10,
            "RR_thigh_joint": 0.75,
            "RR_calf_joint": -1.20,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 32.0}
        damping = {"joint": 3.5}
        action_scale = 0.14
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = DOG_WTW_URDF
        name = "dog_wtw_lite"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip"]
        collapse_fixed_joints = False
        flip_visual_attachments = False
        self_collisions = 1
        armature = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        max_contact_force = 350.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.0
        gait_vel_sigma = 10.0
        foot_radius = 0.02
        contact_force_threshold = 5.0

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            orientation = -5.0
            torques = -0.0001
            dof_acc = -2.5e-7
            feet_air_time = 0.0
            action_rate = -0.01
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            collision = -5.0
            dof_pos_limits = -10.0
            feet_slip = -0.04
            feet_impact_vel = 0.0
            raibert_heuristic = 0.0
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            feet_clearance_cmd_linear = 0.0

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 10.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class diagnostics:
        print_interval_steps = 240
        torque_saturation_fraction = 0.98
        rear_saturation_alert_ratio = 0.10

    class viewer(LeggedRobotCfg.viewer):
        pos = [2.0, -3.0, 1.5]
        lookat = [0.0, 0.0, 0.35]


class DogWTWLightBCfg(DogWTWLightACfg):
    """Experiment B: A plus the official Raibert heuristic."""

    class rewards(DogWTWLightACfg.rewards):
        class scales(DogWTWLightACfg.rewards.scales):
            raibert_heuristic = -10.0


class DogWTWLightCCfg(DogWTWLightBCfg):
    """Experiment C: B plus WTW contact force/velocity timing."""

    class rewards(DogWTWLightBCfg.rewards):
        class scales(DogWTWLightBCfg.rewards.scales):
            ang_vel_xy = -0.02
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0


class DogWTWLightCMass137Cfg(DogWTWLightCCfg):
    """C1 rewards with a 13.707 kg URDF and mass-matched PD gains."""

    class control(DogWTWLightCCfg.control):
        stiffness = {"joint": 20.0}
        damping = {"joint": 2.2}

    class asset(DogWTWLightCCfg.asset):
        file = DOG_WTW_MASS137_URDF


class DogWTWLightCMass137Clearance3Cfg(DogWTWLightCMass137Cfg):
    """Mass-matched C1 with a low swing-foot clearance shaping term."""

    class rewards(DogWTWLightCMass137Cfg.rewards):
        class scales(DogWTWLightCMass137Cfg.rewards.scales):
            feet_clearance_cmd_linear = -3.0


class DogWTWLightOriginal17Contact2Cfg(DogWTWLightCCfg):
    """Original Dog dynamics with 17 Nm limits and softer contact shaping."""

    class rewards(DogWTWLightCCfg.rewards):
        class scales(DogWTWLightCCfg.rewards.scales):
            tracking_contacts_shaped_force = 2.0
            tracking_contacts_shaped_vel = 2.0

    class asset(DogWTWLightCCfg.asset):
        file = DOG_ORIGINAL17_URDF


class DogWTWLightOriginal23Contact2Cfg(DogWTWLightOriginal17Contact2Cfg):
    """Original Dog dynamics with 23 Nm limits and the same contact shaping."""

    class asset(DogWTWLightOriginal17Contact2Cfg.asset):
        file = DOG_ORIGINAL23_URDF


class DogWTWLightCStable2Cfg(DogWTWLightCCfg):
    """C-stable-2: C-stable-1 with softer contact timing shaping."""

    class rewards(DogWTWLightCCfg.rewards):
        class scales(DogWTWLightCCfg.rewards.scales):
            tracking_contacts_shaped_force = 2.0
            tracking_contacts_shaped_vel = 2.0


class DogWTWLightCStable3RaibertCfg(DogWTWLightCStable2Cfg):
    """C-stable-3: C-stable-2 with a softer Raibert foot-placement term."""

    class rewards(DogWTWLightCStable2Cfg.rewards):
        class scales(DogWTWLightCStable2Cfg.rewards.scales):
            raibert_heuristic = -5.0


class DogWTWLightCStable4Cfg(DogWTWLightCStable2Cfg):
    """C-stable-4: C-stable-2 with stronger body angular-velocity damping."""

    class rewards(DogWTWLightCStable2Cfg.rewards):
        class scales(DogWTWLightCStable2Cfg.rewards.scales):
            ang_vel_xy = -0.05


class DogWTWLightCSlow010Cfg(DogWTWLightCCfg):
    """C1 slow-stage: preserve C1 gait rewards while reducing command speed."""

    class commands(DogWTWLightCCfg.commands):
        class ranges(DogWTWLightCCfg.commands.ranges):
            lin_vel_x = [0.10, 0.10]


class DogWTWLightDCfg(DogWTWLightCCfg):
    """Experiment D: C plus WTW swing-foot clearance tracking."""

    class rewards(DogWTWLightCCfg.rewards):
        class scales(DogWTWLightCCfg.rewards.scales):
            ang_vel_xy = -0.001
            feet_clearance_cmd_linear = -30.0


class DogWTWLightAPPOCfg(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = "A_core"
        experiment_name = "dog_wtw_lite"
        max_iterations = 300
        save_interval = 25


class DogWTWLightBPPOCfg(DogWTWLightAPPOCfg):
    class runner(DogWTWLightAPPOCfg.runner):
        run_name = "B_raibert"


class DogWTWLightCPPOCfg(DogWTWLightAPPOCfg):
    class runner(DogWTWLightAPPOCfg.runner):
        run_name = "C_contact"


class DogWTWLightCMass137PPOCfg(DogWTWLightCPPOCfg):
    class runner(DogWTWLightCPPOCfg.runner):
        run_name = "C_mass137_gainmatched"


class DogWTWLightCMass137Clearance3PPOCfg(DogWTWLightCMass137PPOCfg):
    class runner(DogWTWLightCMass137PPOCfg.runner):
        run_name = "C_mass137_clearance3"


class DogWTWLightOriginal17Contact2PPOCfg(DogWTWLightCPPOCfg):
    class runner(DogWTWLightCPPOCfg.runner):
        run_name = "original17_contact2"


class DogWTWLightOriginal23Contact2PPOCfg(DogWTWLightOriginal17Contact2PPOCfg):
    class runner(DogWTWLightOriginal17Contact2PPOCfg.runner):
        run_name = "original23_contact2"


class DogWTWLightCStable2PPOCfg(DogWTWLightCPPOCfg):
    class runner(DogWTWLightCPPOCfg.runner):
        run_name = "C_stable_2_contact"


class DogWTWLightCStable3RaibertPPOCfg(DogWTWLightCStable2PPOCfg):
    class runner(DogWTWLightCStable2PPOCfg.runner):
        run_name = "C_stable_3_raibert"


class DogWTWLightCStable4PPOCfg(DogWTWLightCStable2PPOCfg):
    class runner(DogWTWLightCStable2PPOCfg.runner):
        run_name = "C_stable_4_angvel"


class DogWTWLightCSlow010PPOCfg(DogWTWLightCPPOCfg):
    class runner(DogWTWLightCPPOCfg.runner):
        run_name = "C_slow_010"
        max_iterations = 600


class DogWTWLightDPPOCfg(DogWTWLightAPPOCfg):
    class runner(DogWTWLightAPPOCfg.runner):
        run_name = "D_clearance"
