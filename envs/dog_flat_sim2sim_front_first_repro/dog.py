import os
import xml.etree.ElementTree as ET

import torch
from isaacgym import gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot


class DogSimpleRobot(LeggedRobot):
    def _create_envs(self):
        super()._create_envs()

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_indices = []
        for name in foot_names:
            index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            if index < 0:
                raise RuntimeError(f"Cannot find rigid body {name} for dog_simple rewards")
            ordered_indices.append(index)
        self.simple_feet_indices = torch.tensor(ordered_indices, dtype=torch.long, device=self.device, requires_grad=False)

    def _init_buffers(self):
        super()._init_buffers()

        leg_joint_names = [
            ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        ]
        leg_joint_indices = []
        for leg_names in leg_joint_names:
            leg_joint_indices.append([self.dof_names.index(name) for name in leg_names])
        self.simple_leg_joint_indices = torch.tensor(leg_joint_indices, dtype=torch.long, device=self.device, requires_grad=False)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = torch.reshape(gymtorch.wrap_tensor(rigid_body_state), (self.num_envs, self.num_bodies, 13))

    def _moving_command(self):
        return (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()

    def check_termination(self):
        super().check_termination()

        # Keep the simple task honest: falling/crouching must end the episode,
        # otherwise the policy can rush forward once and collect speed reward.
        settled = self.episode_length_buf > int(0.5 / self.dt)
        too_low = settled & (self.root_states[:, 2] < 0.245)
        too_tilted = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) > 0.45
        self.reset_buf |= too_low | too_tilted

    def _reward_front_rear_action_balance(self):
        # The previous policy exploited rear-leg pushing. Reward comparable front/rear action use.
        front = torch.mean(torch.abs(self.actions[:, self.simple_leg_joint_indices[0:2].reshape(-1)]), dim=1)
        rear = torch.mean(torch.abs(self.actions[:, self.simple_leg_joint_indices[2:4].reshape(-1)]), dim=1)
        return torch.clamp(front / (rear + 1.0e-3), min=0.0, max=1.0) * self._moving_command()

    def _reward_rear_overwork(self):
        front = torch.mean(torch.abs(self.actions[:, self.simple_leg_joint_indices[0:2].reshape(-1)]), dim=1)
        rear = torch.mean(torch.abs(self.actions[:, self.simple_leg_joint_indices[2:4].reshape(-1)]), dim=1)
        return torch.clamp(rear - front - 0.05, min=0.0, max=2.0) * self._moving_command()

    def _reward_front_foot_clearance(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        front_foot_z = self.rigid_body_states[:, self.simple_feet_indices[0:2], 2]
        clearance = torch.clamp((front_foot_z - 0.025) / 0.04, min=0.0, max=1.0)
        return torch.mean(clearance, dim=1) * self._moving_command()

    def _reward_front_foot_drag(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        front_forces = self.contact_forces[:, self.simple_feet_indices[0:2], 2]
        front_foot_vel_xy = self.rigid_body_states[:, self.simple_feet_indices[0:2], 7:9]
        in_contact = front_forces > 5.0
        slip_speed = torch.norm(front_foot_vel_xy, dim=2)
        return torch.sum(slip_speed * in_contact.float(), dim=1) * self._moving_command()


class DogRobot(LeggedRobot):
    def _create_envs(self):
        super()._create_envs()
        self._print_asset_diagnostics()

        # URDF rigid-body order is not guaranteed to be FL, FR, RL, RR. The gait
        # reward needs this exact semantic order, otherwise diagonal pairs are
        # computed incorrectly and the policy can learn rear-leg pushing.
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_indices = []
        for name in foot_names:
            index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            if index < 0:
                raise RuntimeError(f"Cannot find rigid body {name} for dog gait reward")
            ordered_indices.append(index)
        self.gait_feet_indices = torch.tensor(ordered_indices, dtype=torch.long, device=self.device, requires_grad=False)

    def _print_asset_diagnostics(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_path = os.path.abspath(asset_path)
        expected_dof_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        print("[DOG_ASSET_DIAG] URDF absolute path:", asset_path)
        print("[DOG_ASSET_DIAG] robot name:", self.cfg.asset.name)
        print("[DOG_ASSET_DIAG] num_bodies:", self.num_bodies)
        print("[DOG_ASSET_DIAG] num_dofs:", self.num_dofs)
        print("[DOG_ASSET_DIAG] DOF names:", list(self.dof_names))
        print("[DOG_ASSET_DIAG] DOF lower limits:", self.dof_pos_limits[:, 0].detach().cpu().tolist())
        print("[DOG_ASSET_DIAG] DOF upper limits:", self.dof_pos_limits[:, 1].detach().cpu().tolist())
        print("[DOG_ASSET_DIAG] DOF effort limits:", self.torque_limits.detach().cpu().tolist())
        if list(self.dof_names) == expected_dof_order:
            print("[DOG_ASSET_DIAG] DOF_ORDER_OK")
        else:
            print("[DOG_ASSET_DIAG] DOF_ORDER_MISMATCH")
            print("[DOG_ASSET_DIAG] expected order:", expected_dof_order)
            print("[DOG_ASSET_DIAG] actual order:", list(self.dof_names))
        try:
            root = ET.parse(asset_path).getroot()
            total_mass = 0.0
            foot_spheres = []
            urdf_joint_limits = []
            for link in root.findall("link"):
                inertial = link.find("inertial")
                if inertial is not None and inertial.find("mass") is not None:
                    total_mass += float(inertial.find("mass").get("value"))
                if link.get("name", "").endswith("_foot"):
                    for collision in link.findall("collision"):
                        geometry = collision.find("geometry")
                        sphere = geometry.find("sphere") if geometry is not None else None
                        if sphere is not None:
                            foot_spheres.append((link.get("name"), sphere.get("radius")))
            for joint in root.findall("joint"):
                limit = joint.find("limit")
                if limit is not None and limit.get("effort") is not None:
                    urdf_joint_limits.append((
                        joint.get("name"),
                        limit.get("lower"),
                        limit.get("upper"),
                        limit.get("effort"),
                    ))
            print("[DOG_ASSET_DIAG] URDF total mass:", total_mass)
            print("[DOG_ASSET_DIAG] URDF joint limits lower/upper/effort:", urdf_joint_limits)
            print("[DOG_ASSET_DIAG] foot sphere collisions:", foot_spheres)
        except Exception as exc:
            print("[DOG_ASSET_DIAG] failed to parse URDF diagnostics:", repr(exc))

    def _init_buffers(self):
        super()._init_buffers()
        # Joint order in this URDF is FL, FR, RR, RL. Rewards below need semantic
        # leg order FL, FR, RL, RR, so build the mapping by joint name once.
        leg_joint_names = [
            ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        ]
        leg_joint_indices = []
        for leg_names in leg_joint_names:
            leg_joint_indices.append([self.dof_names.index(name) for name in leg_names])
        self.leg_joint_indices = torch.tensor(leg_joint_indices, dtype=torch.long, device=self.device, requires_grad=False)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = torch.reshape(gymtorch.wrap_tensor(rigid_body_state), (self.num_envs, self.num_bodies, 13))
        self.last_foot_pos = self.rigid_body_states[:, self.gait_feet_indices, 0:3].clone()
        self.last_root_x = self.root_states[:, 0].clone()
        self.gait_clock_inputs = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False,
        )

    def _compute_torques(self, actions):
        transition_steps = int(getattr(self.cfg.control, "deploy_match_transition_steps", 0))
        if transition_steps > 0:
            alpha = torch.clamp(
                torch.tensor(float(self.common_step_counter), device=self.device)
                / float(transition_steps),
                min=0.0,
                max=1.0,
            )
            source_stiffness = float(getattr(self.cfg.control, "deploy_match_source_stiffness", 32.0))
            source_damping = float(getattr(self.cfg.control, "deploy_match_source_damping", 3.5))
            source_action_scale = float(getattr(self.cfg.control, "deploy_match_source_action_scale", 0.14))
            target_stiffness = float(getattr(self.cfg.control, "deploy_match_target_stiffness", 50.0))
            target_damping = float(getattr(self.cfg.control, "deploy_match_target_damping", 5.0))
            target_action_scale = float(getattr(self.cfg.control, "deploy_match_target_action_scale", 0.10))
            stiffness = (1.0 - alpha) * source_stiffness + alpha * target_stiffness
            damping = (1.0 - alpha) * source_damping + alpha * target_damping
            action_scale = (1.0 - alpha) * source_action_scale + alpha * target_action_scale
            actions_scaled = actions * action_scale
            torques = stiffness * (actions_scaled + self.default_dof_pos - self.dof_pos) - damping * self.dof_vel
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            torques = super()._compute_torques(actions)
        if getattr(self.cfg.domain_rand, "randomize_motor_strength", False):
            torques = torques * self.motor_strengths
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super().post_physics_step()
        self.last_foot_pos[:] = self.rigid_body_states[:, self.gait_feet_indices, 0:3]
        self.last_root_x[:] = self.root_states[:, 0]

    def _reset_dofs(self, env_ids):
        # 原版 legged_gym 会把关节角随机到 0.5-1.5 倍默认值；对这个 URDF 太激进，
        # 容易一出生就低身位。这里改成默认站姿附近的小扰动。
        noise = 0.05 * (2.0 * torch.rand((len(env_ids), self.num_dof), device=self.device) - 1.0)
        self.dof_pos[env_ids] = self.default_dof_pos + noise
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        # 原版 reset 会给 base 一个 [-0.5, 0.5] 的随机初速度；早期训练时这会放大摔倒/倒退。
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_gait_clock()

    def _update_gait_clock(self):
        # Trot gait from walk-these-ways style clock inputs: FL/RR share phase,
        # FR/RL are 180 degrees apart. The policy can now see the gait timing.
        frequency = 0.95
        gait_phase = torch.remainder(self.episode_length_buf.float() * self.dt * frequency, 1.0)
        phases = torch.stack((
            gait_phase + 0.5,  # FL
            gait_phase,        # FR
            gait_phase,        # RL
            gait_phase + 0.5,  # RR
        ), dim=1)
        self.foot_phase[:] = torch.remainder(phases, 1.0)
        phase_angle = 2.0 * torch.pi * self.foot_phase
        self.gait_clock_inputs[:, 0:4] = torch.sin(phase_angle)
        self.gait_clock_inputs[:, 4:8] = torch.cos(phase_angle)

        stance_center = 0.25
        phase_distance = torch.abs(torch.remainder(self.foot_phase - stance_center + 0.5, 1.0) - 0.5)
        self.desired_contact_states[:] = torch.clamp(1.0 - phase_distance / 0.25, min=0.0, max=1.0)

    def compute_observations(self):
        use_gait_observations = getattr(self.cfg.env, "use_gait_observations", True)
        if not use_gait_observations:
            super().compute_observations()
            return

        # LeggedRobot adds noise while its observation still has 48 elements.
        # For clock tasks the allocated noise vector has 60 elements, so append
        # the phase inputs first and apply noise once to the complete vector.
        add_noise = self.add_noise
        self.add_noise = False
        try:
            super().compute_observations()
        finally:
            self.add_noise = add_noise
        self.obs_buf = torch.cat((self.obs_buf, self.gait_clock_inputs, self.desired_contact_states), dim=-1)
        if add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def check_termination(self):
        super().check_termination()

        # End episodes that become visibly non-straight even when the base stays upright.
        # This prevents long, sideways runs from dominating the velocity rewards.
        settled = self.episode_length_buf > int(0.5 / self.dt)
        too_low = settled & (self.root_states[:, 2] < 0.245)
        too_tilted = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) > 0.75
        lateral_drift = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])

        quat = self.base_quat
        siny_cosp = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cosy_cosp = 1.0 - 2.0 * (torch.square(quat[:, 1]) + torch.square(quat[:, 2]))
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        too_far_sideways = settled & (lateral_drift > 1.35)
        too_far_turned = settled & (torch.abs(yaw) > 1.0)

        self.reset_buf |= too_low | too_tilted | too_far_sideways | too_far_turned

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # 这些指标只用于日志分析：比单看 reward 更容易判断策略是否真的学会前进。
        episode_length = self.episode_length_buf[env_ids].float()
        forward_distance = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        lateral_drift = torch.abs(self.root_states[env_ids, 1] - self.env_origins[env_ids, 1])
        commanded_forward_velocity = self.commands[env_ids, 0]
        actual_forward_velocity = self.root_states[env_ids, 7]
        velocity_tracking_error = torch.abs(commanded_forward_velocity - actual_forward_velocity)
        mean_abs_action = torch.mean(torch.abs(self.actions[env_ids]), dim=1)
        mean_action_rate = torch.mean(torch.abs(self.actions[env_ids] - self.last_actions[env_ids]), dim=1)
        mean_joint_velocity = torch.mean(torch.abs(self.dof_vel[env_ids]), dim=1)
        mean_abs_torque = torch.mean(torch.abs(self.torques[env_ids]), dim=1)
        torque_limit = self.torque_limits.unsqueeze(0).expand(len(env_ids), -1)
        torque_saturation_ratio = torch.mean(
            (torch.abs(self.torques[env_ids]) > 0.98 * torque_limit).float(),
            dim=1,
        )
        foot_contact = self.contact_forces[env_ids][:, self.gait_feet_indices, 2] > 5.0
        num_feet_in_contact = torch.sum(foot_contact.float(), dim=1)
        num_swinging_feet = 4.0 - num_feet_in_contact
        time_out = self.time_out_buf[env_ids]
        base_contact = self.reset_buf[env_ids] & ~time_out
        distance_success = time_out & (forward_distance > 3.0)
        straight_success = distance_success & (lateral_drift < 1.0)
        success = distance_success & (lateral_drift < 1.5)

        super().reset_idx(env_ids)
        self.last_root_x[env_ids] = self.root_states[env_ids, 0]

        if getattr(self.cfg.domain_rand, "randomize_motor_strength", False):
            low, high = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids] = low + (high - low) * torch.rand(
                len(env_ids), self.num_actions, device=self.device
            )

        self.extras["episode"]["success_rate"] = success.float().mean()
        self.extras["episode"]["distance_success_rate"] = distance_success.float().mean()
        self.extras["episode"]["straight_success_rate"] = straight_success.float().mean()
        self.extras["episode"]["base_contact"] = base_contact.float().mean()
        self.extras["episode"]["time_out"] = time_out.float().mean()
        self.extras["episode"]["forward_distance"] = forward_distance.mean()
        self.extras["episode"]["lateral_drift"] = lateral_drift.mean()
        self.extras["episode"]["alive_ratio"] = (episode_length / self.max_episode_length).mean()
        self.extras["episode"]["commanded_forward_velocity"] = commanded_forward_velocity.mean()
        self.extras["episode"]["actual_forward_velocity"] = actual_forward_velocity.mean()
        self.extras["episode"]["velocity_tracking_error"] = velocity_tracking_error.mean()
        self.extras["episode"]["mean_abs_action"] = mean_abs_action.mean()
        self.extras["episode"]["mean_action_rate"] = mean_action_rate.mean()
        self.extras["episode"]["mean_joint_velocity"] = mean_joint_velocity.mean()
        self.extras["episode"]["mean_abs_torque"] = mean_abs_torque.mean()
        self.extras["episode"]["torque_saturation_ratio"] = torque_saturation_ratio.mean()
        self.extras["episode"]["num_feet_in_contact"] = num_feet_in_contact.mean()
        self.extras["episode"]["num_swinging_feet"] = num_swinging_feet.mean()

    def _upright_height_gate(self):
        height_gate = torch.clamp((self.root_states[:, 2] - 0.24) / 0.08, min=0.0, max=1.0)
        upright_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        upright_gate = torch.exp(-3.5 * upright_error)
        return height_gate * upright_gate

    def _forward_motion_gate(self):
        min_speed = getattr(self.cfg.rewards, "phase_motion_min_speed", 0.0)
        full_speed = getattr(self.cfg.rewards, "phase_motion_full_speed", 0.12)
        return torch.clamp(
            (self.root_states[:, 7] - min_speed) / max(full_speed - min_speed, 1.0e-3),
            min=0.0,
            max=1.0,
        ) * self._upright_height_gate()

    def _reward_forward_vel(self):
        # 给真实向前速度一个直接正反馈，避免策略只靠原地摇晃获得 tracking 奖励。
        return torch.clip(self.base_lin_vel[:, 0], min=0.0, max=1.0)

    def _reward_backward_vel(self):
        # 明确惩罚倒退，避免策略用后腿乱蹬但整体不前进。
        return torch.clip(-self.base_lin_vel[:, 0], min=0.0, max=1.0)

    def _reward_world_forward_vel(self):
        # 用世界坐标 X 方向约束“真的往前走”，避免身体转歪后仍靠本体前向速度刷分。
        return torch.clip(self.root_states[:, 7], min=0.0, max=1.0)

    def _reward_forward_progress(self):
        # 直接奖励离出生点越来越远，补上速度 tracking 在低速命令下容易“原地混分”的漏洞。
        distance = self.root_states[:, 0] - self.env_origins[:, 0]
        # 进度奖励封顶低一点，防止“身体弹出去”成为局部最优。
        return torch.clip(distance, min=0.0, max=2.0) / 2.0 * self._upright_height_gate()

    def _reward_step_forward_progress(self):
        # 每个控制步的净 +X 位移。相比瞬时速度，更不容易被前后摆动骗分。
        delta_x = self.root_states[:, 0] - self.last_root_x
        return torch.clamp(delta_x / self.dt, min=0.0, max=0.35) / 0.35 * self._upright_height_gate()

    def _reward_backward_progress(self):
        # step20 学到了节拍但会倒着/斜着挪；这里直接惩罚世界 X 负方向位移。
        distance = self.root_states[:, 0] - self.env_origins[:, 0]
        return torch.clip(-distance, min=0.0, max=3.0) / 3.0

    def _reward_low_speed(self):
        # 有前进命令时，世界 X 速度低于 0.18m/s 就惩罚，避免站着、小碎步或趴着刷分。
        speed_shortfall = torch.clip(0.16 - self.root_states[:, 7], min=0.0, max=0.5)
        return speed_shortfall * (self.commands[:, 0] > 0.2)

    def _reward_straight_forward_progress(self):
        # 只奖励朝世界 X 方向前进且没有明显偏航/横漂的位移，避免绕圈也拿进度分。
        quat = self.base_quat
        siny_cosp = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cosy_cosp = 1.0 - 2.0 * (torch.square(quat[:, 1]) + torch.square(quat[:, 2]))
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        lateral_drift = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        distance = torch.clip(self.root_states[:, 0] - self.env_origins[:, 0], min=0.0, max=5.0) / 5.0
        progress_gate = torch.exp(-0.35 * torch.square(torch.clip(lateral_drift, max=3.0)) - 0.8 * torch.square(yaw))
        return distance * progress_gate

    def _reward_straight_forward_vel(self):
        # 只有“朝世界 X 方向、横向漂移小、偏航小”的前进才给高奖励。
        # 这项用于堵住斜着跑/转圈也能拿到 tracking 分的漏洞。
        quat = self.base_quat
        siny_cosp = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cosy_cosp = 1.0 - 2.0 * (torch.square(quat[:, 1]) + torch.square(quat[:, 2]))
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        lateral_drift = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        straight_gate = torch.exp(
            -2.0 * torch.square(self.root_states[:, 8])
            -0.6 * torch.square(torch.clip(lateral_drift, max=3.0))
            -1.2 * torch.square(yaw)
        )
        return torch.clip(self.root_states[:, 7], min=0.0, max=1.0) * straight_gate

    def _reward_world_velocity_tracking(self):
        # 用世界坐标直接跟踪“向前、零侧向、零偏航角速度”。
        vel_error = torch.square(self.commands[:, 0] - self.root_states[:, 7])
        vel_error += 2.0 * torch.square(self.root_states[:, 8])
        vel_error += 0.5 * torch.square(self.base_ang_vel[:, 2])
        sigma = getattr(self.cfg.rewards, "world_tracking_sigma", 0.2)
        return torch.exp(-vel_error / sigma)

    def _reward_stable_command_speed(self):
        # step23 主奖励：按命令速度向世界 X 方向稳定前进。稳定门控避免重新学成跳跃冲刺。
        vx = self.root_states[:, 7]
        vy = self.root_states[:, 8]
        vz = self.root_states[:, 9]
        yaw_rate = self.base_ang_vel[:, 2]
        sigma = getattr(self.cfg.rewards, "stable_speed_sigma", 0.10)
        speed_match = torch.exp(-torch.square(self.commands[:, 0] - vx) / sigma)
        straight_gate = torch.exp(-1.2 * torch.square(vy) - 0.4 * torch.square(yaw_rate))
        low_hop_gate = torch.exp(-8.0 * torch.square(vz))
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > 5.0
        support_count = torch.sum(contact.float(), dim=1)
        support_gate = torch.clamp((support_count - 1.0) / 2.0, min=0.0, max=1.0)
        return speed_match * straight_gate * low_hop_gate * support_gate * self._upright_height_gate()

    def _reward_forward_speed_floor(self):
        # 给低于目标速度的情况一个连续惩罚，比 episode 末尾位移更早产生学习信号。
        shortfall = torch.clamp(self.commands[:, 0] - self.root_states[:, 7], min=0.0, max=0.5)
        stable_gate = torch.exp(-4.0 * torch.square(self.root_states[:, 9]))
        return shortfall * stable_gate

    def _reward_no_forward_motion(self):
        # 站高但几乎不前进，是 step24/25 的局部最优；这里给连续惩罚。
        standing_upright = self._upright_height_gate()
        shortfall = torch.clamp(0.12 - self.root_states[:, 7], min=0.0, max=0.4)
        return shortfall * standing_upright

    def _reward_commanded_forward_motion(self):
        # 只要站高并且世界 X 速度朝目标方向变好，就给密集奖励。
        # step29 改成 sigmoid，避免 vx<0.02 时 clamp 造成零梯度。
        vx = self.root_states[:, 7]
        slope = getattr(self.cfg.rewards, "command_motion_slope", 12.0)
        threshold = getattr(self.cfg.rewards, "command_motion_threshold", 0.05)
        forward_gate = torch.sigmoid(slope * (vx - threshold))
        straight_gate = torch.exp(-1.5 * torch.square(self.root_states[:, 8]) - 0.4 * torch.square(self.base_ang_vel[:, 2]))
        return forward_gate * straight_gate * self._upright_height_gate()

    def _reward_world_backward_motion(self):
        # 世界 X 方向倒退要明确扣掉，不让原地扭身或倒挪混过速度项。
        return torch.clip(-self.root_states[:, 7], min=0.0, max=0.6)

    def _reward_all_feet_contact(self):
        # 长时间四脚全压在地上通常是站着蹭分，不是步态。
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > 5.0
        return torch.all(contact, dim=1).float() * self._upright_height_gate()

    def _reward_low_base_height(self):
        # 低于目标高度就持续扣分，低到接近趴地时扣得更重。
        return torch.square(torch.clamp(0.30 - self.root_states[:, 2], min=0.0, max=0.14)) / 0.0196

    def _reward_body_upright(self):
        # 直立时接近 1，趴地或大幅俯仰/横滚时接近 0。
        upright_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        height_gate = torch.clamp((self.root_states[:, 2] - 0.24) / 0.08, min=0.0, max=1.0)
        return torch.exp(-4.0 * upright_error) * height_gate

    def _reward_lateral_vel(self):
        # 平地直走阶段不希望靠侧滑获得速度，否则容易出现像转弯一样的步态。
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_world_lateral_vel(self):
        # 世界坐标横向速度越大，播放时越像绕圈或斜着滑。
        return torch.square(self.root_states[:, 8])

    def _reward_lateral_drift(self):
        # 直接压住相对出生点的横向漂移，和 success_rate 的直行判断保持一致。
        drift = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        return torch.square(torch.clip(drift, max=2.0))

    def _reward_yaw_rate(self):
        # 当前命令 yaw=0，显式压住偏航，减少前脚撑地后整机转圈摔倒。
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_yaw_drift(self):
        # 惩罚身体朝向偏离世界 X 方向，解决“能动但越走越转弯”的问题。
        quat = self.base_quat
        siny_cosp = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cosy_cosp = 1.0 - 2.0 * (torch.square(quat[:, 1]) + torch.square(quat[:, 2]))
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.square(yaw)

    def _reward_stand_motion(self):
        # 有前进命令时，身体几乎不动就惩罚，减少“站着刷 tracking”的情况。
        speed_error = torch.clip(self.commands[:, 0] - self.base_lin_vel[:, 0], min=0.0, max=1.0)
        return speed_error * (self.commands[:, 0] > 0.1)

    def _reward_gait_balance(self):
        # 连续接触强度比硬阈值更平滑，能更早区分“正在形成对角步态”和“乱踩”。
        # gait_feet_indices 固定为 FL, FR, RL, RR，不能依赖 URDF 返回的 body 顺序。
        contact = torch.clamp(self.contact_forces[:, self.gait_feet_indices, 2] / 80.0, min=0.0, max=1.0)
        fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]

        quat = self.base_quat
        siny_cosp = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cosy_cosp = 1.0 - 2.0 * (torch.square(quat[:, 1]) + torch.square(quat[:, 2]))
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        lateral_drift = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        gait_gate = torch.exp(-0.15 * torch.square(torch.clip(lateral_drift, max=3.0)) - 0.35 * torch.square(yaw))

        num_contacts = torch.sum(contact, dim=1)
        two_feet = torch.exp(-torch.square(num_contacts - 2.0) / 0.75)
        diagonal = fl * rr + fr * rl
        lateral_pair = fl * rl + fr * rr
        front_rear_pair = fl * fr + rl * rr
        support_spread = torch.minimum(fl + fr, rl + rr) * torch.minimum(fl + rl, fr + rr)

        gait = two_feet + 0.6 * diagonal + 0.2 * support_spread - 0.25 * lateral_pair - 0.15 * front_rear_pair
        return gait * gait_gate * self._upright_height_gate()

    def _reward_diagonal_clock_gait(self):
        contact = torch.clamp(self.contact_forces[:, self.gait_feet_indices, 2] / 80.0, min=0.0, max=1.0)
        contact_match = 1.0 - torch.abs(contact - self.desired_contact_states)
        contact_score = torch.mean(contact_match, dim=1)
        return contact_score * self._forward_motion_gate()

    def _reward_trot_joint_pose(self):
        # Dense diagonal-trot prior. Contact rewards are sparse because the policy
        # only sees them after a foot hits the floor; this pose reward gives an
        # immediate signal to lift the two swing legs and keep the other diagonal
        # pair closer to a quiet support posture.
        swing_center = 0.75
        swing_distance = torch.abs(torch.remainder(self.foot_phase - swing_center + 0.5, 1.0) - 0.5)
        swing = torch.clamp(1.0 - swing_distance / 0.25, min=0.0, max=1.0)
        stance = self.desired_contact_states

        current = self.dof_pos[:, self.leg_joint_indices]
        target = self.default_dof_pos[:, self.leg_joint_indices].repeat(self.num_envs, 1, 1)

        # During swing: pull the thigh slightly forward and fold the calf. During
        # stance: stay near default with a tiny extension, avoiding rear-leg jumps.
        target[:, :, 1] += -0.13 * swing + 0.03 * stance
        target[:, :, 2] += -0.18 * swing + 0.04 * stance

        # Keep abduction close to default; large hip swings are what make playback
        # look like turning or side dragging instead of walking.
        hip_error = torch.square(current[:, :, 0] - target[:, :, 0]) / 0.04
        leg_error = torch.square(current[:, :, 1] - target[:, :, 1]) / 0.055
        leg_error += torch.square(current[:, :, 2] - target[:, :, 2]) / 0.075
        pose_score = torch.exp(-torch.mean(hip_error + leg_error, dim=1))
        return pose_score * self._forward_motion_gate()

    def _reward_front_swing_pose(self):
        # Extra pressure on front-leg swing because current policies tend to push
        # with the rear legs while the front feet scrape the ground.
        front_swing = 1.0 - self.desired_contact_states[:, 0:2]
        front = self.dof_pos[:, self.leg_joint_indices[0:2]]
        target = self.default_dof_pos[:, self.leg_joint_indices[0:2]].repeat(self.num_envs, 1, 1)
        target[:, :, 1] -= 0.14 * front_swing
        target[:, :, 2] -= 0.20 * front_swing
        error = torch.square(front[:, :, 1] - target[:, :, 1]) + 0.8 * torch.square(front[:, :, 2] - target[:, :, 2])
        return torch.mean(torch.exp(-error / 0.045) * front_swing, dim=1) * self._forward_motion_gate()

    def _reward_tracking_contacts_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.gait_feet_indices, :], dim=-1)
        swing_contact = (1.0 - self.desired_contact_states) * (1.0 - torch.exp(-torch.square(foot_forces) / 100.0))
        return torch.mean(swing_contact, dim=1)

    def _reward_tracking_contacts_vel(self):
        foot_pos = self.rigid_body_states[:, self.gait_feet_indices, 0:3]
        foot_vel = torch.norm((foot_pos - self.last_foot_pos) / self.dt, dim=2)
        stance_slip = self.desired_contact_states * (1.0 - torch.exp(-torch.square(foot_vel) / 0.25))
        return torch.mean(stance_slip, dim=1)

    def _reward_foot_clearance_clock(self):
        swing_phase = 1.0 - self.desired_contact_states
        foot_z = self.rigid_body_states[:, self.gait_feet_indices, 2]
        # 目标抬脚高度收低，鼓励“迈步”而不是高抬腿后砸地。
        target_height = 0.045 + 0.04 * swing_phase
        height_error = torch.square(torch.clamp(target_height - foot_z, min=0.0, max=0.12))
        return torch.mean(torch.exp(-height_error / 0.0015) * swing_phase, dim=1) * self._forward_motion_gate()

    def _reward_vertical_motion(self):
        # 专门压制身体上下弹跳；跳着前进时这个值会明显变大。
        return torch.square(self.root_states[:, 9])

    def _reward_airborne(self):
        # 正常低速步行不应该四足同时离地或只剩极少支撑。
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > 5.0
        support_count = torch.sum(contact.float(), dim=1)
        return (support_count < 1.0).float() + 0.35 * (support_count < 2.0).float()

    def _reward_support_stability(self):
        # 奖励低速行走中的 2-3 足稳定支撑，并要求前后、左右都有支撑来源。
        contact = torch.clamp(self.contact_forces[:, self.gait_feet_indices, 2] / 60.0, min=0.0, max=1.0)
        fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
        support_count = torch.sum(contact, dim=1)
        count_score = torch.exp(-torch.square(support_count - 2.0) / 0.9)
        front_rear = torch.minimum(fl + fr, rl + rr)
        left_right = torch.minimum(fl + rl, fr + rr)
        return count_score * torch.clamp(front_rear, max=1.0) * torch.clamp(left_right, max=1.0) * self._forward_motion_gate()

    def _reward_foot_impact(self):
        # 惩罚落脚瞬间过大的垂直冲击，减少前足撑一下、整机弹出去的动作。
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > 5.0
        foot_vel_z = torch.abs((self.rigid_body_states[:, self.gait_feet_indices, 2] - self.last_foot_pos[:, :, 2]) / self.dt)
        return torch.mean(torch.square(torch.clamp(foot_vel_z - 0.35, min=0.0, max=2.0)) * contact.float(), dim=1)

    def _reward_rear_push_jump(self):
        # 后腿同时猛蹬且身体向上弹，是当前“跳着走”的典型漏洞。
        rear_force = torch.mean(torch.clamp(self.contact_forces[:, self.gait_feet_indices[2:4], 2] / 120.0, min=0.0, max=2.0), dim=1)
        upward_vel = torch.clamp(self.root_states[:, 9] - 0.12, min=0.0, max=1.0)
        return rear_force * upward_vel

    def _reward_front_support(self):
        # 防止只靠后腿推，前足蹭地或几乎不承担支撑。
        front_contact = torch.clamp(self.contact_forces[:, self.gait_feet_indices[0:2], 2] / 60.0, min=0.0, max=1.0)
        rear_contact = torch.clamp(self.contact_forces[:, self.gait_feet_indices[2:4], 2] / 60.0, min=0.0, max=1.0)
        balance_error = torch.square(torch.mean(front_contact, dim=1) - torch.mean(rear_contact, dim=1))
        return torch.exp(-balance_error / 0.08) * self._upright_height_gate()

    def _reward_base_ahead_of_support(self):
        """Penalize sending the body ahead before the feet build support."""
        foot_pos = self.rigid_body_states[:, self.gait_feet_indices, 0:3]
        contact = (self.contact_forces[:, self.gait_feet_indices, 2] > 5.0).float()
        contact_count = torch.sum(contact, dim=1)

        # When the robot is briefly airborne, use all feet as a conservative
        # fallback support estimate so the penalty still discourages lunging.
        all_feet_center_x = torch.mean(foot_pos[:, :, 0], dim=1)
        contact_center_x = torch.sum(foot_pos[:, :, 0] * contact, dim=1) / torch.clamp(contact_count, min=1.0)
        support_center_x = torch.where(contact_count > 0.5, contact_center_x, all_feet_center_x)

        base_ahead = torch.clamp(self.root_states[:, 0] - support_center_x - 0.03, min=0.0, max=0.40)
        return base_ahead * self._upright_height_gate()

    def _reward_front_swing_motion(self):
        """Reward actual front-foot lift and forward motion during swing."""
        front_pos = self.rigid_body_states[:, self.gait_feet_indices[0:2], 0:3]
        front_vel = (front_pos - self.last_foot_pos[:, 0:2, 0:3]) / self.dt
        front_swing = 1.0 - self.desired_contact_states[:, 0:2]
        contact = self.contact_forces[:, self.gait_feet_indices[0:2], 2] > 5.0

        forward_motion = torch.clamp((front_vel[:, :, 0] + 0.02) / 0.20, min=0.0, max=1.0)
        horizontal_motion = torch.clamp(torch.norm(front_vel[:, :, 0:2], dim=2) / 0.25, min=0.0, max=1.0)
        lift = torch.clamp((front_pos[:, :, 2] - 0.025) / 0.06, min=0.0, max=1.0)
        readiness = 0.55 * forward_motion + 0.25 * horizontal_motion + 0.20 * lift
        readiness = readiness * front_swing * (~contact).float()
        return torch.mean(readiness, dim=1) * self._forward_motion_gate()

    def _reward_rear_push_without_front_ready(self):
        """Discourage a rear-leg shove before either front leg is ready."""
        front_contact = torch.clamp(
            self.contact_forces[:, self.gait_feet_indices[0:2], 2] / 60.0,
            min=0.0,
            max=1.0,
        )
        front_pos = self.rigid_body_states[:, self.gait_feet_indices[0:2], 0:3]
        front_vel = (front_pos - self.last_foot_pos[:, 0:2, 0:3]) / self.dt
        front_swing = 1.0 - self.desired_contact_states[:, 0:2]
        front_lift = torch.clamp((front_pos[:, :, 2] - 0.025) / 0.06, min=0.0, max=1.0)
        front_forward_motion = torch.clamp((front_vel[:, :, 0] + 0.02) / 0.20, min=0.0, max=1.0)
        front_swing_ready = front_swing * (0.5 * front_lift + 0.5 * front_forward_motion)
        front_ready = torch.clamp(
            torch.maximum(torch.mean(front_contact, dim=1), torch.mean(front_swing_ready, dim=1)),
            min=0.0,
            max=1.0,
        )

        rear_contact = torch.clamp(
            self.contact_forces[:, self.gait_feet_indices[2:4], 2] / 100.0,
            min=0.0,
            max=1.5,
        )
        rear_action = torch.mean(torch.abs(self.actions[:, 6:12]), dim=1)
        rear_drive = torch.mean(rear_contact, dim=1) * (
            0.35 * torch.clamp(self.root_states[:, 7] / 0.22, min=0.0, max=1.5)
            + 0.65 * torch.clamp(rear_action / 0.45, min=0.0, max=1.5)
        )
        return rear_drive * (1.0 - front_ready) * self._upright_height_gate()

    def _reward_front_leg_activity(self):
        front_action = torch.mean(torch.abs(self.actions[:, 0:6] - self.last_actions[:, 0:6]), dim=1)
        rear_action = torch.mean(torch.abs(self.actions[:, 6:12] - self.last_actions[:, 6:12]), dim=1)
        return torch.exp(-torch.square(front_action - rear_action) / 0.04)

    def _reward_foot_clearance(self):
        foot_z = self.rigid_body_states[:, self.gait_feet_indices, 2]
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > 5.0
        target_height = 0.075
        swing_error = torch.square(torch.clamp(target_height - foot_z, min=0.0, max=0.12))
        return torch.mean(torch.exp(-swing_error / 0.0012) * (~contact), dim=1)

    def _reward_front_foot_clearance(self):
        foot_z = self.rigid_body_states[:, self.gait_feet_indices[0:2], 2]
        contact = self.contact_forces[:, self.gait_feet_indices[0:2], 2] > 5.0
        target_height = 0.085
        swing_error = torch.square(torch.clamp(target_height - foot_z, min=0.0, max=0.14))
        return torch.mean(torch.exp(-swing_error / 0.0010) * (~contact), dim=1)

    def _reward_rear_overstride(self):
        foot_x = self.rigid_body_states[:, self.gait_feet_indices[2:4], 0]
        base_x = self.root_states[:, 0].unsqueeze(1)
        return torch.mean(torch.square(torch.clamp(foot_x - base_x - 0.02, min=0.0, max=0.35)), dim=1)

    def _reward_rear_pair_swing(self):
        rear_contact = self.contact_forces[:, self.gait_feet_indices[2:4], 2] > 5.0
        both_rear_air = (~rear_contact[:, 0]) & (~rear_contact[:, 1])
        return both_rear_air.float()

    def _reward_front_slip(self):
        foot_pos = self.rigid_body_states[:, self.gait_feet_indices[0:2], 0:3]
        contact = self.contact_forces[:, self.gait_feet_indices[0:2], 2] > 5.0
        foot_vel_xy = (foot_pos[:, :, 0:2] - self.last_foot_pos[:, 0:2, 0:2]) / self.dt
        return torch.mean(torch.norm(foot_vel_xy, dim=2) * contact.float(), dim=1)
