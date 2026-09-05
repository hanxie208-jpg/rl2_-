import xml.etree.ElementTree as ET

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw


class DogWTWLight(LeggedRobot):
    """Minimal WTW gait/reward layer on top of the Dog's PD controller."""

    COMMAND_NAMES = (
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_yaw",
        "body_height_cmd",
        "gait_frequency",
        "gait_phase",
        "gait_offset",
        "gait_bound",
        "gait_duration",
        "footswing_height",
        "body_pitch",
        "body_roll",
        "stance_width",
        "stance_length",
    )
    COMMAND = {name: index for index, name in enumerate(COMMAND_NAMES)}
    POLICY_DOF_NAMES = (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    OBSERVATION_BLOCKS = (
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("commands", 14),
        ("dof_pos_error", 12),
        ("dof_vel", 12),
        ("previous_action", 12),
        ("clock_inputs", 4),
    )
    GAIT_REWARD_NAMES = {
        "raibert_heuristic",
        "tracking_contacts_shaped_force",
        "tracking_contacts_shaped_vel",
        "feet_clearance_cmd_linear",
    }

    def _create_envs(self):
        super()._create_envs()

        missing_dofs = [name for name in self.POLICY_DOF_NAMES if name not in self.dof_names]
        if missing_dofs:
            raise RuntimeError(f"WTW-Lite missing Dog DOFs: {missing_dofs}")
        self.policy_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.POLICY_DOF_NAMES],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.leg_dof_indices = self.policy_dof_indices.view(4, 3)

        foot_indices = []
        for name in self.FOOT_NAMES:
            index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
            if index < 0:
                raise RuntimeError(f"WTW-Lite cannot find rigid body {name}")
            foot_indices.append(index)
        self.gait_feet_indices = torch.tensor(
            foot_indices, dtype=torch.long, device=self.device, requires_grad=False
        )

    def _get_noise_scale_vec(self, cfg):
        # WTW-Lite stage 1 fixes the robot and disables observation noise.
        return torch.zeros_like(self.obs_buf[0])

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )

        self.command_scale = torch.tensor(
            [2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 0.15, 0.3, 0.3, 1.0, 1.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_phase = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.warped_foot_phase = torch.zeros_like(self.foot_phase)
        self.clock_inputs = torch.zeros_like(self.foot_phase)
        self.desired_contact_states = torch.zeros_like(self.foot_phase)
        self.previous_foot_velocities = self._foot_velocities().clone()
        self.last_last_actions = torch.zeros_like(self.actions)

        self._init_diagnostic_buffers()
        self._print_asset_mapping()

    def _prepare_reward_function(self):
        self.configured_reward_scales = dict(self.reward_scales)
        super()._prepare_reward_function()
        self.episode_sums["total"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.latest_reward_raw = {}
        self.latest_reward_weighted = {}

    def _print_asset_mapping(self):
        print("[WTW_LITE_ASSET] asset DOF order:", list(self.dof_names))
        print("[WTW_LITE_ASSET] policy DOF order:", list(self.POLICY_DOF_NAMES))
        print("[WTW_LITE_ASSET] policy->asset indices:", self.policy_dof_indices.cpu().tolist())
        print("[WTW_LITE_ASSET] foot order:", list(self.FOOT_NAMES))
        print("[WTW_LITE_ASSET] foot indices:", self.gait_feet_indices.cpu().tolist())
        print("[WTW_LITE_ASSET] effort limits:", self.torque_limits.cpu().tolist())

    def step(self, actions):
        self.previous_foot_velocities[:] = self._foot_velocities()
        return super().step(actions)

    def post_physics_step(self):
        previous_previous_action = self.last_actions.clone()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super().post_physics_step()
        self.last_last_actions[:] = previous_previous_action
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.last_last_actions[reset_ids] = 0.0

    def _compute_torques(self, actions):
        asset_order_actions = torch.zeros_like(actions)
        asset_order_actions[:, self.policy_dof_indices] = actions
        return super()._compute_torques(asset_order_actions)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
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

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        diagnostic_values = self._episode_diagnostic_values(env_ids)
        super().reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self._refresh_gait_outputs(env_ids)
        self.last_last_actions[env_ids] = 0.0
        self.previous_foot_velocities[env_ids] = self._foot_velocities()[env_ids]
        self._clear_diagnostic_buffers(env_ids)
        self.extras["episode"].update(diagnostic_values)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0
        count = len(env_ids)
        device = self.device

        def sample(low, high):
            if low == high:
                return torch.full((count,), float(low), device=device)
            return low + (high - low) * torch.rand(count, device=device)

        ranges = self.cfg.commands.ranges
        c = self.COMMAND
        self.commands[env_ids, c["lin_vel_x"]] = sample(*ranges.lin_vel_x)
        self.commands[env_ids, c["lin_vel_y"]] = sample(*ranges.lin_vel_y)
        self.commands[env_ids, c["ang_vel_yaw"]] = sample(*ranges.ang_vel_yaw)
        for name in self.COMMAND_NAMES[3:]:
            self.commands[env_ids, c[name]] = float(getattr(self.cfg.commands, name))

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.gait_indices[:] = torch.remainder(
            self.gait_indices
            + self.dt * self.commands[:, self.COMMAND["gait_frequency"]],
            1.0,
        )
        self._refresh_gait_outputs()

    def _refresh_gait_outputs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        cmd = self.commands[env_ids]
        gait = self.gait_indices[env_ids]
        phase = cmd[:, self.COMMAND["gait_phase"]]
        offset = cmd[:, self.COMMAND["gait_offset"]]
        bound = cmd[:, self.COMMAND["gait_bound"]]
        raw_phase = torch.stack(
            (gait + phase + offset + bound,
             gait + offset,
             gait + bound,
             gait + phase),
            dim=1,
        )
        raw_phase = torch.remainder(raw_phase, 1.0)
        self.foot_phase[env_ids] = raw_phase

        duration = torch.clamp(
            cmd[:, self.COMMAND["gait_duration"]].unsqueeze(1), 0.01, 0.99
        )
        stance = raw_phase < duration
        warped = torch.where(
            stance,
            raw_phase * (0.5 / duration),
            0.5 + (raw_phase - duration) * (0.5 / (1.0 - duration)),
        )
        self.warped_foot_phase[env_ids] = warped
        self.clock_inputs[env_ids] = torch.sin(2.0 * torch.pi * warped)

        kappa = float(self.cfg.rewards.kappa_gait_probs)
        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device),
            torch.tensor(kappa, device=self.device),
        )
        contact_probability = (
            normal.cdf(warped) * (1.0 - normal.cdf(warped - 0.5))
            + normal.cdf(warped - 1.0) * (1.0 - normal.cdf(warped - 1.5))
        )
        self.desired_contact_states[env_ids] = contact_probability

    def compute_observations(self):
        dof_indices = self.policy_dof_indices
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands * self.command_scale,
                (self.dof_pos[:, dof_indices] - self.default_dof_pos[:, dof_indices])
                * self.obs_scales.dof_pos,
                self.dof_vel[:, dof_indices] * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs,
            ),
            dim=-1,
        )
        if self.obs_buf.shape != (self.num_envs, self.num_obs):
            raise RuntimeError(
                f"WTW-Lite observation shape {tuple(self.obs_buf.shape)} "
                f"does not match configured {(self.num_envs, self.num_obs)}"
            )

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        positive_reward = torch.zeros(self.num_envs, device=self.device)
        negative_reward = torch.zeros(self.num_envs, device=self.device)
        gait_reward = torch.zeros(self.num_envs, device=self.device)
        for index, reward_function in enumerate(self.reward_functions):
            name = self.reward_names[index]
            raw = reward_function()
            weighted = raw * self.reward_scales[name]
            self.rew_buf += weighted
            self.episode_sums[name] += weighted
            self.latest_reward_raw[name] = raw.detach().mean()
            self.latest_reward_weighted[name] = weighted.detach().mean()
            # Ji22-style aggregation must classify each environment separately.
            # A batch-wide sign test can mix unrelated environments when a reward
            # term has different signs across the rollout.
            positive_reward += torch.clamp(weighted, min=0.0)
            negative_reward += torch.clamp(weighted, max=0.0)
            if name in self.GAIT_REWARD_NAMES:
                gait_reward += weighted
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        elif self.cfg.rewards.only_positive_rewards_ji22_style:
            self.rew_buf[:] = positive_reward * torch.exp(
                negative_reward / float(self.cfg.rewards.sigma_rew_neg)
            )
        self.episode_sums["total"] += self.rew_buf
        if "termination" in self.reward_scales:
            raw = self._reward_termination()
            weighted = raw * self.reward_scales["termination"]
            self.rew_buf += weighted
            self.episode_sums["termination"] += weighted
        self._accumulate_diagnostics(gait_reward)
        self._maybe_print_runtime_diagnostics()

    def _foot_positions(self):
        return self.rigid_body_states[:, self.gait_feet_indices, 0:3]

    def _foot_velocities(self):
        return self.rigid_body_states[:, self.gait_feet_indices, 7:10]

    def _reward_action_smoothness_1(self):
        target_delta = (self.actions - self.last_actions) * self.cfg.control.action_scale
        return torch.sum(torch.square(target_delta), dim=1)

    def _reward_action_smoothness_2(self):
        target_accel = (
            self.actions - 2.0 * self.last_actions + self.last_last_actions
        ) * self.cfg.control.action_scale
        return torch.sum(torch.square(target_accel), dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(
            self.contact_forces[:, self.gait_feet_indices, :], dim=-1
        )
        reward = -(1.0 - self.desired_contact_states) * (
            1.0 - torch.exp(
                -torch.square(foot_forces) / float(self.cfg.rewards.gait_force_sigma)
            )
        )
        return torch.sum(reward, dim=1) / 4.0

    def _reward_tracking_contacts_shaped_vel(self):
        foot_speed = torch.norm(self._foot_velocities(), dim=2)
        reward = -self.desired_contact_states * (
            1.0 - torch.exp(
                -torch.square(foot_speed) / float(self.cfg.rewards.gait_vel_sigma)
            )
        )
        return torch.sum(reward, dim=1) / 4.0

    def _reward_feet_slip(self):
        threshold = float(self.cfg.rewards.contact_force_threshold)
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > threshold
        contact_filter = torch.logical_or(contact, self.wtw_last_contacts)
        self.wtw_last_contacts[:] = contact
        speed_xy_sq = torch.sum(torch.square(self._foot_velocities()[:, :, :2]), dim=2)
        return torch.sum(contact_filter.float() * speed_xy_sq, dim=1)

    def _reward_feet_impact_vel(self):
        threshold = float(self.cfg.rewards.contact_force_threshold)
        contact = self.contact_forces[:, self.gait_feet_indices, 2] > threshold
        downward_speed = torch.clamp(self.previous_foot_velocities[:, :, 2], max=0.0)
        return torch.sum(contact.float() * torch.square(downward_speed), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        swing_profile = 1.0 - torch.abs(
            1.0
            - 2.0 * torch.clamp(2.0 * self.foot_phase - 1.0, min=0.0, max=1.0)
        )
        target_height = (
            self.commands[:, self.COMMAND["footswing_height"]].unsqueeze(1)
            * swing_profile
            + float(self.cfg.rewards.foot_radius)
        )
        height_error = torch.square(target_height - self._foot_positions()[:, :, 2])
        return torch.sum(height_error * (1.0 - self.desired_contact_states), dim=1)

    def _reward_raibert_heuristic(self):
        translated = self._foot_positions() - self.root_states[:, :3].unsqueeze(1)
        feet_body = torch.zeros_like(translated)
        inverse_yaw = quat_conjugate(self.base_quat)
        for foot_index in range(4):
            feet_body[:, foot_index, :] = quat_apply_yaw(
                inverse_yaw, translated[:, foot_index, :]
            )

        stance_width = self.commands[:, self.COMMAND["stance_width"]].unsqueeze(1)
        stance_length = self.commands[:, self.COMMAND["stance_length"]].unsqueeze(1)
        desired_y = torch.cat(
            (stance_width / 2.0, -stance_width / 2.0,
             stance_width / 2.0, -stance_width / 2.0),
            dim=1,
        )
        desired_x = torch.cat(
            (stance_length / 2.0, stance_length / 2.0,
             -stance_length / 2.0, -stance_length / 2.0),
            dim=1,
        )

        phase = torch.abs(1.0 - 2.0 * self.foot_phase) - 0.5
        frequency = torch.clamp(
            self.commands[:, self.COMMAND["gait_frequency"]].unsqueeze(1), min=0.1
        )
        desired_x += phase * self.commands[:, self.COMMAND["lin_vel_x"]].unsqueeze(1) * (0.5 / frequency)
        yaw_velocity = self.commands[:, self.COMMAND["ang_vel_yaw"]].unsqueeze(1)
        desired_y_offset = phase * yaw_velocity * stance_length * 0.25 / frequency
        desired_y_offset[:, 2:4] *= -1.0
        desired_y += desired_y_offset

        desired_xy = torch.stack((desired_x, desired_y), dim=2)
        return torch.sum(torch.square(desired_xy - feet_body[:, :, :2]), dim=(1, 2))

    def _init_diagnostic_buffers(self):
        n = self.num_envs
        device = self.device
        self.wtw_last_contacts = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.diag_steps = torch.zeros(n, device=device)
        self.diag_sum_vx = torch.zeros(n, device=device)
        self.diag_sum_command_vx = torch.zeros(n, device=device)
        self.diag_sum_base_height = torch.zeros(n, device=device)
        self.diag_contact_steps = torch.zeros(n, 4, device=device)
        self.diag_abs_torque = torch.zeros(n, 4, device=device)
        self.diag_torque_samples = torch.zeros(n, 4, device=device)
        self.diag_torque_saturation = torch.zeros(n, 4, device=device)
        self.diag_current_air_time = torch.zeros(n, 4, device=device)
        self.diag_max_air_time = torch.zeros(n, 4, device=device)
        self.diag_swing_height_sum = torch.zeros(n, 4, device=device)
        self.diag_swing_samples = torch.zeros(n, 4, device=device)
        self.diag_gait_reward_sum = torch.zeros(n, device=device)
        self.diag_pair_sync_error = torch.zeros(n, 2, device=device)
        self.diag_diagonal_separation = torch.zeros(n, device=device)
        self.diag_start_x = self.root_states[:, 0].clone()

    def _clear_diagnostic_buffers(self, env_ids):
        for name in (
            "diag_steps", "diag_sum_vx", "diag_sum_command_vx",
            "diag_sum_base_height", "diag_gait_reward_sum",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "diag_contact_steps", "diag_abs_torque", "diag_torque_samples",
            "diag_torque_saturation", "diag_current_air_time",
            "diag_max_air_time", "diag_swing_height_sum", "diag_swing_samples",
            "diag_pair_sync_error", "diag_diagonal_separation",
        ):
            getattr(self, name)[env_ids] = 0.0
        self.wtw_last_contacts[env_ids] = False
        self.diag_start_x[env_ids] = self.root_states[env_ids, 0]

    def _accumulate_diagnostics(self, gait_reward):
        contact_threshold = float(self.cfg.rewards.contact_force_threshold)
        contacts = self.contact_forces[:, self.gait_feet_indices, 2] > contact_threshold
        semantic_torque = self.torques[:, self.leg_dof_indices]
        semantic_limits = self.torque_limits[self.leg_dof_indices].unsqueeze(0)
        saturation_fraction = float(self.cfg.diagnostics.torque_saturation_fraction)
        saturated = torch.abs(semantic_torque) >= saturation_fraction * semantic_limits

        self.diag_steps += 1.0
        self.diag_sum_vx += self.root_states[:, 7]
        self.diag_sum_command_vx += self.commands[:, self.COMMAND["lin_vel_x"]]
        self.diag_sum_base_height += self.root_states[:, 2]
        self.diag_contact_steps += contacts.float()
        self.diag_abs_torque += torch.sum(torch.abs(semantic_torque), dim=2)
        self.diag_torque_samples += 3.0
        self.diag_torque_saturation += torch.sum(saturated.float(), dim=2)
        self.diag_current_air_time = torch.where(
            contacts, torch.zeros_like(self.diag_current_air_time),
            self.diag_current_air_time + self.dt,
        )
        self.diag_max_air_time = torch.maximum(
            self.diag_max_air_time, self.diag_current_air_time
        )
        swing = self.desired_contact_states < 0.5
        self.diag_swing_height_sum += self._foot_positions()[:, :, 2] * swing.float()
        self.diag_swing_samples += swing.float()
        self.diag_gait_reward_sum += gait_reward
        pair_sync = torch.stack(
            (
                torch.abs(contacts[:, 0].float() - contacts[:, 3].float()),
                torch.abs(contacts[:, 1].float() - contacts[:, 2].float()),
            ),
            dim=1,
        )
        diagonal_contact = 0.5 * (
            contacts[:, 0].float() + contacts[:, 3].float()
        )
        opposite_diagonal_contact = 0.5 * (
            contacts[:, 1].float() + contacts[:, 2].float()
        )
        self.diag_pair_sync_error += pair_sync
        self.diag_diagonal_separation += torch.abs(
            diagonal_contact - opposite_diagonal_contact
        )

    def _episode_diagnostic_values(self, env_ids):
        if not hasattr(self, "diag_steps"):
            return {}
        steps = torch.clamp(self.diag_steps[env_ids], min=1.0)
        torque_samples = torch.clamp(self.diag_torque_samples[env_ids], min=1.0)
        swing_samples = torch.clamp(self.diag_swing_samples[env_ids], min=1.0)
        steps = torch.clamp(self.diag_steps[env_ids], min=1.0)
        contact_ratio = self.diag_contact_steps[env_ids] / steps.unsqueeze(1)
        torque_saturation = self.diag_torque_saturation[env_ids] / torque_samples
        rear_alert_threshold = float(self.cfg.diagnostics.rear_saturation_alert_ratio)
        rear_alert = torch.any(torque_saturation[:, 2:4] > rear_alert_threshold, dim=1)

        values = {
            "forward_velocity": torch.mean(self.diag_sum_vx[env_ids] / steps),
            "commanded_forward_velocity": torch.mean(self.diag_sum_command_vx[env_ids] / steps),
            "forward_distance": torch.mean(self.root_states[env_ids, 0] - self.diag_start_x[env_ids]),
            "episode_length": torch.mean(self.diag_steps[env_ids]),
            "base_height": torch.mean(self.diag_sum_base_height[env_ids] / steps),
            "gait_reward": torch.mean(self.diag_gait_reward_sum[env_ids] / steps),
            "foot_air_time": torch.mean(self.diag_max_air_time[env_ids]),
            "foot_clearance": torch.mean(self.diag_swing_height_sum[env_ids] / swing_samples),
            "FL_RR_sync_error": torch.mean(self.diag_pair_sync_error[env_ids, 0] / steps),
            "FR_RL_sync_error": torch.mean(self.diag_pair_sync_error[env_ids, 1] / steps),
            "diagonal_separation": torch.mean(self.diag_diagonal_separation[env_ids] / steps),
            "mean_abs_torque": torch.mean(self.diag_abs_torque[env_ids] / torque_samples),
            "torque_saturation_ratio": torch.mean(torque_saturation),
            "REAR_TORQUE_SATURATION": torch.mean(rear_alert.float()),
        }
        for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
            values[f"{leg}_contact_ratio"] = torch.mean(contact_ratio[:, leg_index])
            values[f"{leg}_torque_saturation_ratio"] = torch.mean(
                torque_saturation[:, leg_index]
            )
        return values

    def _maybe_print_runtime_diagnostics(self):
        interval = int(self.cfg.diagnostics.print_interval_steps)
        if interval <= 0 or self.common_step_counter == 0:
            return
        if self.common_step_counter % interval != 0:
            return
        semantic_torque = self.torques[:, self.leg_dof_indices]
        semantic_limits = self.torque_limits[self.leg_dof_indices].unsqueeze(0)
        saturated = torch.abs(semantic_torque) >= (
            float(self.cfg.diagnostics.torque_saturation_fraction) * semantic_limits
        )
        saturation = saturated.float().mean(dim=(0, 2))
        rear_alert = torch.any(
            saturation[2:4] > float(self.cfg.diagnostics.rear_saturation_alert_ratio)
        ).item()
        steps = torch.clamp(self.diag_steps, min=1.0)
        contact_ratio = torch.mean(
            self.diag_contact_steps / steps.unsqueeze(1), dim=0
        )
        swing_samples = torch.clamp(self.diag_swing_samples, min=1.0)
        clearance = torch.mean(
            self.diag_swing_height_sum / swing_samples, dim=0
        )
        pair_sync = torch.mean(
            self.diag_pair_sync_error / steps.unsqueeze(1),
            dim=0,
        )
        diagonal_separation = torch.mean(self.diag_diagonal_separation / steps)
        print(
            "[WTW_LITE_RUNTIME]",
            f"step={self.common_step_counter}",
            f"vx={self.root_states[:, 7].mean().item():.4f}",
            f"cmd_vx={self.commands[:, 0].mean().item():.4f}",
            f"distance={(self.root_states[:, 0] - self.diag_start_x).mean().item():.4f}",
            f"height={self.root_states[:, 2].mean().item():.4f}",
            f"contact_FL_FR_RL_RR={contact_ratio.cpu().tolist()}",
            f"max_air_time={self.diag_max_air_time.mean().item():.4f}",
            f"clearance_FL_FR_RL_RR={clearance.cpu().tolist()}",
            f"diag_sync_FL_RR_FR_RL={pair_sync.cpu().tolist()}",
            f"diag_separation={diagonal_separation.item():.4f}",
            f"gait_reward={(self.diag_gait_reward_sum / steps).mean().item():.6f}",
            f"mean_abs_torque={torch.abs(semantic_torque).mean().item():.4f}",
            f"max_abs_torque={torch.abs(semantic_torque).max().item():.4f}",
            f"sat_FL_FR_RL_RR={saturation.cpu().tolist()}",
            f"REAR_TORQUE_SATURATION={rear_alert}",
        )

    def observation_statistics(self):
        stats = []
        start = 0
        for name, width in self.OBSERVATION_BLOCKS:
            block = self.obs_buf[:, start:start + width]
            stats.append(
                {
                    "name": name,
                    "slice": f"{start}:{start + width}",
                    "min": block.min().item(),
                    "max": block.max().item(),
                    "mean": block.mean().item(),
                    "std": block.std(unbiased=False).item(),
                    "nan": bool(torch.isnan(block).any().item()),
                    "inf": bool(torch.isinf(block).any().item()),
                }
            )
            start += width
        return stats

    def urdf_facts(self):
        root = ET.parse(self.cfg.asset.file).getroot()
        total_mass = sum(float(m.get("value")) for m in root.findall(".//mass"))
        foot_collisions = {}
        for link in root.findall("link"):
            name = link.get("name")
            if name in self.FOOT_NAMES:
                geometry = link.find("collision/geometry")
                child = geometry[0]
                foot_collisions[name] = {
                    "type": child.tag,
                    "radius": child.get("radius"),
                }
        return {"total_mass": total_mass, "foot_collisions": foot_collisions}
