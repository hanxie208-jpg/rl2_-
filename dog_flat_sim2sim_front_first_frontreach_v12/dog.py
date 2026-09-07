"""Rear-hip guard and diagnostics for V12."""

import torch

from legged_gym.envs.dog_flat_sim2sim_front_first_frontreach_v11.dog import (
    DogFrontReachV11Robot,
)


class DogFrontReachV12Robot(DogFrontReachV11Robot):
    """Penalize only excessive RL/RR hip abduction displacement."""

    rear_hip_deadband = 0.20

    def _init_buffers(self):
        super()._init_buffers()
        self.v12_diag_rear_hip_abs = torch.zeros(self.num_envs, device=self.device)
        self.v12_diag_rear_hip_excess = torch.zeros(
            self.num_envs, device=self.device
        )
        self.v12_diag_rear_hip_velocity = torch.zeros(
            self.num_envs, device=self.device
        )

    def _rear_hip_state(self):
        rear_hip_indices = self.leg_joint_indices[2:4, 0]
        offset = (
            self.dof_pos[:, rear_hip_indices]
            - self.default_dof_pos[:, rear_hip_indices]
        )
        velocity = self.dof_vel[:, rear_hip_indices]
        excess = torch.clamp(
            torch.abs(offset) - self.rear_hip_deadband,
            min=0.0,
        )
        return offset, excess, velocity

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        offset, excess, velocity = self._rear_hip_state()
        self.v12_diag_rear_hip_abs += torch.mean(torch.abs(offset), dim=1)
        self.v12_diag_rear_hip_excess += torch.mean(excess, dim=1)
        self.v12_diag_rear_hip_velocity += torch.mean(
            torch.abs(velocity), dim=1
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        steps = torch.clamp(self.v9_diag_steps[env_ids], min=1.0)
        rear_hip_abs = torch.mean(self.v12_diag_rear_hip_abs[env_ids] / steps)
        rear_hip_excess = torch.mean(
            self.v12_diag_rear_hip_excess[env_ids] / steps
        )
        rear_hip_velocity = torch.mean(
            self.v12_diag_rear_hip_velocity[env_ids] / steps
        )

        super().reset_idx(env_ids)
        episode = self.extras["episode"]
        episode["rear_hip_mean_abs_offset"] = rear_hip_abs
        episode["rear_hip_mean_excess"] = rear_hip_excess
        episode["rear_hip_mean_abs_velocity"] = rear_hip_velocity

        self.v12_diag_rear_hip_abs[env_ids] = 0.0
        self.v12_diag_rear_hip_excess[env_ids] = 0.0
        self.v12_diag_rear_hip_velocity[env_ids] = 0.0

    def _reward_rear_hip_twist(self):
        _, excess, _ = self._rear_hip_state()
        return torch.mean(torch.square(excess), dim=1) * self._gait_command_gate()
