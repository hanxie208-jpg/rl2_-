#!/usr/bin/env python3
"""Run a short, no-training audit of a registered Dog WTW-Lite task."""

from datetime import datetime
import os
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401 - Isaac Gym must load before torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - imports task registrations
from legged_gym.utils import get_args, task_registry


ALLOWED_TASKS = {
    "dog_wtw_lite_a",
    "dog_wtw_lite_b",
    "dog_wtw_lite_c",
    "dog_wtw_lite_c_mass137",
    "dog_wtw_lite_c_mass137_clearance3",
    "dog_wtw_lite_original17_contact2",
    "dog_wtw_lite_original23_contact2",
    "dog_wtw_lite_c_stable_2",
    "dog_wtw_lite_c_stable_3_raibert",
    "dog_wtw_lite_c_stable_4",
    "dog_wtw_lite_c_slow_010",
    "dog_wtw_lite_d",
}
PREFLIGHT_STEPS = 150
REPORT_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "WTW_LITE_PREFLIGHT_REPORT.md")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _observation_slices(env):
    result = {}
    start = 0
    for name, width in env.OBSERVATION_BLOCKS:
        result[name] = slice(start, start + width)
        start += width
    return result, start


def _urdf_facts(path, foot_names):
    root = ET.parse(path).getroot()
    masses = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in root.findall("link")
    }
    efforts = {
        joint.get("name"): float(joint.find("limit").get("effort"))
        for joint in root.findall("joint")
        if joint.get("type") in ("revolute", "continuous")
    }
    foot_collisions = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in foot_names:
            continue
        geometry = link.find("collision/geometry")
        shape = geometry[0] if geometry is not None and len(geometry) else None
        foot_collisions[name] = {
            "type": shape.tag if shape is not None else "missing",
            "radius": shape.get("radius") if shape is not None else None,
        }
    return masses, efforts, foot_collisions


def _reward_rows(env):
    rows = []
    for name in sorted(env.configured_reward_scales):
        configured = float(env.configured_reward_scales[name])
        if configured == 0.0:
            continue
        raw = env.latest_reward_raw.get(name)
        weighted = env.latest_reward_weighted.get(name)
        rows.append(
            (
                name,
                configured,
                float(env.reward_scales[name]),
                float(raw.item()) if raw is not None else None,
                float(weighted.item()) if weighted is not None else None,
            )
        )
    return rows


def main():
    args = get_args()
    failures = []
    warnings = []

    def check(condition, message):
        if not bool(condition):
            failures.append(message)

    if args.task not in ALLOWED_TASKS:
        raise SystemExit(
            f"Preflight only accepts {sorted(ALLOWED_TASKS)}, got {args.task!r}"
        )

    args.num_envs = min(args.num_envs or 4, 10)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    obs, privileged_obs = env.reset()

    slices, block_total = _observation_slices(env)
    check(env.num_dof == 12, f"Expected 12 DOFs, found {env.num_dof}")
    check(env.num_actions == 12, f"Expected 12 actions, found {env.num_actions}")
    check(env.cfg.commands.num_commands == 14, "Command dimension is not 14")
    check(env.num_obs == 63, f"Configured observation dimension is {env.num_obs}, not 63")
    check(block_total == env.num_obs, "Observation block widths do not sum to num_obs")
    check(tuple(obs.shape) == (env.num_envs, 63), f"Runtime obs shape is {tuple(obs.shape)}")
    check(privileged_obs is None, "Privileged observations are unexpectedly enabled")
    check(env.cfg.rewards.only_positive_rewards_ji22_style,
          "Official ji22 reward aggregation is not enabled")
    check(list(env.POLICY_DOF_NAMES) == [env.dof_names[i] for i in env.policy_dof_indices],
          "Policy-to-asset DOF name mapping is inconsistent")

    urdf_masses, urdf_efforts, foot_collisions = _urdf_facts(
        env.cfg.asset.file, env.FOOT_NAMES
    )
    actor_props = env.gym.get_actor_rigid_body_properties(
        env.envs[0], env.actor_handles[0]
    )
    actual_mass = sum(float(prop.mass) for prop in actor_props)
    urdf_mass = sum(urdf_masses.values())
    policy_effort = env.torque_limits[env.policy_dof_indices]
    expected_urdf_mass = (
        13.70708 if "mass137" in args.task else 21.71528
    )
    expected_effort = (
        17.0 if args.task == "dog_wtw_lite_original17_contact2"
        else 23.0 if args.task == "dog_wtw_lite_original23_contact2"
        else 22.0
    )
    check(abs(urdf_mass - expected_urdf_mass) < 1e-5,
          f"URDF total mass is {urdf_mass}, expected {expected_urdf_mass} kg")
    check(abs(actual_mass - urdf_mass) < 1e-4,
          f"PhysX mass {actual_mass} differs from URDF mass {urdf_mass}")
    check(all(abs(value - expected_effort) < 1e-6 for value in urdf_efforts.values()),
          f"URDF effort limits are not all {expected_effort} Nm: {urdf_efforts}")
    check(torch.allclose(policy_effort, torch.full_like(policy_effort, expected_effort)),
          f"PhysX effort limits are not all {expected_effort} Nm: {policy_effort.tolist()}")
    for foot_name in env.FOOT_NAMES:
        collision = foot_collisions.get(foot_name, {})
        check(collision.get("type") == "sphere",
              f"{foot_name} collision is not a sphere: {collision}")
        try:
            radius = float(collision.get("radius"))
        except (TypeError, ValueError):
            radius = float("nan")
        check(abs(radius - 0.02) < 1e-9,
              f"{foot_name} sphere radius is {collision.get('radius')}, not 0.02 m")

    contact_min = torch.full((4,), float("inf"), device=env.device)
    contact_max = torch.full((4,), float("-inf"), device=env.device)
    max_abs_torque = torch.zeros(4, device=env.device)
    torque_saturation_count = torch.zeros(4, device=env.device)
    torque_sample_count = 0
    base_height_sum = 0.0
    penalized_contact_count = 0.0
    reset_count = 0
    all_finite = True
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    for _ in range(PREFLIGHT_STEPS):
        obs, _, _, dones, _ = env.step(zero_actions)
        contact_min = torch.minimum(contact_min, env.desired_contact_states.min(dim=0).values)
        contact_max = torch.maximum(contact_max, env.desired_contact_states.max(dim=0).values)
        semantic_torques = env.torques[:, env.leg_dof_indices]
        max_abs_torque = torch.maximum(
            max_abs_torque, torch.abs(semantic_torques).amax(dim=(0, 2))
        )
        saturation_threshold = (
            float(env.cfg.diagnostics.torque_saturation_fraction)
            * policy_effort.view(4, 3).unsqueeze(0)
        )
        torque_saturation_count += (
            torch.abs(semantic_torques) >= saturation_threshold
        ).float().sum(dim=(0, 2))
        torque_sample_count += env.num_envs * 3
        base_height_sum += float(env.root_states[:, 2].mean().item())
        penalized_contact_count += float(
            (
                torch.norm(
                    env.contact_forces[:, env.penalised_contact_indices, :], dim=2
                ) > 0.1
            ).float().sum(dim=1).mean().item()
        )
        reset_count += int(dones.sum().item())
        all_finite = all_finite and bool(torch.isfinite(obs).all().item())

    check(all_finite, "Observation contains NaN or Inf during rollout")
    check(bool(torch.isfinite(env.rew_buf).all().item()),
          "Aggregate reward contains NaN or Inf")
    check(bool(torch.all(env.rew_buf >= 0.0).item()),
          "Ji22 aggregate reward became negative")
    check(bool(torch.all((contact_max - contact_min) > 0.25).item()),
          f"Desired contact states did not vary enough: min={contact_min.tolist()}, max={contact_max.tolist()}")
    check(bool(torch.all(max_abs_torque <= policy_effort.view(4, 3).max(dim=1).values + 1e-5).item()),
          f"Torque exceeded URDF effort limit: per-leg max={max_abs_torque.tolist()}")
    mean_base_height = base_height_sum / PREFLIGHT_STEPS
    mean_penalized_contacts = penalized_contact_count / PREFLIGHT_STEPS
    check(mean_base_height > 0.30,
          f"Mean base height is too low for the neutral pose: {mean_base_height}")
    check(mean_penalized_contacts < 0.10,
          f"Neutral pose has persistent non-foot contact: {mean_penalized_contacts} bodies/step")
    check(float(torch.max(torch.abs(env.desired_contact_states[:, 0] - env.desired_contact_states[:, 3])).item()) < 1e-5,
          "Trot diagonal FL/RR desired contacts are not synchronized")
    check(float(torch.max(torch.abs(env.desired_contact_states[:, 1] - env.desired_contact_states[:, 2])).item()) < 1e-5,
          "Trot diagonal FR/RL desired contacts are not synchronized")
    check(float(torch.mean(torch.abs(env.desired_contact_states[:, 0] - env.desired_contact_states[:, 1])).item()) > 0.25,
          "The two trot diagonal pairs are not separated")

    command_expected = env.commands * env.command_scale
    check(torch.allclose(obs[:, slices["commands"]], command_expected, atol=1e-5),
          "Command block is not at the declared observation slice")
    check(torch.allclose(obs[:, slices["clock_inputs"]], env.clock_inputs, atol=1e-5),
          "Clock block is not at the declared observation slice")

    saved_actions = env.actions.clone()
    probe = torch.linspace(-0.05, 0.05, env.num_actions, device=env.device)
    env.actions[:] = probe.unsqueeze(0)
    env.compute_observations()
    check(torch.allclose(env.obs_buf[:, slices["previous_action"]], env.actions, atol=1e-6),
          "Previous-action block is not at the declared observation slice")
    env.actions[:] = saved_actions
    env.compute_observations()

    observation_stats = env.observation_statistics()
    for item in observation_stats:
        check(not item["nan"] and not item["inf"],
              f"Observation block {item['name']} contains NaN or Inf")

    saturation_ratio = torque_saturation_count / torque_sample_count
    saturation_alert = saturation_ratio > float(
        env.cfg.diagnostics.rear_saturation_alert_ratio
    )
    if bool(torch.any(saturation_alert[2:4]).item()):
        warnings.append("REAR_TORQUE_SATURATION=TRUE during zero-action rollout")

    reward_rows = _reward_rows(env)
    report = [
        "# WTW-Lite Preflight Report",
        "",
        f"- Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- Task: `{args.task}`",
        f"- Environments: `{env.num_envs}`",
        f"- Zero-action policy steps: `{PREFLIGHT_STEPS}`",
        f"- Result: `{'FAIL' if failures else 'PASS'}`",
        f"- Episode resets observed: `{reset_count}`",
        "",
        "## Robot",
        "",
        f"- URDF: `{env.cfg.asset.file}`",
        f"- URDF total mass: `{urdf_mass:.8f} kg`",
        f"- PhysX actor total mass: `{actual_mass:.8f} kg`",
        f"- Asset DOF order: `{list(env.dof_names)}`",
        f"- Policy DOF order: `{list(env.POLICY_DOF_NAMES)}`",
        f"- Policy to asset indices: `{env.policy_dof_indices.tolist()}`",
        f"- Effort limits in policy order: `{policy_effort.tolist()} Nm`",
        f"- Foot collisions: `{foot_collisions}`",
        "",
        "## Control",
        "",
        f"- Type: `{env.cfg.control.control_type}`",
        f"- Kp: `{env.cfg.control.stiffness}`",
        f"- Kd: `{env.cfg.control.damping}`",
        f"- Action scale: `{env.cfg.control.action_scale} rad`",
        f"- Decimation: `{env.cfg.control.decimation}`",
        f"- Max abs torque FL/FR/RL/RR: `{max_abs_torque.tolist()} Nm`",
        f"- Torque saturation ratio FL/FR/RL/RR: `{saturation_ratio.tolist()}`",
        f"- Rear saturation marker: `{'TRUE' if bool(torch.any(saturation_alert[2:4]).item()) else 'FALSE'}`",
        f"- Mean base height: `{mean_base_height:.8f} m`",
        f"- Mean penalized contacts: `{mean_penalized_contacts:.8f} bodies/policy-step`",
        f"- Final aggregate reward mean: `{env.rew_buf.mean().item():.8f}`",
        "",
        "## Gait",
        "",
        f"- Command names: `{list(env.COMMAND_NAMES)}`",
        f"- Command row 0: `{env.commands[0].tolist()}`",
        f"- Desired contact minimum FL/FR/RL/RR: `{contact_min.tolist()}`",
        f"- Desired contact maximum FL/FR/RL/RR: `{contact_max.tolist()}`",
        f"- Final clock row 0: `{env.clock_inputs[0].tolist()}`",
        f"- Final desired contact row 0: `{env.desired_contact_states[0].tolist()}`",
        "",
        "## Observation",
        "",
        f"Runtime shape: `{tuple(env.obs_buf.shape)}`. Privileged observation: `{privileged_obs}`.",
        "",
        "| Block | Slice | Min | Max | Mean | Std | NaN | Inf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in observation_stats:
        report.append(
            "| {name} | `{slice}` | {min} | {max} | {mean} | {std} | {nan} | {inf} |".format(
                **{key: _fmt(value) for key, value in item.items()}
            )
        )

    report.extend(
        [
            "",
            "## Reward Snapshot",
            "",
            "Configured scale is before `dt`; runtime scale is after `dt`.",
            "The final reward uses official ji22 aggregation: positive sum times `exp(negative sum / 0.02)`.",
            "",
            "| Reward | Configured scale | Runtime scale | Raw mean | Weighted mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, configured, runtime, raw, weighted in reward_rows:
        report.append(
            f"| `{name}` | {_fmt(configured)} | {_fmt(runtime)} | "
            f"{_fmt(raw) if raw is not None else 'n/a'} | "
            f"{_fmt(weighted) if weighted is not None else 'n/a'} |"
        )

    report.extend(["", "## Checks", ""])
    if failures:
        report.extend(f"- FAIL: {message}" for message in failures)
    else:
        report.append("- All structural and runtime checks passed.")
    report.extend(f"- WARNING: {message}" for message in warnings)
    report.extend(
        [
            "",
            "This preflight is a wiring test, not evidence that a learned locomotion policy succeeds.",
            "",
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))

    print(f"[WTW_LITE_PREFLIGHT] {'FAIL' if failures else 'PASS'}")
    print(f"[WTW_LITE_PREFLIGHT] report={REPORT_PATH}")
    for message in failures:
        print(f"[WTW_LITE_PREFLIGHT] FAIL: {message}")
    for message in warnings:
        print(f"[WTW_LITE_PREFLIGHT] WARNING: {message}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
