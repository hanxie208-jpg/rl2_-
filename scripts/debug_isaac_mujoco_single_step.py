#!/usr/bin/env python3
"""Direct Isaac-vs-MuJoCo no-contact single-physics-step dynamics check."""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymtorch

import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_vec(row, sim, field, joint_names):
    return np.asarray([float(row[f"{sim}_{name}_{field}"]) for name in joint_names], dtype=np.float64)


def make_isaac_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.pos[2] = 0.8
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    return env


def set_isaac_state(env, row, csv_joint_names, base_z):
    device = env.device
    env.root_states[0, :] = 0.0
    env.root_states[0, 0:3] = torch.tensor([0.0, 0.0, base_z], device=device)
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    env.root_states[0, 7:13] = 0.0

    q_by_name = {name: float(row[f"isaac_{name}_q"]) for name in csv_joint_names}
    qdot_by_name = {name: float(row[f"isaac_{name}_qdot"]) for name in csv_joint_names}
    for i, name in enumerate(env.dof_names):
        env.dof_pos[0, i] = q_by_name[name]
        env.dof_vel[0, i] = qdot_by_name[name]

    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)


def isaac_single_step(env, row, tau_row, csv_joint_names, base_z):
    set_isaac_state(env, row, csv_joint_names, base_z)
    qdot0 = env.dof_vel[0].detach().clone()
    q0 = env.dof_pos[0].detach().clone()

    tau_by_name = {name: float(tau_row[f"isaac_{name}_torque"]) for name in csv_joint_names}
    tau = torch.zeros_like(env.torques)
    for i, name in enumerate(env.dof_names):
        tau[0, i] = tau_by_name[name]

    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
    env.gym.simulate(env.sim)
    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    qdot1 = env.dof_vel[0].detach().clone()
    q1 = env.dof_pos[0].detach().clone()
    qdd = (qdot1 - qdot0) / env.sim_params.dt
    contacts = env.contact_forces[0].detach().clone()
    return q0, qdot0, q1, qdot1, qdd, tau[0].detach().clone(), contacts


def build_mujoco(cfg_path: Path):
    root = cfg_path.parents[3]
    deploy = load_module(root / "deploy/deploy_mujoco/deploy_mujoco.py", "deploy_mujoco")
    debug = load_module(root / "deploy/deploy_mujoco/debug_pd_open_loop.py", "debug_pd_open_loop")
    import yaml
    import mujoco

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = debug.build_model(deploy, cfg, None)
    joint_names = list(cfg["joint_order"])
    qpos_adrs, dof_adrs = debug.joint_addresses(model, joint_names)
    model.dof_armature[dof_adrs] = float(cfg.get("dynamics", {}).get("armature", 0.0))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return mujoco, cfg, model, joint_names, qpos_adrs, dof_adrs


def mujoco_single_step(mujoco, cfg, model, row, tau_row, joint_names, qpos_adrs, dof_adrs, base_z):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adrs] = row_vec(row, "isaac", "q", joint_names)
    data.qvel[dof_adrs] = row_vec(row, "isaac", "qdot", joint_names)
    tau = row_vec(tau_row, "isaac", "torque", joint_names)
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_adrs] = tau
    mujoco.mj_forward(model, data)
    qacc = data.qacc[dof_adrs].copy()
    qfrc_applied = data.qfrc_applied[dof_adrs].copy()
    qfrc_bias = data.qfrc_bias[dof_adrs].copy()
    mujoco.mj_step(model, data)
    return data.qpos[qpos_adrs].copy(), data.qvel[dof_adrs].copy(), qacc, qfrc_applied, qfrc_bias


def main():
    args = get_args()
    root = Path(__file__).resolve().parents[2]
    traj_path = root / "logs/flat_dog/exported/validation_runs/rr_isaac_mujoco_compare_step50.csv"
    cfg_path = root / "deploy/deploy_mujoco/configs/dog_step50_60d.yaml"
    rows = read_rows(traj_path)
    csv_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    base_z = 0.8
    steps = [3, 4, 5, 6]

    env = make_isaac_env(args)
    mujoco, cfg, model, mj_joint_names, qpos_adrs, dof_adrs = build_mujoco(cfg_path)

    out_dir = root / "logs/flat_dog/exported/validation_runs" / f"isaac_mujoco_single_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "single_step.csv"

    fieldnames = [
        "transition", "joint",
        "tau", "isaac_qdd", "mujoco_qacc", "qacc_err",
        "isaac_qdot_next", "mujoco_qdot_next", "qdot_next_err",
        "isaac_q_next", "mujoco_q_next", "q_next_err",
        "mujoco_qfrc_applied", "mujoco_qfrc_bias",
    ]
    records = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            row = rows[step]
            tau_row = rows[step + 1]
            q0_i, qd0_i, q1_i, qd1_i, qdd_i, tau_i, contacts = isaac_single_step(env, row, tau_row, csv_joint_names, base_z)
            q1_m, qd1_m, qacc_m, qfrc_m, bias_m = mujoco_single_step(
                mujoco, cfg, model, row, tau_row, mj_joint_names, qpos_adrs, dof_adrs, base_z
            )
            isaac_by_name = {name: i for i, name in enumerate(env.dof_names)}
            for mj_idx, name in enumerate(mj_joint_names):
                i = isaac_by_name[name]
                rec = {
                    "transition": f"{step}->{step+1}",
                    "joint": name,
                    "tau": float(tau_i[i].cpu()),
                    "isaac_qdd": float(qdd_i[i].cpu()),
                    "mujoco_qacc": float(qacc_m[mj_idx]),
                    "qacc_err": float(qacc_m[mj_idx] - qdd_i[i].cpu().numpy()),
                    "isaac_qdot_next": float(qd1_i[i].cpu()),
                    "mujoco_qdot_next": float(qd1_m[mj_idx]),
                    "qdot_next_err": float(qd1_m[mj_idx] - qd1_i[i].cpu().numpy()),
                    "isaac_q_next": float(q1_i[i].cpu()),
                    "mujoco_q_next": float(q1_m[mj_idx]),
                    "q_next_err": float(q1_m[mj_idx] - q1_i[i].cpu().numpy()),
                    "mujoco_qfrc_applied": float(qfrc_m[mj_idx]),
                    "mujoco_qfrc_bias": float(bias_m[mj_idx]),
                }
                writer.writerow(rec)
                records.append(rec)

    summary = {
        "out_dir": str(out_dir),
        "csv": str(out_csv),
        "base_z": base_z,
        "isaac_dt": float(env.sim_params.dt),
        "mujoco_dt": float(cfg["sim_dt"]),
        "isaac_dof_names": list(env.dof_names),
        "mujoco_joint_names": mj_joint_names,
        "max_contact_force_norm": float(torch.linalg.norm(env.contact_forces[0], dim=1).max().cpu()),
        "first_qacc_err_gt_10": next((r for r in records if abs(r["qacc_err"]) > 10.0), None),
        "first_qdot_err_gt_1e-3": next((r for r in records if abs(r["qdot_next_err"]) > 1e-3), None),
        "max_abs_qacc_err_by_joint": {
            name: max(abs(r["qacc_err"]) for r in records if r["joint"] == name)
            for name in mj_joint_names
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
