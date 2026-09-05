# Independent reproduction record

- Task: `dog_flat_sim2sim_front_first_repro`
- Source classes: `DogFlatSim2SimFrontFirstCfg`, `DogFlatSim2SimFrontFirstCfgPPO`
- Snapshot: `config_snapshot.json`
- Snapshot SHA-256: `c69b3eee75345ebea2d8c64adde7b085caa10bf4dd6c707baf6ff281825753c9`
- Source backup: `rollback_backups/20260905_213840_before_front_first_repro/`

Resolved values include 60 observations, 12 actions, 384 training environments,
P control with stiffness 32.0, damping 3.5, action scale 0.14, PPO actor/critic
`[128, 64, 32]`, learning rate `2e-5`, entropy coefficient `0.0`, and 1200
maximum iterations. The snapshot also retains all inherited terrain, command,
reward, noise, simulation, and runner fields.

Verification completed on 2026-09-05:

- `py_compile`: passed for the reproduction package, registration, train, and play scripts.
- Snapshot/source and snapshot/reproduction field comparisons: passed.
- Registry lookup: passed.
- Headless smoke training: `--num_envs 1 --max_iterations 1`, 24 timesteps completed; output in `logs/dog_flat_sim2sim_front_first_repro/Sep05_21-48-12_smoke/`.
- Headless smoke playback: 2 steps from `model_1.pt`, policy export completed.

Use the commands in `README.md` in this directory for full training and playback.
