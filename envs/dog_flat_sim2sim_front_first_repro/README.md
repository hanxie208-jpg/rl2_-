# Dog front-first sim2sim reproduction

This task is registered as `dog_flat_sim2sim_front_first_repro`. Its environment
and PPO settings are loaded from `config_snapshot.json`, which contains the
fully resolved values of `DogFlatSim2SimFrontFirstCfg` and
`DogFlatSim2SimFrontFirstCfgPPO` captured from the source hashes recorded in the
snapshot. The robot implementation is copied into this package so the task
does not import the mutable `legged_gym.envs.dog.dog` module.

Run from `rl_stack/legged_gym` with the `leggedgym` conda environment:

```bash
conda activate leggedgym
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Train from the frozen configuration. The explicit experiment name keeps logs isolated.
python legged_gym/scripts/train.py \
  --task dog_flat_sim2sim_front_first_repro \
  --experiment_name dog_flat_sim2sim_front_first_repro \
  --run_name frozen_front_first \
  --headless

# Play the newest checkpoint from that experiment (use --load_run for a specific run).
python legged_gym/scripts/play.py \
  --task dog_flat_sim2sim_front_first_repro \
  --experiment_name dog_flat_sim2sim_front_first_repro
```

To select a concrete run/checkpoint, pass its directory name and integer
checkpoint, for example `--load_run Sep05_21-48-12_smoke --checkpoint 1`.

For a smoke run, append `--num_envs 1 --max_iterations 1` to the training
command. For a headless playback check, append `--headless` and set
`PLAY_MAX_STEPS=50 PLAY_DISABLE_PLOT=1`.
