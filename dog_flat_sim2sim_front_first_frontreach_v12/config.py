"""Conservative V11 refinement with a rear-hip twist penalty."""

from legged_gym.envs.dog_flat_sim2sim_front_first_frontreach_v11.config import (
    DogFrontReachV11Cfg,
    DogFrontReachV11CfgPPO,
)


class DogFrontReachV12Cfg(DogFrontReachV11Cfg):
    def __init__(self):
        super().__init__()

        # The reward has a 0.20 rad dead band, so ordinary hip motion remains
        # free while the large rear-hip folding seen in V11 is discouraged.
        self.rewards.scales.rear_hip_twist = -4.0


class DogFrontReachV12CfgPPO(DogFrontReachV11CfgPPO):
    def __init__(self):
        super().__init__()
        self.runner.run_name = "front_reach_v12_rear_hip_guard_from_v11_475"
        self.runner.max_iterations = 300
        self.runner.save_interval = 25

        # Applied after checkpoint loading by train_frontreach_exploration.py.
        # V11 ended with std around 1.25; low-noise refinement should preserve
        # its clock phase while allowing the new hip penalty to reshape motion.
        self.policy.init_noise_std = 0.12
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.entropy_coef = 0.0
        self.algorithm.schedule = "fixed"
