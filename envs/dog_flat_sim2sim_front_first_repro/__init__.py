"""Frozen reproduction task for the front-first sim2sim dog experiment."""

from .dog import DogRobot
from .dog_flat_sim2sim_front_first_repro_config import (
    DogFlatSim2SimFrontFirstReproCfg,
    DogFlatSim2SimFrontFirstReproCfgPPO,
)

DogFlatSim2SimFrontFirstReproRobot = DogRobot

__all__ = [
    "DogFlatSim2SimFrontFirstReproRobot",
    "DogFlatSim2SimFrontFirstReproCfg",
    "DogFlatSim2SimFrontFirstReproCfgPPO",
]
