"""Load the resolved front-first environment and PPO configuration snapshot.

The source task uses nested class inheritance.  This reproduction task loads
the fully resolved leaf values from ``config_snapshot.json`` instead, so later
edits to ``legged_gym.envs.dog.dog_config`` cannot silently change it.
"""

import json
from collections.abc import MutableMapping
from pathlib import Path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


_SNAPSHOT_PATH = Path(__file__).with_name("config_snapshot.json")


class _FrozenNode(MutableMapping):
    """Mapping with the attribute access expected by legged_gym configs."""

    __slots__ = ("_data",)

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        if key == "_data":
            object.__setattr__(self, key, value)
        else:
            self[key] = value


def _namespace(value):
    if isinstance(value, dict):
        return _FrozenNode({key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _load_section(section):
    with _SNAPSHOT_PATH.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    return _namespace(snapshot[section])


class _FrozenConfig:
    _snapshot_section = None

    def __init__(self):
        section = _load_section(self._snapshot_section)
        for key, value in section.items():
            setattr(self, key, value)


class DogFlatSim2SimFrontFirstReproCfg(_FrozenConfig, LeggedRobotCfg):
    """Resolved ``DogFlatSim2SimFrontFirstCfg`` values at snapshot time."""

    _snapshot_section = "env"


class DogFlatSim2SimFrontFirstReproCfgPPO(_FrozenConfig, LeggedRobotCfgPPO):
    """Resolved ``DogFlatSim2SimFrontFirstCfgPPO`` values at snapshot time."""

    _snapshot_section = "ppo"
