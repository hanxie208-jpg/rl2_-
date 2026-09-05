# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
# from .base.legged_robot import LeggedRobot
# from .anymal_c.anymal import Anymal
# from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
# from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
# from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
# from .cassie.cassie import Cassie
# from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
# from .dog.dog import DogRobot, DogSimpleRobot
# from .dog.dog_config import (
#     DogFlatCfg,
#     DogFlatCfgPPO,
#     DogFlatClockCfg,
#     DogFlatClockCfgPPO,
#     DogFlatSim2SimCfg,
#     DogFlatSim2SimCfgPPO,
#     DogFlatSim2SimFrontFirstCfg,
#     DogFlatSim2SimFrontFirstCfgPPO,
#     DogFlatSim2SimFrontFirstHighBaseCfg,
#     DogFlatSim2SimFrontFirstHighBaseCfgPPO,
#     DogFlatSim2SimFrontFirstPose080Cfg,
#     DogFlatSim2SimFrontFirstPose080CfgPPO,
#     DogFlatSim2SimDeployMatchCfg,
#     DogFlatSim2SimDeployMatchCfgPPO,
#     DogFlatSim2SimRobustCfg,
#     DogFlatSim2SimRobustCfgPPO,
#     DogFriendCfg,
#     DogFriendCfgPPO,
#     DogSimpleCfg,
#     DogSimpleCfgPPO,
# )
# from .dog_wtw.dog_wtw_config import DogWtwFlatCfg, DogWtwFlatCfgPPO
# from .dog_flat_sim2sim_front_first_repro import (
#     DogFlatSim2SimFrontFirstReproRobot,
#     DogFlatSim2SimFrontFirstReproCfg,
#     DogFlatSim2SimFrontFirstReproCfgPPO,
# )
# from .dog_wtw_lite import (
#     DogWTWLight,
#     DogWTWLightACfg,
#     DogWTWLightAPPOCfg,
#     DogWTWLightBCfg,
#     DogWTWLightBPPOCfg,
#     DogWTWLightCCfg,
#     DogWTWLightCPPOCfg,
#     DogWTWLightCMass137Cfg,
#     DogWTWLightCMass137PPOCfg,
#     DogWTWLightCMass137Clearance3Cfg,
#     DogWTWLightCMass137Clearance3PPOCfg,
#     DogWTWLightOriginal17Contact2Cfg,
#     DogWTWLightOriginal17Contact2PPOCfg,
#     DogWTWLightOriginal23Contact2Cfg,
#     DogWTWLightOriginal23Contact2PPOCfg,
#     DogWTWLightCStable2Cfg,
#     DogWTWLightCStable2PPOCfg,
#     DogWTWLightCStable3RaibertCfg,
#     DogWTWLightCStable3RaibertPPOCfg,
#     DogWTWLightCStable4Cfg,
#     DogWTWLightCStable4PPOCfg,
#     DogWTWLightCSlow010Cfg,
#     DogWTWLightCSlow010PPOCfg,
#     DogWTWLightDCfg,
#     DogWTWLightDPPOCfg,
# )
# from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO


# import os

# from legged_gym.utils.task_registry import task_registry

# task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
# task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
# task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
# task_registry.register( "dog_flat", DogRobot, DogFlatCfg(), DogFlatCfgPPO() )
# task_registry.register( "dog_flat_clock", DogRobot, DogFlatClockCfg(), DogFlatClockCfgPPO() )
# task_registry.register( "dog_flat_sim2sim", DogRobot, DogFlatSim2SimCfg(), DogFlatSim2SimCfgPPO() )
# task_registry.register( "dog_flat_sim2sim_front_first", DogRobot, DogFlatSim2SimFrontFirstCfg(), DogFlatSim2SimFrontFirstCfgPPO() )
# task_registry.register(
#     "dog_flat_sim2sim_front_first_repro",
#     DogFlatSim2SimFrontFirstReproRobot,
#     DogFlatSim2SimFrontFirstReproCfg(),
#     DogFlatSim2SimFrontFirstReproCfgPPO(),
# )
# task_registry.register( "dog_flat_sim2sim_front_first_hz040", DogRobot, DogFlatSim2SimFrontFirstHighBaseCfg(), DogFlatSim2SimFrontFirstHighBaseCfgPPO() )
# task_registry.register( "dog_flat_sim2sim_front_first_pose080", DogRobot, DogFlatSim2SimFrontFirstPose080Cfg(), DogFlatSim2SimFrontFirstPose080CfgPPO() )
# task_registry.register( "dog_flat_sim2sim_deploy_match", DogRobot, DogFlatSim2SimDeployMatchCfg(), DogFlatSim2SimDeployMatchCfgPPO() )
# task_registry.register( "dog_flat_sim2sim_robust", DogRobot, DogFlatSim2SimRobustCfg(), DogFlatSim2SimRobustCfgPPO() )
# task_registry.register( "dog_friend", LeggedRobot, DogFriendCfg(), DogFriendCfgPPO() )
# task_registry.register( "dog_simple", DogSimpleRobot, DogSimpleCfg(), DogSimpleCfgPPO() )
# task_registry.register( "dog_wtw_flat", DogRobot, DogWtwFlatCfg(), DogWtwFlatCfgPPO() )
# task_registry.register("dog_wtw_lite_a", DogWTWLight, DogWTWLightACfg(), DogWTWLightAPPOCfg())
# task_registry.register("dog_wtw_lite_b", DogWTWLight, DogWTWLightBCfg(), DogWTWLightBPPOCfg())
# task_registry.register("dog_wtw_lite_c", DogWTWLight, DogWTWLightCCfg(), DogWTWLightCPPOCfg())
# task_registry.register("dog_wtw_lite_c_mass137", DogWTWLight, DogWTWLightCMass137Cfg(), DogWTWLightCMass137PPOCfg())
# task_registry.register("dog_wtw_lite_c_mass137_clearance3", DogWTWLight, DogWTWLightCMass137Clearance3Cfg(), DogWTWLightCMass137Clearance3PPOCfg())
# task_registry.register("dog_wtw_lite_original17_contact2", DogWTWLight, DogWTWLightOriginal17Contact2Cfg(), DogWTWLightOriginal17Contact2PPOCfg())
# task_registry.register("dog_wtw_lite_original23_contact2", DogWTWLight, DogWTWLightOriginal23Contact2Cfg(), DogWTWLightOriginal23Contact2PPOCfg())
# task_registry.register("dog_wtw_lite_c_stable_2", DogWTWLight, DogWTWLightCStable2Cfg(), DogWTWLightCStable2PPOCfg())
# task_registry.register("dog_wtw_lite_c_stable_3_raibert", DogWTWLight, DogWTWLightCStable3RaibertCfg(), DogWTWLightCStable3RaibertPPOCfg())
# task_registry.register("dog_wtw_lite_c_stable_4", DogWTWLight, DogWTWLightCStable4Cfg(), DogWTWLightCStable4PPOCfg())
# task_registry.register("dog_wtw_lite_c_slow_010", DogWTWLight, DogWTWLightCSlow010Cfg(), DogWTWLightCSlow010PPOCfg())
# task_registry.register("dog_wtw_lite_d", DogWTWLight, DogWTWLightDCfg(), DogWTWLightDPPOCfg())


# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .dog.dog import DogRobot, DogSimpleRobot
from .dog.dog_config import (
    DogFlatCfg,
    DogFlatCfgPPO,
    DogFlatClockCfg,
    DogFlatClockCfgPPO,
    DogFlatSim2SimCfg,
    DogFlatSim2SimCfgPPO,
    DogFlatSim2SimFrontFirstCfg,
    DogFlatSim2SimFrontFirstCfgPPO,
    DogFlatSim2SimFrontFirstHighBaseCfg,
    DogFlatSim2SimFrontFirstHighBaseCfgPPO,
    DogFlatSim2SimFrontFirstPose080Cfg,
    DogFlatSim2SimFrontFirstPose080CfgPPO,
    DogFlatSim2SimDeployMatchCfg,
    DogFlatSim2SimDeployMatchCfgPPO,
    DogFlatSim2SimRobustCfg,
    DogFlatSim2SimRobustCfgPPO,
    DogFriendCfg,
    DogFriendCfgPPO,
    DogSimpleCfg,
    DogSimpleCfgPPO,
    DogCleanCfg,           # <-- 新增：你整理的干净配置
    DogCleanCfgPPO,        # <-- 新增：对应 PPO 配置
)
from .dog_wtw.dog_wtw_config import DogWtwFlatCfg, DogWtwFlatCfgPPO
from .dog_flat_sim2sim_front_first_repro import (
    DogFlatSim2SimFrontFirstReproRobot,
    DogFlatSim2SimFrontFirstReproCfg,
    DogFlatSim2SimFrontFirstReproCfgPPO,
)
from .dog_wtw_lite import (
    DogWTWLight,
    DogWTWLightACfg,
    DogWTWLightAPPOCfg,
    DogWTWLightBCfg,
    DogWTWLightBPPOCfg,
    DogWTWLightCCfg,
    DogWTWLightCPPOCfg,
    DogWTWLightCMass137Cfg,
    DogWTWLightCMass137PPOCfg,
    DogWTWLightCMass137Clearance3Cfg,
    DogWTWLightCMass137Clearance3PPOCfg,
    DogWTWLightOriginal17Contact2Cfg,
    DogWTWLightOriginal17Contact2PPOCfg,
    DogWTWLightOriginal23Contact2Cfg,
    DogWTWLightOriginal23Contact2PPOCfg,
    DogWTWLightCStable2Cfg,
    DogWTWLightCStable2PPOCfg,
    DogWTWLightCStable3RaibertCfg,
    DogWTWLightCStable3RaibertPPOCfg,
    DogWTWLightCStable4Cfg,
    DogWTWLightCStable4PPOCfg,
    DogWTWLightCSlow010Cfg,
    DogWTWLightCSlow010PPOCfg,
    DogWTWLightDCfg,
    DogWTWLightDPPOCfg,
)
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "dog_flat", DogRobot, DogFlatCfg(), DogFlatCfgPPO() )
task_registry.register( "dog_flat_clock", DogRobot, DogFlatClockCfg(), DogFlatClockCfgPPO() )
task_registry.register( "dog_flat_sim2sim", DogRobot, DogFlatSim2SimCfg(), DogFlatSim2SimCfgPPO() )

# ========== 修改这一行：指向你干净的 DogCleanCfg ==========
task_registry.register( "dog_flat_sim2sim_front_first", DogRobot, DogCleanCfg(), DogCleanCfgPPO() )
# ==========================================================

task_registry.register(
    "dog_flat_sim2sim_front_first_repro",
    DogFlatSim2SimFrontFirstReproRobot,
    DogFlatSim2SimFrontFirstReproCfg(),
    DogFlatSim2SimFrontFirstReproCfgPPO(),
)
task_registry.register( "dog_flat_sim2sim_front_first_hz040", DogRobot, DogFlatSim2SimFrontFirstHighBaseCfg(), DogFlatSim2SimFrontFirstHighBaseCfgPPO() )
task_registry.register( "dog_flat_sim2sim_front_first_pose080", DogRobot, DogFlatSim2SimFrontFirstPose080Cfg(), DogFlatSim2SimFrontFirstPose080CfgPPO() )
task_registry.register( "dog_flat_sim2sim_deploy_match", DogRobot, DogFlatSim2SimDeployMatchCfg(), DogFlatSim2SimDeployMatchCfgPPO() )
task_registry.register( "dog_flat_sim2sim_robust", DogRobot, DogFlatSim2SimRobustCfg(), DogFlatSim2SimRobustCfgPPO() )
task_registry.register( "dog_friend", LeggedRobot, DogFriendCfg(), DogFriendCfgPPO() )
task_registry.register( "dog_simple", DogSimpleRobot, DogSimpleCfg(), DogSimpleCfgPPO() )
task_registry.register( "dog_wtw_flat", DogRobot, DogWtwFlatCfg(), DogWtwFlatCfgPPO() )
task_registry.register("dog_wtw_lite_a", DogWTWLight, DogWTWLightACfg(), DogWTWLightAPPOCfg())
task_registry.register("dog_wtw_lite_b", DogWTWLight, DogWTWLightBCfg(), DogWTWLightBPPOCfg())
task_registry.register("dog_wtw_lite_c", DogWTWLight, DogWTWLightCCfg(), DogWTWLightCPPOCfg())
task_registry.register("dog_wtw_lite_c_mass137", DogWTWLight, DogWTWLightCMass137Cfg(), DogWTWLightCMass137PPOCfg())
task_registry.register("dog_wtw_lite_c_mass137_clearance3", DogWTWLight, DogWTWLightCMass137Clearance3Cfg(), DogWTWLightCMass137Clearance3PPOCfg())
task_registry.register("dog_wtw_lite_original17_contact2", DogWTWLight, DogWTWLightOriginal17Contact2Cfg(), DogWTWLightOriginal17Contact2PPOCfg())
task_registry.register("dog_wtw_lite_original23_contact2", DogWTWLight, DogWTWLightOriginal23Contact2Cfg(), DogWTWLightOriginal23Contact2PPOCfg())
task_registry.register("dog_wtw_lite_c_stable_2", DogWTWLight, DogWTWLightCStable2Cfg(), DogWTWLightCStable2PPOCfg())
task_registry.register("dog_wtw_lite_c_stable_3_raibert", DogWTWLight, DogWTWLightCStable3RaibertCfg(), DogWTWLightCStable3RaibertPPOCfg())
task_registry.register("dog_wtw_lite_c_stable_4", DogWTWLight, DogWTWLightCStable4Cfg(), DogWTWLightCStable4PPOCfg())
task_registry.register("dog_wtw_lite_c_slow_010", DogWTWLight, DogWTWLightCSlow010Cfg(), DogWTWLightCSlow010PPOCfg())
task_registry.register("dog_wtw_lite_d", DogWTWLight, DogWTWLightDCfg(), DogWTWLightDPPOCfg())