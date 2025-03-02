#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from . import discovery_obstacles, example, tasks_comms


class CustomVmasTask(Task):
    """Enum for VMAS tasks."""

    # List enum here
    DISCOVERY_OBSTACLES = None
    TASKS_COMMS = None
    EXAMPLE = None

    def get_env_fun(
            self,
            num_envs: int,
            continuous_actions: bool,
            seed: Optional[int],
            device: DEVICE_TYPING,
        ) -> Callable[[], EnvBase]:

            config = copy.deepcopy(self.config)
            if self is CustomVmasTask.DISCOVERY_OBSTACLES: # This is the only modification we make ....
                scenario = discovery_obstacles.Scenario()
            elif self is CustomVmasTask.TASKS_COMMS:
                scenario = tasks_comms.Scenario() # .... ends here
            elif self is CustomVmasTask.EXAMPLE:
                scenario = example.Scenario() # .... ends here
            else:
                scenario = self.name.lower()
            # group_map = MarlGroupMapType.ALL_IN_ONE_GROUP
            return lambda: VmasEnv(
                scenario=scenario,
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                **config,
            )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "vmas"
