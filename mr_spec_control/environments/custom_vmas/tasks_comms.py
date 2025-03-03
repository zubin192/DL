import math
import typing
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator import rendering
# from sim.coordinator import Coordinator
# from sim.worker import Worker
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class Scenario(BaseScenario):
    """
    The methods that are **compulsory to instantiate** are:

    - :class:`make_world`
    - :class:`reset_world_at`
    - :class:`observation`
    - :class:`reward`

    The methods that are **optional to instantiate** are:

    - :class:`info`
    - :class:`extra_render`
    - :class:`process_action`
    - :class:`pre_step`
    - :class:`post_step`

    """

    def make_world(self, batch_dim, device, **kwargs):
        """
        This function needs to be implemented when creating a scenario.
        In this function the user should instantiate the world and insert agents and landmarks in it.

        Args:
            batch_dim (int): the number of vecotrized environments.
            device (Union[str, int, torch.device], optional): the device of the environmemnt.
            kwargs (dict, optional): named arguments passed from environment creation

        Returns:
            :class:`~vmas.simulator.core.World` : the :class:`~vmas.simulator.core.World`
            instance which is automatically set in :class:`~world`.

        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def make_world(self, batch_dim: int, device: torch.device, **kwargs):
            ...         # Pass any kwargs you desire when creating the environment
            ...         n_agents = kwargs.get("n_agents", 5)
            ...
            ...         # Create world
            ...         world = World(batch_dim, device, dt=0.1, drag=0.25, dim_c=0)
            ...         # Add agents
            ...         for i in range(n_agents):
            ...             agent = Agent(
            ...                 name=f"agent {i}",
            ...                 collide=True,
            ...                 mass=1.0,
            ...                 shape=Sphere(radius=0.04),
            ...                 max_speed=None,
            ...                 color=Color.BLUE,
            ...                 u_range=1.0,
            ...             )
            ...             world.add_agent(agent)
            ...         # Add landmarks
            ...         for i in range(5):
            ...             landmark = Landmark(
            ...                 name=f"landmark {i}",
            ...                 collide=True,
            ...                 movable=False,
            ...                 shape=Box(length=0.3,width=0.1),
            ...                 color=Color.RED,
            ...             )
            ...             world.add_landmark(landmark)
            ...         return world
        """

        # Pass any kwargs you desire when creating the environment
        # Overall
        self.use_mothership = kwargs.pop("use_mothership", False)
        self.num_agents = kwargs.get("num_agents", 5)
        self.num_tasks = kwargs.get("num_tasks", 5)
        self.random_tasks = kwargs.get("random_tasks", True)
        self.tasks_respawn = kwargs.pop("tasks_respawn", True)
        self._comms_range = kwargs.pop("comms_range", 1.0)
        self.task_comp_range = kwargs.pop("task_comp_range", 0.05)
        # Rendering
        self.agent_radius = kwargs.pop("agent_radius", 0.025)
        self.task_radius = kwargs.pop("task_radius", self.agent_radius)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.25)
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)
        # Reward-specific
        self.shared_reward = kwargs.pop("shared_reward", False)
        self.time_penalty = kwargs.pop("time_penalty", -1)
        self._agents_per_task = kwargs.pop("agents_per_task", 1)


        modality_funcs = kwargs.get("modality_funcs", [])
        sim_action_func = kwargs.get("sim_action_func", None)

        max_vel = 10.0
        self.comms_dec_rate = 8
        self.comms_decay = True
        self.viewer_zoom = 1
        self.complete_task_coeff = 1.0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        if self.use_mothership: # mothership adds extra agent
            self.num_agents += 1

        self.batch_dim = batch_dim
        # self.device = device

        # Create world
        # world = World(batch_dim, device, dt=0.1, drag=0.5, dim_c=0)
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )

        # Add agents
        for i in range(self.num_agents):
            # Mothership
            if i == 0 and self.use_mothership:
                name = f"mothership_{i}"
                movable = False
            else:
                name = f"passenger_{i}"
                movable = True

            agent = Agent(
                name=name,
                collide=True,
                movable=movable,
                shape=Sphere(radius=self.agent_radius),
                mass=5.0,
                max_speed=max_vel,
                u_range=1.0,
                color=Color.BLUE,
                # sim_action_func=sim_action_func,
            )
            agent.completion_reward = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        # Add tasks
        self._tasks = []
        for i in range(self.num_tasks):
            task = Landmark(
                name=f"task {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.task_radius),
                color=Color.RED,
            )
            world.add_landmark(task)
            self._tasks.append(task)

        self.complete_tasks = torch.zeros(batch_dim, self.num_tasks, device=device)
        self.shared_completion_rew = torch.zeros(batch_dim, device=device)
        self.time_rew = torch.zeros(batch_dim, device=device)


        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

        return world

    def reset_world_at(self, env_index=None):
        """Resets the world at the specified env_index.

        When a ``None`` index is passed, the world should make a vectorized (batched) reset.
        The ``entity.set_x()`` methods already have this logic integrated and will perform
        batched operations when index is ``None``.

        When this function is called, all entities have already had their state reset to zeros according to the ``env_index``.
        In this function you shoud change the values of the reset states according to your task.
        For example, some functions you might want to use are:

        - ``entity.set_pos()``,
        - ``entity.set_vel()``,
        - ``entity.set_rot()``,
        - ``entity.set_ang_vel()``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            env_index (int, otpional): index of the environment to reset. If ``None`` a vectorized reset should be performed.

        Spawning at fixed positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            ...        for i, agent in enumerate(self.world.agents):
            ...            agent.set_pos(
            ...                torch.tensor(
            ...                     [-0.2 + 0.1 * i, 1.0],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...        for i, landmark in enumerate(self.world.landmarks):
            ...            landmark.set_pos(
            ...                torch.tensor(
            ...                     [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...            landmark.set_rot(
            ...                torch.tensor(
            ...                     [torch.pi / 4 if i % 2 else -torch.pi / 4],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )

        Spawning at random positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import ScenarioUtils
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            >>>         ScenarioUtils.spawn_entities_randomly(
            ...             self.world.agents + self.world.landmarks,
            ...             self.world,
            ...             env_index,
            ...             min_dist_between_entities=0.02,
            ...             x_bounds=(-1.0,1.0),
            ...             y_bounds=(-1.0,1.0),
            ...         )

        """

        # First-time startup
        if env_index is None:
            self.all_time_complete_tasks = torch.full(
                (self.world.batch_dim, self.num_tasks),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_complete_tasks[env_index] = False

        # Place entities
        if self.use_mothership:
            placable_entities = self._tasks[: self.num_tasks] + \
                    [self.world.agents[0]]
        else:
            placable_entities = self._tasks[: self.num_tasks] + \
                    self.world.agents[: self.num_agents]

        ScenarioUtils.spawn_entities_randomly(
            entities=placable_entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
        for task in self._tasks[self.num_tasks :]:
            task.set_pos(self.get_outside_pos(env_index), batch_index=env_index)

        # Spawn passengers around mothership
        if self.use_mothership:
            mothership_pos = self.world.agents[0].state.pos
            for agent in self.world.agents[1:]:
                agent.set_pos(mothership_pos + (torch.rand(1, 2, device=self.world.device)*2 - 1)*0.1, batch_index=env_index)

    def observation(self, agent):
        """This function computes the observations for ``agent`` in a vectorized way.

        The returned tensor should contain the observations for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim, n_agent_obs)``, or be a dict with leaves following that shape.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the observations for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the observation

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         # get positions of all landmarks in this agent's reference frame
            ...         landmark_rel_poses = []
            ...         for landmark in self.world.landmarks:
            ...             landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
            ...         return torch.cat([agent.state.pos, agent.state.vel, *landmark_rel_poses], dim=-1)

        You can also return observations in a dictionary

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         return {"pos": agent.state.pos, "vel": agent.state.vel}

        """
        obs = {}
        obs["pos"] = agent.state.pos
        # obs["vel"] = agent.state.vel
        obs["task_pos"] = torch.cat([t.state.pos for t in self._tasks], dim=1)

        # NOTE Passengers get ONLY their sensor views?
        if "mothership" in agent.name:
            # Mothership obs (global agents & tasks)
            obs["passenger_pos"] = torch.cat(
                    [a.state.pos for a in self.world.agents[1:]], dim=1
                )
            obs["task_pos"] = torch.cat([t.state.pos for t in self._tasks], dim=1)
        else:
            # Passenger obs

            # TODO: Extract passenger-specific actions from mothership.action.u
            if self.use_mothership:
                if self.world.agents[0].action.u is None:
                    obs["mothership_actions"] = torch.zeros((self.world.batch_dim,
                                                            self.world.agents[0].action_size),
                                                            device=self.world.device)
                else:
                    obs["mothership_actions"] = self.world.agents[0].action.u

        return obs

    def reward(self, agent):
        """This function computes the reward for ``agent`` in a vectorized way.

        The returned tensor should contain the reward for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim)`` and dtype ``torch.float``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the reward for

        Returns:
             torch.Tensor: reward tensor of shape ``(self.world.batch_dim)``

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reward(self, agent):
            ...         # reward every agent proportionally to distance from first landmark
            ...         rew = -torch.linalg.vector_norm(agent.state.pos - self.world.landmarks[0].state.pos, dim=-1)
            ...         return rew
        """

        # Reward every agent with sum of completed tasks
        # completed_tasks = []
        # for landmark in self.world.landmarks:
        #     completed_tasks.append(landmark.complete)

        # completed_tasks = torch.stack(completed_tasks, dim=1)
        # rew = torch.sum(completed_tasks, dim=1, dtype=float).unsqueeze(-1)

    # TODO Compute mothership reward
        if self.use_mothership:
            is_first = agent == self.world.agents[1] # 0 is mothership
        else:
            is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        # Time reward, Covering reward
        if is_first:
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.tasks_pos = torch.stack([t.state.pos for t in self._tasks], dim=1)
            self.agents_tasks_dists = torch.cdist(self.agents_pos, self.tasks_pos)
            self.agents_per_task = torch.sum(
                (self.agents_tasks_dists < self.task_comp_range).type(torch.int),
                dim=1,
            )
            self.complete_tasks = self.agents_per_task >= self._agents_per_task

            self.shared_completion_rew[:] = 0
            for a in self.world.agents:
                self.shared_completion_rew += self.passenger_completion_reward(a)
            self.shared_completion_rew[self.shared_completion_rew != 0] /= 2

        tasks_rew = (
            agent.completion_reward
            if not self.shared_reward
            else self.shared_completion_rew
        )

        # Respawn tasks
        if is_last:
            if self.tasks_respawn:
                occupied_positions_agents = [self.agents_pos]
                for i, task in enumerate(self._tasks):
                    occupied_positions_tasks = [
                        o.state.pos.unsqueeze(1)
                        for o in self._tasks
                        if o is not task
                    ]
                    occupied_positions = torch.cat(
                        occupied_positions_agents + occupied_positions_tasks,
                        dim=1,
                    )
                    pos = ScenarioUtils.find_random_pos_for_entity(
                        occupied_positions,
                        env_index=None,
                        world=self.world,
                        min_dist_between_entities=self._min_dist_between_entities,
                        x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                        y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                    )

                    task.state.pos[self.complete_tasks[:, i]] = pos[
                        self.complete_tasks[:, i]
                    ].squeeze(1)
            else:
                self.all_time_complete_tasks += self.complete_tasks
                for i, task in enumerate(self._tasks):
                    task.state.pos[self.complete_tasks[:, i]] = self.get_outside_pos(
                        None
                    )[self.complete_tasks[:, i]]


        # Distance to nearest task reward (inverse)
        tasks_pos = torch.stack([task.state.pos for task in self._tasks])
        dists_to_tasks = torch.linalg.norm(agent.state.pos - tasks_pos, dim=2)
        near_task_rew = 1 / torch.min(dists_to_tasks)
        # print(f"Dists to tasks\n {dists_to_tasks}, Nearest: {near_task_rew}")

        return tasks_rew #+ self.time_rew + near_task_rew

    def passenger_completion_reward(self, agent):
        """Reward for covering targets"""
        agent_index = self.world.agents.index(agent)

        agent.completion_reward[:] = 0
        targets_covered_by_agent = (
            self.agents_tasks_dists[:, agent_index] < self.task_comp_range
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.complete_tasks
        ).sum(dim=-1)
        agent.completion_reward += (
            num_covered_targets_covered_by_agent * self.complete_task_coeff
        )
        return agent.completion_reward

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "tasks_reward": (
                agent.completion_reward
                if not self.shared_reward
                else self.shared_completion_rew
            ),
            "tasks_complete": self.complete_tasks.sum(-1),
        }
        return info

    def done(self):
        return self.all_time_complete_tasks.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":

        geoms: List[Geom] = []
        # task ranges
        # for task in self._tasks:
        #     range_circle = rendering.make_circle(self.task_comp_range, filled=False)
        #     xform = rendering.Transform()
        #     xform.set_translation(*task.state.pos[env_index])
        #     range_circle.add_attr(xform)
        #     range_circle.set_color(*self.task_color.value)
        #     geoms.append(range_circle)
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.RED.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms

if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(scenario)
