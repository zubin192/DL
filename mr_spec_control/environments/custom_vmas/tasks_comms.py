import math

import numpy as np
import torch
from vmas import render_interactively
# from sim.coordinator import Coordinator
# from sim.worker import Worker
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


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
        self.use_mothership = kwargs.pop("use_mothership", True)
        self.num_agents = kwargs.get("num_agents", 5)
        self.num_tasks = kwargs.get("num_tasks", 5)
        self.random_tasks = kwargs.get("random_tasks", True)
        self.tasks_respawn = kwargs.pop("tasks_respawn", True)
        self._comms_range = kwargs.pop("comms_range", 1.0)
        # Reward-specific
        self.shared_reward = kwargs.pop("shared_reward", False)
        self.time_penalty = kwargs.pop("time_penalty", -1)
        # Rendering
        self.agent_radius = kwargs.pop("agent_radius", 0.025)
        self.task_radius = kwargs.pop("task_radius", self.agent_radius)
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)

        modality_funcs = kwargs.get("modality_funcs", [])
        sim_action_func = kwargs.get("sim_action_func", None)

        max_vel = 10.0
        self.task_comp_thresh = 0.05
        self.comms_dec_rate = 8
        self.comms_decay = True
        self.viewer_zoom = 1

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
            agent.covering_reward = agent.collision_rew.clone()
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

        self.covered_tasks = torch.zeros(batch_dim, self.num_tasks, device=device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=device)
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
        placable_entities = self._obstacles[: self.n_obstacles] + \
            self._targets[: self.num_tasks] + \
                self.world.agents[: self.n_agents]

        ScenarioUtils.spawn_entities_randomly(
            entities=placable_entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
        for target in self._targets[self.n_targets :]:
            target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)

        # Spawn passengers around mothership
        # mothership_pos = self.world.agents[0].state.pos
        # for agent in self.world.agents[1:]:
        #     agent.set_pos(mothership_pos + (torch.rand(1, 2, device=self.world.device)*2 - 1)*0.1, batch_index=env_index)


        for i, agent in enumerate(self.world.agents):
            if "coordinator" in agent.name:
                agent.set_pos(
                    torch.tensor(
                        [0.0, 0.0],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            else:
                angle = 2 * math.pi * (i - 1) / self.num_agents
                agent.set_pos(
                    torch.tensor(
                        [0.1 * math.cos(angle), 0.1 * math.sin(angle)],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            agent.comms_noise = torch.zeros(
                (self.world.batch_dim,), device=self.world.device
            )

        # for i, landmark in enumerate(self.world.landmarks):
        #     if self.random_tasks:
        #         angle = 2 * math.pi * (i) / self.num_tasks
        #         landmark.set_pos(
        #             torch.tensor(
        #                 [math.cos(angle), math.sin(angle)],
        #                 dtype=torch.float32,
        #                 device=self.world.device,
        #             ),
        #             batch_index=env_index,
        #         )
        #     else:
        #         coords = np.random.rand(2)
        #         landmark.set_pos(
        #             torch.tensor(
        #                 [coords[0], coords[1]],
        #                 dtype=torch.float32,
        #                 device=self.world.device,
        #             ),
        #             batch_index=env_index,
        #         )

        #     if env_index is None:
        #         landmark.complete = torch.full(
        #             (self.world.batch_dim,), False, device=self.world.device
        #         )
        #         landmark.reset_render()
        #         self._done[:] = False
        #     else:
        #         landmark.complete[env_index] = False
        #         landmark.is_rendering[env_index] = True
        #         self._done[env_index] = False

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
        # NOTE Processing task completion here
        for landmark in self.world.landmarks:
            completion_mask = landmark.complete.clone()
            abs_dists = torch.abs(landmark.state.pos - agent.state.pos)
            landmark.complete = torch.norm(abs_dists, dim=1) < self.task_comp_thresh
            landmark.complete[completion_mask] = True

        output_dict = {}

        # Evaluate pos localization with noise
        if "coordinator" not in agent.name and self.comms_decay:
            cum_noises = []
            for a_other in self.world.agents:
                if a_other.name != agent.name:
                    noise_to_other = torch.exp(
                        self.comms_dec_rate
                        * torch.norm(a_other.state.pos - agent.state.pos, dim=1)
                        - self.comms_dec_rate
                    )
                    cum_noises.append(a_other.comms_noise + noise_to_other)

            stacked_comms = torch.stack(cum_noises, dim=0)

            # Find the minimum noise values
            best_comms = torch.min(stacked_comms, dim=0).values
            # NOTE: Treat actual pos as mean, noise as SE. Sample from distro.
            agent.comms_noise = best_comms

            # Apply noise to pos
            output_dict["pos"] = torch.normal(
                agent.state.pos,
                agent.comms_noise.unsqueeze(-1).expand(self.batch_dim, 2),
            )
        else:
            output_dict["pos"] = agent.state.pos

        # Get other agents' positions
        for a_other in self.world.agents:
            if a_other.name != agent.name:
                output_dict[a_other.name + " pos"] = (
                    a_other.state.pos - output_dict["pos"]
                )

        # Get positions of all landmarks in this agent's reference frame
        for landmark in self.world.landmarks:
            output_dict[landmark.name + " pos"] = (
                landmark.state.pos - output_dict["pos"]
            )
            output_dict[landmark.name + " status"] = landmark.complete.unsqueeze(-1)

        return output_dict

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
        completed_tasks = []
        for landmark in self.world.landmarks:
            completed_tasks.append(landmark.complete)

        completed_tasks = torch.stack(completed_tasks, dim=1)
        rew = torch.sum(completed_tasks, dim=1, dtype=float).unsqueeze(-1)

        return rew

if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(scenario)
