import math

import torch
from vmas.simulator.core import Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from sim.coordinator import Coordinator
from sim.worker import Worker


class ScenarioTaskComms(BaseScenario):

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
        self.num_agents = kwargs.get("num_agents", 5)
        self.num_tasks = kwargs.get("num_tasks", 5)
        modality_funcs = kwargs.get("modality_funcs", [])
        sim_action_func = kwargs.get("sim_action_func", None)
        
        max_vel = 0.2
        self.task_comp_thresh = 0.05
        self.comms_dec_rate = 8
        self.comms_decay = True

        self.batch_dim = batch_dim
        self.device = device
        
        # Create world
        world = World(batch_dim, device, dt=0.1, drag=0.5, dim_c=0)
        # Add agents
        for i in range(self.num_agents):
            if i == 0: # Agent 0 is mothership
                agent = Coordinator(
                    name=f"coordinator {i}",
                    collide=True,
                    mass=100.0,
                    shape=Sphere(radius=0.04),
                    max_speed=0.0,
                    color=Color.BLUE,
                    u_range=1.0,
                )
            else: # All other agents are homogeneous
                agent = Worker(
                    name=f"worker {i}",
                    collide=True,
                    mass=1.0,
                    shape=Sphere(radius=0.02),
                    max_speed=max_vel,
                    color=Color.BLUE,
                    u_range=1.0,
                    modality_funcs=modality_funcs,
                    sim_action_func=sim_action_func,
                )
            world.add_agent(agent)
        # Add tasks
        for i in range(self.num_tasks):
            task = Landmark(
                name=f"task {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=0.04),
                color=Color.RED,
            )
            world.add_landmark(task)
     
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

        return world
    

    def reset_world_at(self, env_index = None):
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
                            [0.1*math.cos(angle),
                             0.1*math.sin(angle)],
                            dtype=torch.float32,
                            device=self.world.device,
                    ),
                        batch_index=env_index,
                )
            agent.comms_noise = torch.zeros((self.world.batch_dim,),device=self.world.device
            )
                
        for i, landmark in enumerate(self.world.landmarks):
            angle = 2 * math.pi * (i) / self.num_tasks
            landmark.set_pos(
                torch.tensor(
                        [math.cos(angle),
                         math.sin(angle)],
                        dtype=torch.float32,
                        device=self.world.device,
                ),
                    batch_index=env_index,
            )

            if env_index is None:
                landmark.complete = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
                self._done[:] = False
            else:
                landmark.complete[env_index] = False
                landmark.is_rendering[env_index] = True
                self._done[env_index] = False


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
            abs_dists = (torch.abs(landmark.state.pos - agent.state.pos))
            landmark.complete = torch.norm(abs_dists, dim=1) < self.task_comp_thresh
            landmark.complete[completion_mask] = True
            # print("Task", landmark.name, "Status:\n", landmark.complete)
        
        output_dict = {}
        
        # Evaluate pos localization with noise
        if 'coordinator' not in agent.name and self.comms_decay:
            cum_noises = []
            for a_other in self.world.agents:
                if a_other.name != agent.name:
                    # print("")
                    # print("Dist to other:\n", torch.norm(a_other.state.pos - agent.state.pos, dim=1))
                    noise_to_other = torch.exp(self.comms_dec_rate*torch.norm(a_other.state.pos - agent.state.pos, dim=1)-self.comms_dec_rate) #/self.max_random_dist)
                    # print("Noise to other:\n", noise_to_other)
                    cum_noises.append(a_other.comms_noise + noise_to_other)
                    
                    # print("Cumulative noise with other:\n", a_other.comms_noise + noise_to_other)
            
            stacked_comms = torch.stack(cum_noises, dim=0)   
            # print("Stacked comms:\n", stacked_comms)    

            # Find the minimum noise values
            best_comms = torch.min(stacked_comms, dim=0).values
            # print("Best comms:\n", best_comms)
            # NOTE: Treat actual pos as mean, noise as SE. Sample from distro.
            agent.comms_noise = best_comms
            
            # TODO apply noise to pos
            # print("Pos:\n", agent.state.pos, "Noise:\n", agent.comms_noise.unsqueeze(-1).expand(self.batch_dim,2))
            output_dict['pos'] = torch.normal(agent.state.pos, 
                                              agent.comms_noise.unsqueeze(-1).expand(self.batch_dim,2))
            # print("Actual pos:\n", agent.state.pos)
            # print("Noisy pos:\n", output_dict['pos'])
            
        else:
            output_dict['pos'] = agent.state.pos
        
        # agent_rel_poses = []
        for a_other in self.world.agents:
            if a_other.name != agent.name:
                # agent_rel_poses.append(a_other.state.pos - agent.state.pos)
                output_dict[a_other.name+" pos"] = a_other.state.pos - output_dict['pos']
        
        # get positions of all landmarks in this agent's reference frame
        # task_rel_poses = []
        # task_complete_statuses = []
        for landmark in self.world.landmarks:
            # task_rel_poses.append(landmark.state.pos - agent.state.pos)
            # task_complete_statuses.append(landmark.complete.unsqueeze(-1))
            output_dict[landmark.name+" pos"] = landmark.state.pos - output_dict['pos']
            output_dict[landmark.name+" status"] = landmark.complete.unsqueeze(-1)
            

        return output_dict


    # NOTE we don't use this reward right now
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

        # reward every agent proportionally to distance from first landmark
        completed_tasks = []
        for landmark in self.world.landmarks:
            completed_tasks.append(landmark.complete)
        
        # print("List:\n", completed_tasks)
        completed_tasks = torch.stack(completed_tasks, dim=1)
        # print("Stacked:\n", completed_tasks)
        rew = torch.sum(completed_tasks, dim=1, dtype=float).unsqueeze(-1)
        
        return rew