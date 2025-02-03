import copy
import time
from typing import Union

import numpy as np
import torch
from moviepy import ImageSequenceClip
from vmas import make_env
from vmas.simulator.scenario import BaseScenario

from modalities.tasks_comms import get_mode_do_comms, get_mode_do_tasks
from scenarios.tasks_comms import ScenarioTaskComms
from specialization_policy import SpecializationPolicy

# === EA Stuff ===


def tournament_selection(population: list[SpecializationPolicy], tournament_size=10):
    """
    Select one individual using tournament selection. Randomly pick `tournament_size`
    individuals and select the one with the highest fitness.
    """
    policies = np.random.choice(population, tournament_size)
    best = max(policies, key=lambda pol: pol.fitness)
    return best


def crossover(parent1: SpecializationPolicy, parent2: SpecializationPolicy):
    """Perform crossover between two networks."""
    child = copy.deepcopy(parent1)
    child.fitness = None
    for param1, param2 in zip(child.parameters(), parent2.parameters()):
        mask = torch.rand_like(param1) < 0.5
        param1.data[mask] = param2.data[mask]
    return child


def mutate(policy: SpecializationPolicy, mutation_rate=0.1):
    """Mutate a network by adding noise to its parameters."""
    for param in policy.parameters():
        if torch.rand(1).item() < mutation_rate:
            noise = torch.randn_like(param) * 0.1
            param.data.add_(noise)


def vmas_env_EA_train_static(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    random_action: bool,
    **kwargs,
):
    """
    This function trains policies by evaluating batch_dim policies on same sim.

    Use if we want to train policy for static environment.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """

    # Initialize scenario and env
    scenario_name = (
        scenario if isinstance(scenario, str) else scenario.__class__.__name__
    )

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        **kwargs,
    )

    # Get observation dims for scenario
    # Need dims batch_size x obs_size
    obs = env.reset()
    combined_obs = []
    for key in obs[0].keys():  # convert all to floats
        if key == "pos":
            continue
        combined_obs.append(obs[0][key].float())
    combined_obs = torch.cat(combined_obs, dim=1)

    # Initialize policy population (one for each env)
    pol_pop = [
        SpecializationPolicy(
            combined_obs.shape[1],
            num_agents - 1,
            num_modes,
            device=device,
        )
        for _ in range(num_envs)
    ]

    # EA Loop
    render = False
    ep = 0
    best_rew = 0.0
    while ep < epochs:
        print("Epoch:", ep)

        if ep == epochs - 1:
            render = True

        # Evaluate policy fitnesses
        frame_list = []  # For creating a gif
        init_time = time.time()

        # ==== Scenario simulation loop ====
        obs = env.reset()  # NOTE reset env for each simulation loop
        for i, agent in enumerate(env.agents):  # initialize obs
            agent.process_obs(obs[i])

        for s in range(n_steps):
            print(f"Step {s}")
            # Update coordinator specialization assignments
            # NOTE Uses one policy per env in batch here!
            specs = env.agents[0].update_specializations_pop(pol_pop)

            # Act using specs scaling
            actions = []
            for i, agent in enumerate(env.agents):
                if i != 0:
                    agent.specialization = specs[i - 1]
                agent.process_obs(obs[i])
                actions.append(agent.get_action(env, i))

            # print("Actions:", actions)
            obs, rews, dones, info = env.step(actions)

            # TODO - How to render specific environments?
            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                )
                frame_list.append(frame)

        if render:
            fps = 30
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_gif(f"img/{scenario_name}.gif", fps=fps)

        # Calculate fitness (average over cum_rews entries)
        rews = torch.cat(rews[1:], dim=1)
        avg_rews = torch.mean(rews, dim=1).tolist()
        for i, pol in enumerate(pol_pop):
            pol.fitness = avg_rews[i]

        best = max(pol_pop, key=lambda pol: pol.fitness)
        print("\tBest Rew:", best.fitness)

        # Select, crossover, mutate population
        new_pop = [copy.deepcopy(best)]
        new_pop[0].fitness = None
        while len(new_pop) < env.batch_dim:
            parent1 = tournament_selection(pol_pop, tournament_size=len(pol_pop) // 4)
            parent2 = tournament_selection(pol_pop, tournament_size=len(pol_pop) // 4)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate=0.5)
            new_pop.append(child)
        pol_pop = new_pop

        ep += 1

    # Process results, render
    total_time = time.time() - init_time

    return best


def vmas_env_EA_train_random(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    random_action: bool,
    **kwargs,
):
    """
    This function trains policies by evaluating each pol over batch_dim environment simulations.

    Use if we want to evaluate policy effectiveness over batch_dim random sims.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """

    # Initialize scenario and env
    scenario_name = (
        scenario if isinstance(scenario, str) else scenario.__class__.__name__
    )

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        **kwargs,
    )

    # Get observation dims for scenario
    # Need dims batch_size x obs_size
    obs = env.reset()
    combined_obs = []
    for key in obs[0].keys():  # convert all to floats
        if key == "pos":
            continue
        # print(key, obs[0][key].float())
        combined_obs.append(obs[0][key].float())
    combined_obs = torch.cat(combined_obs, dim=1)
    # print("combined_obs:", combined_obs, combined_obs.shape)

    # Initialize policy population
    # output_shape = (num_agents-1, env.batch_dim, num_modes)
    pol_pop = [
        SpecializationPolicy(
            combined_obs.shape[1], num_agents - 1, num_modes, device=device
        )
        for _ in range(num_pols)
    ]

    # EA Loop
    ep = 0
    while ep < epochs:
        # Evaluate policy fitnesses
        for pol in pol_pop:
            if pol.fitness != None:
                continue

            frame_list = []  # For creating a gif
            init_time = time.time()

            # ==== Scenario simulation loop ====
            obs = env.reset()  # NOTE reset env for each simulation loop
            for i, agent in enumerate(env.agents):  # initialize obs
                agent.process_obs(obs[i])

            # cum_rews = None
            for s in range(n_steps):
                print(f"Step {s}")
                # Update coordinator specialization assignments
                specs = env.agents[0].update_specializations(pol)

                # Act using specs scaling
                actions = []
                for i, agent in enumerate(env.agents):
                    if i != 0:
                        agent.specialization = specs[i - 1]
                    agent.process_obs(obs[i])
                    actions.append(agent.get_action(env, i))

                # print("Actions:", actions)
                obs, rews, dones, info = env.step(actions)

                if render:
                    frame = env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                    )
                    frame_list.append(frame)

            if render:
                from moviepy import ImageSequenceClip

                fps = 30
                clip = ImageSequenceClip(frame_list, fps=fps)
                clip.write_gif(f"img/{scenario_name}.gif", fps=fps)

        # TODO Calculate fitness (average over cum_rews entries)
        print("Rewards:", rews)
        avg_rew = torch.mean(torch.cat(rews, dim=1))
        pol.fitness = avg_rew.values()
        print("Fitness:", pol.fitness)

        # TODO Select, crossover, mutate population
        ep += 1

        # Process results, render
        total_time = time.time() - init_time

    return max(pol_pop, key=lambda policy: policy.fitness)


if __name__ == "__main__":
    num_agents = 3  # NOTE agent 0 is mothership
    num_tasks = 2
    num_modes = 2  # NOTE specific to problem type!
    num_pols = 1
    epochs = 5
    num_envs = 32  # Batch size
    n_steps = 50  # 100
    device = "cpu"  # cuda

    verbose = False
    render = True

    modality_funcs = [get_mode_do_tasks, get_mode_do_comms]
    scenario = ScenarioTaskComms()

    trained_pol = vmas_env_EA_train_static(
        scenario=scenario,
        render=render,
        num_envs=num_envs,
        n_steps=n_steps,
        device=device,
        continuous_actions=False,
        random_action=False,
        # Environment variables
        num_agents=num_agents,
        num_tasks=num_tasks,
        num_modes=num_modes,
        modality_funcs=modality_funcs,
        # Training variables
        epochs=epochs,
        num_pols=num_pols,
        verbose=verbose,
    )
