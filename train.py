
import time
from typing import Union

import numpy as np
# import pyvirtualdisplay
import torch
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.scenario import BaseScenario

from modalities.tasks_comms import get_mode_do_comms, get_mode_do_tasks
from scenarios.tasks_comms import ScenarioTaskComms
from specialization_policy import SpecializationPolicy


def vmas_env_EA_train(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    random_action: bool,
    **kwargs
):
    """Example function to use a vmas environment.
    
    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """

    # num_pols = kwargs.get("num_pols", None)
    # num_agents = kwargs.get("num_agents", None)
    # num_tasks = kwargs.get("num_tasks", None)
    # modality_funcs = kwargs.get("modality_funcs", None)
    # epochs = kwargs.get("epochs", None)
    # verbose = kwargs.get("verbose", False)

    # Initialize policy population
    pol_pop = [SpecializationPolicy() for _ in range(num_pols)]

    # Initialize scenario and env
    scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__
    env = make_env(scenario=scenario,
                    num_envs=num_envs,
                    device=device,
                    continuous_actions=continuous_actions,
                    seed=0,
                    # Environment specific variables
                    **kwargs
                )

    # EA Loop
    ep = 0
    while ep < epochs:
        # Evaluate policy fitnesses
        for pol in pol_pop:
            if pol.fitness != None: continue

            frame_list = []  # For creating a gif
            init_time = time.time()
            
            # ==== Scenario simulation loop ====
            obs = env.reset() # NOTE reset env for each simulation loop
            for i, agent in enumerate(env.agents): # initialize obs
                agent.process_obs(obs[i])
            cum_rews = None
            for s in range(n_steps):
                print(f"Step {s}")

                # TODO Update coordinator specialization assignments
                specs = env.agents[0].update_specializations(env, pol)

                # Act using specs scaling
                actions = []
                for i, agent in enumerate(env.agents):
                    if i != 0:
                        agent.specialization = specs[i]
                    agent.process_obs(obs[i])
                    actions.append(agent.get_action(env))
                
                print("Actions:", actions)
                obs, rews, dones, info = env.step(actions)
                if cum_rews == None: cum_rews = rews
                else:
                    cum_rews = [torch.add(a_cum_rew, a_rew) for a_cum_rew, a_rew in zip(cum_rews, rews)]

                if verbose: print("Cum rews:", cum_rews)

                if render:
                    frame = env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                    )
                    frame_list.append(frame)

            if render:
                from moviepy import ImageSequenceClip
                fps=30
                clip = ImageSequenceClip(frame_list, fps=fps)
                clip.write_gif(f'img/{scenario_name}.gif', fps=fps)

        # TODO Calculate fitness (average over cum_rews entries)
        pol.fitness = torch.mean(torch.tensor([torch.mean(a_cum_rews) for a_cum_rews in cum_rews]))
        if verbose: print("Fitness:", pol.fitness)

        # TODO Select, crossover, mutate population
        ep+=1

        # Process results, render
        total_time = time.time() - init_time
        
    return np.max(pol_pop, key=lambda policy: policy.fitness)

        

if __name__ == "__main__":
    num_agents=4 # NOTE agent 0 is mothership
    num_tasks=3
    num_pols=1
    epochs=1
    num_envs = 4 # 32
    n_steps = 5 # 100
    device = 'cpu' #cuda

    verbose = False
    render = True

    modality_funcs = [get_mode_do_tasks, get_mode_do_comms]
    scenario = ScenarioTaskComms()

    trained_pol = vmas_env_EA_train(
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
                                    modality_funcs=modality_funcs,
                                    # Training variables
                                    epochs=epochs,
                                    num_pols=num_pols,
                                    verbose=verbose
                                )

    # from IPython.display import Image
    # Image(f'{scenario_name}.gif')