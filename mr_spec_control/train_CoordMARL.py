
import os
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from environments.custom_vmas.common import CustomVmasTask

# !apt-get update
# !apt-get install -y x11-utils python3-opengl xvfb
# !pip install pyvirtualdisplay torchvision "av<14"
# import pyvirtualdisplay
# display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
# display.start()


# def use_vmas_env(
#     render: bool,
#     num_envs: int,
#     n_steps: int,
#     device: str,
#     scenario: Union[str, BaseScenario],
#     continuous_actions: bool,
#     **kwargs
# ):
#     """Example function to use a vmas environment.

#     This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

#     Args:
#         continuous_actions (bool): Whether the agents have continuous or discrete actions
#         scenario (str, BaseScenario): Name of scenario or scenario class
#         device (str): Torch device to use
#         render (bool): Whether to render the scenario
#         num_envs (int): Number of vectorized environments
#         n_steps (int): Number of steps before returning done

#     """

#     scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__

#     env = make_env(
#         scenario=scenario,
#         num_envs=num_envs,
#         device=device,
#         continuous_actions=continuous_actions,
#         seed=0,
#         # Environment specific variables
#         **kwargs
#     )

#     frame_list = []  # For creating a gif
#     init_time = time.time()
#     step = 0

#     for s in range(n_steps):
#         step += 1
#         print(f"Step {step}")

#         actions = []
#         for i, agent in enumerate(env.agents):
#             action = env.get_random_action(agent)

#             actions.append(action)

#         obs, rews, dones, info = env.step(actions)

#         if render:
#             frame = env.render(mode="rgb_array")
#             frame_list.append(frame)

#     total_time = time.time() - init_time
#     print(
#         f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
#         f"for {scenario_name} scenario."
#     )

#     if render:
#         from moviepy.editor import ImageSequenceClip
#         fps=30
#         clip = ImageSequenceClip(frame_list, fps=fps)
#         clip.write_gif(f'{scenario_name}.gif', fps=fps)


if __name__ == "__main__":

    # ==== INFRASTRUCTURE TODOs ====
    # TODO: Fix observation dimensions
    # TODO: Implement initial passenger MLP model, get sim running
    # TODO: Implement initial mothership MAT model, get sim running
    # TODO:  Get mothership model output to passenger model input

    # ==== METHOD/DESIGN TODOs ====
    # TODO: Figure out mothership reward. Same as passengers'?
    # TODO: Figure out critic model(s) - same critic for all? Mask out to specific passengers?
        # One for mothership, one for passengeres?

    # ==== EXPERIMENT TODOs ====
    # TODO for Class: Everything should be in place to run mixed-obs experiments

    # Hyperparameters
    train_device = "cpu" # @param {"type":"string"}
    vmas_device = "cpu" # @param {"type":"string"}
    num_envs = 8 # @param {"type":"integer"}

    # Load task configuration
    task_config_path = "mr_spec_control/conf/task/custom_vmas/discovery_mothership.yaml"
    task = CustomVmasTask.DISCOVERY_MOTHERSHIP.get_from_yaml(task_config_path)
    # task.config = {
    #     "max_steps": 100,
    #     "n_agents_holonomic": 4,
    #     "n_agents_diff_drive": 0,
    #     "n_agents_car": 0,
    #     "lidar_range": 0,
    #     "comms_rendering_range": comms_radius, # Changed
    #     "shared_rew": False,
    # }

    # Load RL algorithm config
    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = MappoConfig.get_from_yaml()

    # Load policy model configs
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Load experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()

    # Override config parameters
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = train_device

    experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
    experiment_config.gamma = 0.99
    experiment_config.on_policy_collected_frames_per_batch = 60_000 # Number of frames collected each iteration
    experiment_config.on_policy_n_envs_per_worker = 600 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
    experiment_config.on_policy_n_minibatch_iters = 45
    experiment_config.on_policy_minibatch_size = 4096
    experiment_config.evaluation = True
    experiment_config.render = True
    experiment_config.share_policy_params = True # Policy parameter sharing on
    experiment_config.evaluation_interval = 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
    experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation
    experiment_config.loggers = ["csv"] # Log to csv, usually you should use wandb

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    experiment.algorithm.group_map["mothership"] = ["mothership"]
    experiment.algorithm.group_map["passengers"] = ["passenger_0", "passenger_1", "passenger_2", "passenger_3"]

    # exp_json_file = str(Path(experiment.folder_name) / Path(experiment.name + ".json"))

    # Run experiment
    experiment.run()

    # # Render the environment
    # use_vmas_env(
    #     render=True,
    #     num_envs=num_envs,
    #     n_steps=100,
    #     device=vmas_device,
    #     scenario=MyScenario(),
    #     continuous_actions=True,
    #     # Scenario kwargs
    #     )
