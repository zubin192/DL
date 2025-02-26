
import os
from pathlib import Path

from benchmarl.algorithms import EnsembleAlgorithmConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import EnsembleModelConfig, MlpConfig, CnnConfig
from environments.custom_vmas.common import CustomVmasTask
from models.joint_models import JointModelsConfig


if __name__ == "__main__":

    # ==== INFRASTRUCTURE TODOs ====
    # TODO: Implement initial passenger MLP model, get sim running
    # TODO: Implement initial mothership MAT model, get sim running
    # TODO: Format mothership action output

    # ==== METHOD/DESIGN TODOs ====
    # TODO: Figure out mothership reward. Same as passengers'?
    # TODO: Figure out critic model(s) - same critic for all? Mask out to specific passengers?
        # One for mothership, one for passengeres?


    # Hyperparameters
    train_device = "cuda" # @param {"type":"string"}
    vmas_device = "cuda" # @param {"type":"string"}
    num_envs = 8 # @param {"type":"integer"}

    # Load task configuration
    task_config_path = "mr_spec_control/conf/task/custom_vmas/discovery_mothership.yaml"
    task = CustomVmasTask.DISCOVERY_OBSTACLES.get_from_yaml(task_config_path)
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
    # algorithm_config = EnsembleAlgorithmConfig(
    #     {"mothership": MappoConfig.get_from_yaml(),
    #      "passenger": MappoConfig.get_from_yaml()
    #     }
    # )
    algorithm_config = MappoConfig.get_from_yaml()
    # algorithm_config = MappoConfig(
    #     share_param_critic=True, # Critic param sharing on
    #     clip_epsilon=0.2,
    #     entropy_coef=0.001, # We modify this, default is 0
    #     critic_coef=1,
    #     loss_critic_type="l2",
    #     lmbda=0.9,
    #     scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
    #     use_tanh_normal=True,
    #     minibatch_advantage=False,
    # )

    # Load policy model configs
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    # model_config_path = "mr_spec_control/conf/models/joint_models.yaml"
    # JointModelsConfig.get_from_yaml(model_config_path)
    # model_config = EnsembleModelConfig(
    #     {"mothership": MlpConfig.get_from_yaml(),
    #      "passenger": MlpConfig.get_from_yaml()
    #      }
    # )
    model_config = CnnConfig.get_from_yaml()
    # model_config = MlpConfig.get_from_yaml()
    # model_config = MlpConfig(
    #     num_cells=[256, 256], # Two layers with 256 neurons each
    #     layer_class=torch.nn.Linear,
    #     activation_class=torch.nn.Tanh,
    # )

    critic_model_config = CnnConfig.get_from_yaml()
    # critic_model_config = MlpConfig.get_from_yaml()
    # critic_model_config = MlpConfig(
    #     num_cells=[256, 256], # Two layers with 256 neurons each
    #     layer_class=torch.nn.Linear,
    #     activation_class=torch.nn.Tanh,
    # )


    # Load experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()

    # Override config parameters
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = train_device

    experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
    experiment_config.gamma = 0.99
    experiment_config.on_policy_collected_frames_per_batch = 20_000 # Number of frames collected each iteration (max_steps from config * n_envs_per_worker)
    experiment_config.on_policy_n_envs_per_worker = 100 # Number of vmas vectorized enviornemnts (each will collect max_steps steps, see max_steps in task_config -> 200 * max_steps = 50_000 the number above)
    experiment_config.on_policy_n_minibatch_iters = 45
    experiment_config.on_policy_minibatch_size = 4096
    experiment_config.evaluation = True
    experiment_config.render = True
    experiment_config.share_policy_params = True # Policy parameter sharing
    experiment_config.evaluation_interval = 60_000 # Interval in terms of frames, will evaluate every 60_000 / 20_000 = 3 iterations
    experiment_config.evaluation_episodes = 100 # Number of vmas vectorized enviornemnts used in evaluation
    experiment_config.loggers = ["csv"] # Log to csv, usually you should use wandb

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    # experiment.algorithm.group_map["mothership"] = ["mothership_0"]
    # experiment.algorithm.group_map["passengers"] = ["passenger_1", "passenger_2", "passenger_3", "passenger_4"]

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
