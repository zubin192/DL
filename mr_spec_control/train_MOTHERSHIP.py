
import os
from pathlib import Path

from benchmarl.algorithms import EnsembleAlgorithmConfig, MappoConfig, IppoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import EnsembleModelConfig, MlpConfig
from environments.custom_vmas.common import CustomVmasTask


if __name__ == "__main__":

    # ==== INFRASTRUCTURE TODOs ====
    # TODO: Implement initial mothership MAT model, get sim running
    # TODO: Format mothership action output

    # ==== METHOD/DESIGN TODOs ====
    # TODO: Figure out mothership reward. Same as passengers'?
    # TODO: Figure out critic model(s) - same critic for all? Mask out to specific passengers?
        # One for mothership, one for passengeres?


    # Hyperparameters
    train_device = "cuda" # @param {"type":"string"}
    vmas_device = "cuda" # @param {"type":"string"}

    # Load task configuration
    task_config_path = "mr_spec_control/conf/task/custom_vmas/tasks_comms.yaml"
    task = CustomVmasTask.TASKS_COMMS.get_from_yaml(task_config_path)

    # Load RL algorithm config
    algorithm_config = IppoConfig.get_from_yaml()
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
    model_config_path = "mr_spec_control/conf/models/custom_mlp.yaml"
    model_config = MlpConfig.get_from_yaml(model_config_path)

    critic_model_config = MlpConfig.get_from_yaml(model_config_path)
    # critic_model_config.centralised = True
    # critic_model_config.has_agent_input_dim = True

    # Load experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()

    # Override config parameters
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = train_device

    experiment_config.max_n_frames = 1_000_000 # Number of frames before training ends
    experiment_config.gamma = 0.99
    experiment_config.on_policy_collected_frames_per_batch = 3_000 # Number of frames collected each iteration (max_steps from config * n_envs_per_worker)
    experiment_config.on_policy_n_envs_per_worker = 30 # Number of vmas vectorized enviornemnts (each will collect max_steps steps, see max_steps in task_config -> 50 * max_steps = 5_000 the number above)
    experiment_config.on_policy_n_minibatch_iters = 32
    # experiment_config.on_policy_minibatch_size = 64

    experiment_config.evaluation = True
    experiment_config.render = True
    experiment_config.share_policy_params = False # Policy parameter sharing
    experiment_config.evaluation_interval = 6_000 # Interval in terms of frames, will evaluate every eval_interval/frames_per_batch iterations
    experiment_config.evaluation_episodes = 30 # Number of vmas vectorized enviornemnts used in evaluation

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
