import os
from pathlib import Path
import time
import torch
import wandb

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import CnnConfig
from environments.custom_vmas.common import CustomVmasTask

wandb.login()

if __name__ == "__main__":
    # Device settings
    #eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_device = "cpu"

    vmas_device = eval_device

    # Load task configuration
    task_config_path = "mr_spec_control/conf/task/custom_vmas/discovery_mothership_CNN.yaml"
    task = CustomVmasTask.DISCOVERY_OBSTACLES.get_from_yaml(task_config_path)

    # Load RL algorithm configuration
    algorithm_config = MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.001,
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=True,
    )

    # Load policy model configurations
    cnn_config_path = "mr_spec_control/conf/models/custom_cnn.yaml"
    model_config = CnnConfig.get_from_yaml(cnn_config_path)
    critic_model_config = CnnConfig.get_from_yaml(cnn_config_path)

    # Load experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = eval_device

    # Modify evaluation-specific parameters
    experiment_config.evaluation = True
    experiment_config.render = True  # Enable rendering for visual assessment
    experiment_config.evaluation_episodes = 20  # Number of evaluation episodes
    experiment_config.evaluation_interval = 5 * experiment_config.on_policy_collected_frames_per_batch
    experiment_config.loggers = ["wandb"]  # Enable W&B logging
    experiment_config.project_name = "mr_spec_control_eval"

    # Load trained policy checkpoint
    policy_path = "/home/ajay-ratty/Desktop/mr-specialization-control/mr_spec_control/runs/cnn_888/checkpoints/best.pt"
    if os.path.exists(policy_path):
        checkpoint = torch.load(policy_path, map_location=eval_device)
        print(f"Loaded policy from {policy_path}")
    else:
        raise FileNotFoundError(f"Policy checkpoint not found at {policy_path}")

    # Initialize experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    # Retrieve environment from experiment
    env = experiment.test_env

    @torch.no_grad()
    def evaluation_loop():
        evaluation_start = time.time()

        video_frames = [] if experiment.task.has_render(env) and experiment.config.render else None

        def callback(env, td):
            if video_frames is not None:
                video_frames.append(experiment.task.__class__.render_callback(experiment, env, td))

        if env.batch_size == ():
            rollouts = [env.rollout(
                max_steps = experiment.max_steps,
                policy=experiment.policy,
                callback=callback if i == 0 else None,
                auto_cast_to_device=True,
                break_when_any_done=False,
            ) for i in range(experiment.config.evaluation_episodes)]
        else:
            rollouts = env.rollout(
                max_steps = experiment.max_steps,
                policy=experiment.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            rollouts = list(rollouts.unbind(0))

        evaluation_time = time.time() - evaluation_start
        experiment.logger.log({"timers/evaluation_time": evaluation_time}, step=experiment.n_iters_performed)
        experiment.logger.log_evaluation(rollouts, video_frames=video_frames, step=experiment.n_iters_performed, total_frames=experiment.total_frames)
        experiment._on_evaluation_end(rollouts)

    print("Running evaluation loop...")
    print("Rendering test...")
    for _ in range(10):  # Render the environment before evaluation
        env.render(mode='rgb_array')
        time.sleep(1)
    print("Rendering test complete.")
    evaluation_loop()
