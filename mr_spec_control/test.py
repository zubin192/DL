from benchmarl.algorithms import EnsembleAlgorithmConfig, IppoConfig, MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, EnsembleModelConfig
from benchmarl.models import DeepsetsConfig, GnnConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas/simple_tag.yaml"
    task = VmasTask.SIMPLE_TAG.get_from_yaml()

    algorithm_config = EnsembleAlgorithmConfig(
        {"agent": MappoConfig.get_from_yaml(), "adversary": IppoConfig.get_from_yaml()}
    )

    model_config = EnsembleModelConfig(
        {"agent": MlpConfig.get_from_yaml(), "adversary": MlpConfig.get_from_yaml()}
    )
    critic_model_config = EnsembleModelConfig(
        {
            "agent": MlpConfig.get_from_yaml(),
            "adversary": MlpConfig.get_from_yaml(),
        }
    )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
