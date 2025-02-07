
import os
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from scenarios.discovery_mothership import Scenario

# Load experiment configuration
experiment_config = ExperimentConfig.get_from_yaml()

# Load task configuration
task = Scenario()

# Load RL algorithm config
# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()

# Load policy model configs
# Loads from "benchmarl/conf/model/layers/mlp.yaml"
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

# TODO: Clean up discovery constants
# TODO: Implement passenger MLP model
# TODO: Implement mothership MAT model
# TODO: Get mothership model output to passenger model input
# TODO: Figure out mothership reward. Same as passengers'?
# TODO: Figure out critic model(s) - same critic for all? Mask out to specific passengers?
    # One for mothership, one for passengeres?

# Final experiment setup
experiment_config.max_n_iters = 3
experiment_config.loggers = []

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)

exp_json_file = str(Path(experiment.folder_name) / Path(experiment.name + ".json"))

# Run experiment
experiment.run()
