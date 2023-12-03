import logging
import os
import sys

import optuna

def init_experiment(config):
    base_path = "experiments"
    base_experiment_path = os.path.join(base_path, config.experiment_name)

    # create folder if exists otherwise create folder but add a number to the folder name
    if not os.path.exists(base_experiment_path):
        os.makedirs(base_experiment_path)
    else:
        i = 1
        while os.path.exists(base_experiment_path + f"_{i}"):
            i += 1
        base_experiment_path = base_experiment_path + f"_{i}"
        os.makedirs(base_experiment_path)
    
    # create subfolders
    config.checkpoint_path= os.path.join(base_experiment_path, "checkpoints")
    os.makedirs(config.checkpoint_path)

    config.log_path = os.path.join(base_experiment_path, "log.log")

    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(config.log_path),
        logging.StreamHandler(sys.stdout)
    ])
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.