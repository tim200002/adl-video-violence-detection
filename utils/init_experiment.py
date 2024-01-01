import logging
import os
import sys
import dill

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

    cwd = os.getcwd()
    config.study_path = "sqlite:///" + os.path.join(cwd, base_experiment_path, f"{config.experiment_name}.db")
    
    # Take config and save it as a txt file with nice formatting
    with open(os.path.join(base_experiment_path, "config.txt"), "w") as f:
        # extract all variables from config
        for key, value in vars(config).items():
            # check if value is a class
            if hasattr(value, "__dict__"):
                continue
            # check if private i.e. starts with _
            elif key.startswith("_"):
                continue
            else:
                f.write(f"{key}: {value}\n")
    
    # also pickle config
    with open(os.path.join(base_experiment_path, "config.pkl"), "wb") as f:
        dill.dump(config, f)


    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(config.log_path),
        logging.StreamHandler(sys.stdout)
    ])
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    logging.info(f"Start experiment: {config.experiment_name}")


def resume_experiment(resume_path):
    if not os.path.exists(resume_path):
        raise ValueError(f"Resume path {resume_path} does not exist.")
    
    # load config
    with open(os.path.join(resume_path, "config.pkl"), "rb") as f:
        config = dill.load(f)

    
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(config.log_path, mode="a"),
        logging.StreamHandler(sys.stdout)
    ])
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    logging.info(f"Resume experiment: {config.experiment_name}")

    return config