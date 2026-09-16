import logging

import pandas as pd
import yaml

from data.dataloader import run_dataloader, split_train_val_test_sets
from data.preprocessing import fit_preprocessor, transform_preprocessor
from models.train import train_base_model, train_best_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PATH_CONFIG_PATH = "config/config_paths.yaml"
PATH_CONFIG_MODELING = "config/config_modeling.yaml"
PATH_CONFIG_PROCESSING = "config/config_processing.yaml"
MODEL_NAME = "xgboost"


def run_pipeline_training(
    path_config_path, path_config_processing, path_config_modeling, model_name=MODEL_NAME
):
    with open(path_config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(path_config_processing, "r") as f:
        config_processing = yaml.safe_load(f)
        config.update(config_processing)
    with open(path_config_modeling, "r") as f:
        config_modeling = yaml.safe_load(f)
        config.update(config_modeling)
    LOGGER.info("Config files loaded")

    X_initial, y_initial, _ = run_dataloader(config)

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_train_val_test_sets(
        X_initial, y_initial
    )
    preprocessor = fit_preprocessor(X_train_raw, config)
    X_train = transform_preprocessor(X_train_raw, config, preprocessor)
    X_val = transform_preprocessor(X_val_raw, config, preprocessor)
    LOGGER.info("Train/validation/test split and preprocessing done")

    # Train the base model and set alias base_model.
    train_base_model(
        X_train,
        y_train,
        config,
        model_name,
        registry_name=config["models"][model_name]["registry_name"],
        alias="base_model",
    )
    LOGGER.info("Base model trained")
    # Hyperparameter tuning and best model registration.
    train_best_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config,
        model_name,
        registry_name=config["models"][model_name]["registry_name"],
        alias="champion",
        n_trials=20,
    )
    LOGGER.info("Model train with optimized hyperparameters")


if __name__ == "__main__":
    run_pipeline_training(
        path_config_path=PATH_CONFIG_PATH,
        path_config_processing=PATH_CONFIG_PROCESSING,
        path_config_modeling=PATH_CONFIG_MODELING,
    )
