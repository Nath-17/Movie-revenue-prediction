import logging

import pandas as pd
import yaml

from data.dataloader import run_dataloader, split_train_val_test_sets
from data.preprocessing import fit_preprocessor, transform_preprocessor
from models.predict_evaluate import predict, save_predictions, error_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PATH_CONFIG_PATH = "config/config_paths.yaml"
PATH_CONFIG_MODELING = "config/config_modeling.yaml"
PATH_CONFIG_PROCESSING = "config/config_processing.yaml"
OUTPUT_PATH = "data/outputs/predictions"
REGISTRY_NAME = "XGBoostModel"
MODEL_ALIAS = "champion"

def run_pipeline_predicting_evaluating(
    path_config_path, path_config_processing, path_config_modeling, model_name=REGISTRY_NAME,model_alias=MODEL_ALIAS, output_path = OUTPUT_PATH
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
    X_test = transform_preprocessor(X_test_raw, config, preprocessor)
    LOGGER.info("Train/validation/test split and preprocessing done")
    
    df_predictions = predict(X_test, model_name, model_alias)
    saved_predictions_file_path = save_predictions(df_predictions, output_path)
    LOGGER.info(f"Preditions done with {model_name} ({model_alias}).")
    LOGGER.info(f"Predictions saved at: {saved_predictions_file_path}")


if __name__ == "__main__":
        run_pipeline_predicting_evaluating(
        path_config_path=PATH_CONFIG_PATH,
        path_config_processing=PATH_CONFIG_PROCESSING,
        path_config_modeling=PATH_CONFIG_MODELING,
    )

