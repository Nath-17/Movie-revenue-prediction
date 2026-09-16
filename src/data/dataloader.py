import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def load_data(local_path, online_path):
    if not os.path.exists(local_path):
        X_train = pd.read_csv(online_path + "challenge_train_features.csv", index_col=0)
        y_train = pd.read_csv(online_path + "challenge_train_revenue.csv", index_col=0)
        X_test = pd.read_csv(online_path + "challenge_test_features.csv", index_col=0)
        save_data(X_train, y_train, X_test, local_path)
    else:
        X_train = pd.read_csv(
            os.path.join(local_path, "challenge_train_features.csv"), index_col=0
        )
        y_train = pd.read_csv(
            os.path.join(local_path, "challenge_train_revenue.csv"), index_col=0
        )
        X_test = pd.read_csv(
            os.path.join(local_path, "challenge_test_features.csv"), index_col=0
        )
    return X_train, y_train, X_test


def save_data(X_train, y_train, X_test, path):
    if not os.path.exists(path):
        os.makedirs(path)
    X_train.to_csv(os.path.join(path, "challenge_train_features.csv"))
    y_train.to_csv(os.path.join(path, "challenge_train_revenue.csv"))
    X_test.to_csv(os.path.join(path, "challenge_test_features.csv"))
    return 0


def split_train_val_test_sets(
    X, y, test_size=0.2, validation_size=0.2, random_state=42
):
    """
    Split labeled data into train, validation, and test sets.

    ``test_size`` and ``validation_size`` are fractions of the full dataset.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_dataloader(config):
    LOGGER.info("Loading data...")
    X_train, y_train, X_test_unknown = load_data(
        config["local_inputs_data_path"], config["online_inputs_data_path"]
    )

    return X_train, y_train, X_test_unknown


if __name__ == "__main__":
    with open("config/config_paths.yaml", "r") as f:
        config_path = yaml.safe_load(f)
    X_train, y_train, X_test = run_dataloader(config_path)
