from typing import Any

import mlflow
import mlflow.xgboost
import optuna
import yaml
from xgboost import XGBRegressor

from models.model_registry import register_model_with_alias

REGISTRY_NAME = "XGBoostModel"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def get_base_params(config: dict, model_name: str) -> dict:
    return config["models"][model_name]["parameters"]

def get_grid_params(config: dict, model_name: str) -> dict:
    return config["models"][model_name]["grid_parameters"]

def make_params_from_grid(trial: optuna.Trial, grid: dict[str, list[Any]]) -> dict:
    params = {}
    for key, values in grid.items():
        if isinstance(values, list):
            params[key] = trial.suggest_categorical(key, values)
        else:
            params[key] = values
    return params


def objective(trial: optuna.Trial, config: dict, model_name: str,
              X_train, y_train, X_test, y_test):
    base_params = get_base_params(config, model_name)
    grid_params = get_grid_params(config, model_name)
    sampled = make_params_from_grid(trial, grid_params)
    params = {**base_params, **sampled}

    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

def run_optimization(config: dict, model_name: str,
                     X_train, y_train, X_test, y_test,
                     n_trials: int = 50):
    study = optuna.create_study(direction="maximize")

    # One top-level MLflow run for the full Optuna study.
    with mlflow.start_run(run_name=f"{model_name}_optuna_tuning"):
        study.optimize(
            lambda trial: objective(trial, config, model_name, X_train, y_train, X_test, y_test),
            n_trials=n_trials,
        )

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_score", study.best_value)

    return study.best_trial.params




def train_base_model(X_train, y_train, config: dict, model_name: str,
                     registry_name: str = REGISTRY_NAME, alias: str = "base_model"):
    params = get_base_params(config, model_name)

    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    register_model_with_alias(model, registry_name, alias, artifact_path="base_model")
    return model


def train_best_model(
    X_train,
    y_train,
    X_test,
    y_test,
    config: dict,
    model_name: str,
    registry_name: str = REGISTRY_NAME,
    alias: str = "champion",
    n_trials: int = 20,
):
    best_params = run_optimization(config, model_name, X_train, y_train, X_test, y_test, n_trials=n_trials)

    # Train a final model with the best parameters.
    best_model = XGBRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    register_model_with_alias(best_model, registry_name, alias, artifact_path="best_model")

    return best_model, best_params
