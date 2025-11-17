# tune_catboost_film.py

import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from train_model import preprocess_data
from dotenv import load_dotenv
import os

# ------------------ init W&B ------------------
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)
PROJECT_NAME = "film-revenue-catboost-tuning"

# ------------------ data + preprocessing ------------------
X = pd.read_csv("data/challenge_train_features.csv", index_col=0)
y = pd.read_csv("data/challenge_train_revenue.csv", index_col=0).iloc[:, 0]
X_test = pd.read_csv("data/challenge_test_features.csv", index_col=0)

X_proc, X_test_proc, y_log = preprocess_data(X, X_test, y)


# ------------------ Metric officielle (base 10) ------------------
def rmsle_base10(y_true, y_pred):
    log_true = np.log10(y_true + 1)
    log_pred = np.log10(y_pred + 1)
    return np.sqrt(np.mean((log_pred - log_true) ** 2))


# ------------------ fonction W&B K-Fold ------------------
def train_and_log(config=None):
    with wandb.init(config=config, project=PROJECT_NAME):
        cfg = wandb.config

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc)):
            X_train_f = X_proc.iloc[train_idx]
            y_train_f = y_log.iloc[train_idx]

            X_val_f = X_proc.iloc[val_idx]
            y_val_f = y_log.iloc[val_idx]

            model = CatBoostRegressor(
                iterations=cfg.iterations,
                learning_rate=cfg.learning_rate,
                depth=cfg.depth,
                l2_leaf_reg=cfg.l2_leaf_reg,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            )

            model.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)

            # Prédiction log
            y_val_pred_log = model.predict(X_val_f)

            # Passage en échelle originale
            y_true = np.expm1(y_val_f)
            y_pred = np.expm1(y_val_pred_log).clip(0, None)

            # Score officiel
            score = rmsle_base10(y_true, y_pred)
            fold_scores.append(score)

        # Score moyen sur les 5 folds
        mean_score = np.mean(fold_scores)

        wandb.log({"rmsle_base10": mean_score})
        print(f" CV RMSLE_base10 = {mean_score:.5f}")


# ------------------ Configuration du sweep ------------------
sweep_config = {
    "method": "bayes",
    "metric": {"name": "rmsle_base10", "goal": "minimize"},
    "parameters": {
        "iterations": {"values": [600, 800, 1000, 1200]},
        "learning_rate": {"distribution": "uniform", "min": 0.02, "max": 0.10},
        "depth": {"values": [4, 5, 6, 7]},
        "l2_leaf_reg": {"values": [1, 3, 6, 10]},
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)

    wandb.agent(sweep_id, function=train_and_log, count=40)
