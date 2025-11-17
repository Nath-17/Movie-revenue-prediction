# tune_catboost_film.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
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
                learning_rate=cfg.learning_rate,
                depth=cfg.depth,
                l2_leaf_reg=cfg.l2_leaf_reg,
                loss_function="RMSE",
                random_seed=42,
                early_stopping_rounds=50,
                verbose=False,
            )

            model.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)

            y_val_pred_log = model.predict(X_val_f)

            score = mean_squared_error(y_val_f, y_val_pred_log)
            fold_scores.append(score)

        # mean score over 5 folds
        mean_score = np.mean(fold_scores)

        wandb.log({"msle": mean_score})
        print(f" CV MSLE = {mean_score}")


# ------------------ sweep configuraion ------------------
sweep_config = {
    "method": "bayes",
    "metric": {"name": "msle", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"distribution": "uniform", "min": 0.02, "max": 0.10},
        "depth": {"values": [4, 5, 6, 7]},
        "l2_leaf_reg": {"values": [1, 3, 6, 10]},
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)

    wandb.agent(sweep_id, function=train_and_log, count=40)
