import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_log_error

CHOOSEN_MODEL = "catboost"  # "catboost", "xgboost", "histgradientboosting"
# ----------------------------------------------------------------
# Preprocessing functions
# ----------------------------------------------------------------

# Preprocessing functions


# Clip popularity scores at 20
def clip_popularity(X):
    X = X.copy()
    X["popularity_score"] = X["popularity_score"].clip(upper=20)
    return X


# Create a binary indicator for zero budget (nan values)
def budget_missing_indicator(X):
    X = X.copy()
    X["budget_is_zero"] = (X["budget"] == 0).astype(int)
    return X


def log_budget(X):
    X = X.copy()
    X["budget"] = np.log1p(X["budget"].clip(lower=0))
    return X


def log_popularity(X):
    X = X.copy()
    X["popularity_score"] = np.log1p(X["popularity_score"].clip(lower=0))
    return X


def budget_0_to_nan(X):
    X = X.copy()
    X["budget_nonzero"] = X["budget"].replace(0, np.nan)
    return X


def collection_to_binary(X):
    X = X.copy()
    X["has_collection"] = X["collection"].notna().astype(int)
    return X


def english_to_binary(X):
    """
    Convert language column to binary (0/1) where:
    1 = movie is in English ('en')
    0 = movie is in another language
    """
    X = X.copy()
    X["language"] = (X["language"] == "en").astype(int)
    return X


def US_to_binary(X):
    """
    Convert country column to binary (0/1) where:
    1 = movie is from US
    0 = movie is from another country
    """
    X = X.copy()
    X["country"] = (X["country"] == "US").astype(int)
    return X


def get_frequent_genres(X, min_occurrence=150):
    """
    Get list of genres that appear more than min_occurrence times
    return a list of frequent genres + "Other"
    """
    X = X.copy()
    X["genres_list"] = X["genre"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    genre_counts = X["genres_list"].explode().value_counts()
    selected_genres = genre_counts[genre_counts >= min_occurrence].index.tolist()
    return selected_genres + ["Other"]


def encode_genres(X, selected_genres):
    """
    Encode genres into binary columns based on a predefined list of genres
    """
    X = X.copy()
    # Split genres into lists
    X["genres_list"] = X["genre"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )

    # Map less frequent genres to "Other"
    X["genres_list"] = X["genres_list"].apply(
        lambda lst: [g if g in selected_genres else "Other" for g in lst]
    )

    # Create binary columns for each genre
    for genre in selected_genres:
        X[genre] = X["genres_list"].apply(lambda lst: int(genre in lst))

    return X


def add_language_count(X):
    """
    create a column 'language_count' that counts the number of languages listed
    in the 'language' column.
    """
    X = X.copy()

    X["language_count"] = (
        X["language"]
        .fillna("")
        .apply(lambda val: len(str(val).split(",")) if str(val).strip() != "" else 0)
    )
    return X


def preprocess_data(X_train, X_test, y_train):
    """
    We use this function to apply previous preprocessing functions on both train and test sets
    """
    X_train = clip_popularity(X_train)
    X_test = clip_popularity(X_test)

    X_train = budget_missing_indicator(X_train)
    X_test = budget_missing_indicator(X_test)

    X_train = budget_0_to_nan(X_train)
    X_test = budget_0_to_nan(X_test)

    X_train = collection_to_binary(X_train)
    X_test = collection_to_binary(X_test)

    X_train = english_to_binary(X_train)
    X_test = english_to_binary(X_test)

    X_train = US_to_binary(X_train)
    X_test = US_to_binary(X_test)

    selected_genres = get_frequent_genres(X_train, min_occurrence=150)
    X_train = encode_genres(X_train, selected_genres)
    X_test = encode_genres(X_test, selected_genres)

    X_train = log_budget(X_train)
    X_test = log_budget(X_test)

    X_train = log_popularity(X_train)
    X_test = log_popularity(X_test)

    # Interaction features
    # X_train['budget_x_popularity'] = X_train['budget'] * X_train['popularity_score']
    # X_test['budget_x_popularity'] = X_test['budget'] * X_test['popularity_score']

    # X_train['has_collection_x_budget'] = X_train['has_collection'] * X_train['budget']
    # X_test['has_collection_x_budget'] = X_test['has_collection'] * X_test['budget']

    # X_train['length_x_popularity'] = X_train['length'] * X_train['popularity_score']
    # X_test['length_x_popularity'] = X_test['length'] * X_test['popularity_score']

    X_train = add_language_count(X_train)
    X_test = add_language_count(X_test)

    # Final feature set
    feature_columns = [
        "popularity_score",
        "budget",
        "budget_is_zero",
        "has_collection",
        "language",
        "country",
        "length",
        "language_count",
    ] + selected_genres
    X_train_final = X_train[feature_columns]
    X_test_final = X_test[feature_columns]

    # Transform target variable
    y_train_log = np.log1p(y_train)

    return X_train_final, X_test_final, y_train_log


# ----------------------------------------------------------------
# TRAINING MODELS
# ----------------------------------------------------------------
def train_model(X_train, y_train, model_name="catboost"):
    """
    Train a regression model (CatBoost or XGBoost) on the training data
    """
    if model_name == "catboost":
        train_pool = Pool(X_train, y_train)
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            l2_leaf_reg=4,
            eval_metric="MSLE",
            random_seed=2,
            verbose=100,
        )
        model.fit(train_pool)

    elif model_name == "xgboost":
        model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    elif model_name == "histgradientboosting":
        model = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.075, max_iter=200, random_state=42
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(
            "Unsupported model_name. Choose 'catboost', 'xgboost', or 'histgradientboosting'"
        )

    return model


# ----------------------------------------------------------------
# save predictions on a submission file
# ----------------------------------------------------------------


def save_submission(y_pred, filename="test.txt"):
    """
    Save predictions to a txt file for submission
    """
    pred_str = ",".join([str(int(p)) for p in y_pred])
    with open(filename, "w") as f:
        f.write(pred_str)


# ----------------------------------------------------------------
# Model evaluation
# ----------------------------------------------------------------


def evaluate_model(X, y, model, name):
    scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, scoring=scorer, cv=cv)
    print(f"{name} RMSLE mean: {np.sqrt(scores.mean()):.4f}")


# ----------------------------------------------------------------

if __name__ == "__main__":
    X_train = pd.read_csv("data/challenge_train_features.csv", index_col=0)
    y_train = pd.read_csv("data/challenge_train_revenue.csv", index_col=0)
    X_test = pd.read_csv("data/challenge_test_features.csv", index_col=0)

    # Preprocess data
    X_train_processed, X_test_processed, y_train_log = preprocess_data(
        X_train, X_test, y_train
    )

    # Train model
    model = train_model(X_train_processed, y_train_log, model_name=CHOOSEN_MODEL)
    y_pred = model.predict(X_test_processed)
    y_pred = np.expm1(y_pred)

    # Save submission
    save_submission(y_pred, filename=f"{CHOOSEN_MODEL}.csv")

    # Evaluate models
    print("Evaluating models with 5-Fold Cross-Validation:")
    evaluate_model(X_train_processed, y_train_log, model, CHOOSEN_MODEL)
