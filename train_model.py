from sklearn.base import clone as sklearn_clone
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

CHOOSEN_MODEL = "catboost"  # "catboost", "xgboost", "randomforest"
ALL_MODELS = True

# ----------------------------------------------------------------
# Preprocessing functions
# ----------------------------------------------------------------

# Preprocessing functions


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
    X["country_US"] = (X["country"] == "US").astype(int)
    return X


def count_countries(X):
    """create a column 'country_count' that counts the number of countries listed
    in the 'country' column.
    """
    X = X.copy()
    X["country_count"] = (
        X["country"]
        .fillna("")
        .apply(lambda val: len(str(val).split(",")) if str(val).strip() != "" else 0)
    )
    return X


def get_main_countries(X, top_n=7):
    """
    Renvoie les top N pays principaux les plus fréquents.
    Le pays principal est le premier dans la chaîne 'country' avant la virgule.
    """
    # Extraire pays principal
    country_main = (
        X["country"]
        .apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")
        .astype(str)
        .str.strip()
    )
    top_countries = country_main.value_counts().nlargest(top_n).index.tolist()

    return top_countries


def add_country_dummies(X, top_countries):
    """
    Crée des dummies pour les pays principaux présents dans top_countries.
    Les pays non présents sont regroupés en 'Other'.
    """
    X = X.copy()
    X["country_main"] = (
        X["country"]
        .apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")
        .astype(str)
        .str.strip()
    )

    # mettre les autres pays à 'Other'
    X["country_main"] = X["country_main"].apply(
        lambda c: c if c in top_countries else "Other"
    )
    dummies = pd.get_dummies(X["country_main"], prefix="country", dtype=int)
    X = pd.concat([X, dummies], axis=1)

    return X, list(dummies.columns)


def add_top_company_features(X, top_revenue_companies):
    """
    Ajoute deux colonnes :
    - 'num_top_companies' : nombre de studios du film présents dans la liste top_revenue_companies
    - 'is_top_company' : 1 si au moins un studio du film est dans la liste, sinon 0
    """
    X = X.copy()

    def count_star_companies(company_str):
        if not isinstance(company_str, str) or company_str.strip() == "":
            return 0
        companies = [c.strip() for c in company_str.split(",")]
        return sum(c in top_revenue_companies for c in companies)

    # Nombre de studios prestigieux dans chaque film
    X["num_top_companies"] = X["company"].fillna("").apply(count_star_companies)

    # Variable binaire si au moins un studio prestigieux
    X["is_top_company"] = (X["num_top_companies"] > 0).astype(int)

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


def add_star_actor_features(X, star_actors):
    """
    Ajoute deux colonnes :
    - 'num_stars' : nombre d'acteurs prestigieux présents dans le casting du film
    - 'has_stars' : 1 si au moins un acteur prestigieux est présent
    """
    X = X.copy()

    def count_stars(actor_str):
        if not isinstance(actor_str, str) or actor_str.strip() == "":
            return 0
        actors = [a.strip() for a in actor_str.split(",")]
        return sum(a in star_actors for a in actors)

    # Nombre d'acteurs prestigieux par film
    X["num_stars"] = X["cast"].fillna("").apply(count_stars)

    # Présence ou non d'au moins un acteur prestigieux
    X["has_stars"] = (X["num_stars"] > 0).astype(int)

    return X


def extract_release_year(X, date_column="date"):
    """
    Extract release year from date column and correct future years
    """
    X = X.copy()
    X["date_format"] = pd.to_datetime(
        X[date_column], format="%m/%d/%y", errors="coerce"
    )
    # corriger le problème des années futures
    mask_future = X["date_format"].dt.year > 2025
    X.loc[mask_future, "date_format"] -= pd.offsets.DateOffset(years=100)
    X["release_year"] = X["date_format"].dt.year

    return X


def preprocess_data(X_train, X_test, y_train):
    """
    We use this function to apply previous preprocessing functions on both train and test sets
    """

    X_train = budget_missing_indicator(X_train)
    X_test = budget_missing_indicator(X_test)

    X_train = budget_0_to_nan(X_train)
    X_test = budget_0_to_nan(X_test)

    X_train = collection_to_binary(X_train)
    X_test = collection_to_binary(X_test)

    # X_train = US_to_binary(X_train)
    # X_test = US_to_binary(X_test)

    selected_genres = get_frequent_genres(X_train, min_occurrence=150)
    X_train = encode_genres(X_train, selected_genres)
    X_test = encode_genres(X_test, selected_genres)

    X_train = log_budget(X_train)
    X_test = log_budget(X_test)

    X_train = count_countries(X_train)
    X_test = count_countries(X_test)

    top_countries = get_main_countries(X_train, top_n=6)
    X_train, created_cols = add_country_dummies(X_train, top_countries)
    X_test, _ = add_country_dummies(X_test, top_countries)

    X_train = log_popularity(X_train)
    X_test = log_popularity(X_test)

    X_train = english_to_binary(X_train)
    X_test = english_to_binary(X_test)

    # Interaction features
    X_train["budget_x_popularity"] = (
        X_train["budget_nonzero"] * X_train["popularity_score"]
    )
    X_test["budget_x_popularity"] = (
        X_test["budget_nonzero"] * X_test["popularity_score"]
    )

    X_train["budget_nonzero_length_ratio"] = X_train["budget_nonzero"] / X_train[
        "length"
    ].replace(0, np.nan)
    X_test["budget_nonzero_length_ratio"] = X_test["budget_nonzero"] / X_test[
        "length"
    ].replace(0, np.nan)

    X_train["length_x_popularity"] = X_train["length"] * X_train["popularity_score"]
    X_test["length_x_popularity"] = X_test["length"] * X_test["popularity_score"]

    X_train = extract_release_year(X_train)
    X_test = extract_release_year(X_test)

    top_revenue_companies = [
        "Walt Disney Pictures",
        "Marvel Studios",
        "Lucasfilm",
        "20th Century Studios",
        "Warner Bros. Pictures",
        "Universal Pictures",
        "Paramount Pictures",
        "Columbia Pictures",
        "Legendary Entertainment",
        "Pixar Animation Studios",
    ]

    X_train = add_top_company_features(X_train, top_revenue_companies)
    X_test = add_top_company_features(X_test, top_revenue_companies)

    star_actors = [
        "Charlie Chaplin",
        "Clark Gable",
        "Humphrey Bogart",
        "James Stewart",
        "Cary Grant",
        "Marlon Brando",
        "Robert De Niro",
        "Al Pacino",
        "Jack Nicholson",
        "Clint Eastwood",
        "Harrison Ford",
        "Sylvester Stallone",
        "Arnold Schwarzenegger",
        "Tom Hanks",
        "Tom Cruise",
        "Leonardo DiCaprio",
        "Brad Pitt",
        "Johnny Depp",
        "Will Smith",
        "Denzel Washington",
        "George Clooney",
        "Robert Downey Jr.",
        "Chris Hemsworth",
        "Ryan Reynolds",
        "Joaquin Phoenix",
        "Adam Driver",
        "Timothée Chalamet",
        "Chris Evans",
        "Mark Ruffalo",
        "Samuel L. Jackson",
        "Marilyn Monroe",
        "Audrey Hepburn",
        "Katharine Hepburn",
        "Grace Kelly",
        "Elizabeth Taylor",
        "Ingrid Bergman",
        "Meryl Streep",
        "Jane Fonda",
        "Jodie Foster",
        "Sigourney Weaver",
        "Julia Roberts",
        "Nicole Kidman",
        "Cate Blanchett",
        "Sandra Bullock",
        "Cameron Diaz",
        "Angelina Jolie",
        "Charlize Theron",
        "Scarlett Johansson",
        "Jennifer Lawrence",
        "Emma Stone",
    ]
    X_train = add_star_actor_features(X_train, star_actors)
    X_test = add_star_actor_features(X_test, star_actors)

    # Final feature set
    feature_columns = (
        [
            "popularity_score",
            "budget_nonzero",
            "budget_is_zero",
            "has_collection",
            "language",
            "length",
            "country_count",
            "is_top_company",
            "has_stars",
            "budget_nonzero_length_ratio",
            "release_year",
        ]
        + selected_genres
        + created_cols
    )

    X_train_final = X_train[feature_columns]
    X_test_final = X_test[feature_columns]

    # Transform target variable
    y_train_log = np.log10(1 + y_train)

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
            # iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            l2_leaf_reg=8,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
        )
        model.fit(train_pool)

    elif model_name == "xgboost":
        model = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.05,
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

    elif model_name == "randomforest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    else:
        raise ValueError(
            "Unsupported model_name. Choose 'catboost', 'xgboost', or 'randomforest'"
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


def compare_models(X, y):
    y = y.values.ravel()
    models = {
        "CatBoost": CatBoostRegressor(
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            l2_leaf_reg=8,
            random_seed=2,
            verbose=0,
            early_stopping_rounds=50,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=3000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=2,
            n_jobs=-1,
            early_stopping_rounds=50,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=2, n_jobs=-1
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model_template in models.items():
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

            model = sklearn_clone(model_template)

            # 2. Créer les jeux de train/validation pour CE fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if name == "CatBoost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

            elif name == "XGBoost":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0,
                )

            elif name == "RandomForest":
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_pred = y_pred.clip(0, None)

            score = mean_squared_error(y_val, y_pred)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        print(f"{name} MSLE mean: {mean_score} over 5 folds")


# ----------------------------------------------------------------

if __name__ == "__main__":
    X_train = pd.read_csv("data/challenge_train_features.csv", index_col=0)
    y_train = pd.read_csv("data/challenge_train_revenue.csv", index_col=0)
    X_test = pd.read_csv("data/challenge_test_features.csv", index_col=0)

    # X_train, y_train = remove_outliers(X_train, y_train)

    # Preprocess data
    X_train_processed, X_test_processed, y_train_log = preprocess_data(
        X_train, X_test, y_train
    )
    if ALL_MODELS:
        for model_name in ["catboost", "xgboost", "randomforest"]:
            print(f"Training and evaluating model: {model_name}")
            model = train_model(X_train_processed, y_train_log, model_name=model_name)
            y_pred = model.predict(X_test_processed)

            y_pred = (10**y_pred) - 1
            # Save submission
            save_submission(y_pred, filename=f"{model_name}.txt")

            # evaluate_model(X_train_processed, y_train_log, model, model_name)

    else:
        # Train model
        model = train_model(X_train_processed, y_train_log, model_name=CHOOSEN_MODEL)
        y_pred = model.predict(X_test_processed)
        y_pred = (10**y_pred) - 1

        # Save submission
        save_submission(y_pred, filename=f"{CHOOSEN_MODEL}.txt")

    # Evaluate models
    print("Evaluating models with 5-Fold Cross-Validation:")
    compare_models(X_train_processed, y_train_log)
