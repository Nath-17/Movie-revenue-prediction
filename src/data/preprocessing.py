import logging

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = "D:/Projets_code/Movie-revenue-prediction/config/config_processing.yaml"


def budget_missing_indicator(X):
    X = X.copy()
    X["budget_is_zero"] = (X["budget"] == 0).astype(int)
    return X


def log_budget(X):
    "convert budget in log"
    X = X.copy()
    X["budget"] = np.log1p(X["budget"].clip(lower=0))
    return X


def log_popularity(X):
    "convert popularity score in log"
    X = X.copy()
    X["popularity_score"] = np.log1p(X["popularity_score"].clip(lower=0))
    return X


def budget_0_to_nan(X):
    "replace zero budget with nan in a new column : budget_nonzero"
    X = X.copy()
    X["budget_nonzero"] = X["budget"].replace(0, np.nan)
    return X


def collection_to_binary(X):
    "create a binary column has_collection where 1 indicates the movie belongs to a collection"
    X = X.copy()
    X["has_collection"] = X["collection"].notna().astype(int)
    return X


def english_to_binary(X):
    """
    Convert language column to binary where:
    1 = movie is in English ('en')
    0 = movie is in another language
    """
    X = X.copy()
    X["language"] = (X["language"] == "en").astype(int)
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
    return N top countries based on frequency in the country column.
    The main country is the first in the country list.
    """
    # extract main country
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
    Create dummies for main countries present in top_countries.
    Countries not present are grouped into Other.
    """
    X = X.copy()
    X["country_main"] = (
        X["country"]
        .apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")
        .astype(str)
        .str.strip()
    )

    # set other countries to Other
    X["country_main"] = X["country_main"].apply(
        lambda c: c if c in top_countries else "Other"
    )
    dummies = pd.get_dummies(X["country_main"], prefix="country", dtype=int)
    X = pd.concat([X, dummies], axis=1)

    return X, list(
        dummies.columns
    )  # return also the list of created columns to add later to features list


def add_top_company_features(X, top_revenue_companies):
    """
    Adds two columns:
    num_top_companies : number of the movie's studios present in the top_revenue_companies list
    is_top_company: 1 if at least one of the movie's studios is in the list, otherwise 0
    """
    X = X.copy()

    def count_star_companies(company_str):
        if not isinstance(company_str, str) or company_str.strip() == "":
            return 0
        companies = [c.strip() for c in company_str.split(",")]
        return sum(c in top_revenue_companies for c in companies)

    # Number of prestigious studios in each movie
    X["num_top_companies"] = X["company"].fillna("").apply(count_star_companies)

    # Binary variable if at least 1 big studio
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
    Adds two columns:
    num_stars: number of prestigious actors present in the movie's cast
    has_stars: 1 if at least one prestigious actor is present
    """
    X = X.copy()

    def count_stars(actor_str):
        if not isinstance(actor_str, str) or actor_str.strip() == "":
            return 0
        actors = [a.strip() for a in actor_str.split(",")]
        return sum(a in star_actors for a in actors)

    # Number of prestigious actors in each movie
    X["num_stars"] = X["cast"].fillna("").apply(count_stars)

    # Presence or absence of at least one prestigious actor
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
    # correct issue with years > 2025
    mask_future = X["date_format"].dt.year > 2025
    X.loc[mask_future, "date_format"] -= pd.offsets.DateOffset(years=100)
    X["release_year"] = X["date_format"].dt.year

    return X


def fit_preprocessor(X, config):
    """Learn the category lists and feature columns from the training data."""
    number_countries = config["number_countries"]
    min_genre_occurrence = config["number_min_genre_occurrence"]

    top_countries = get_main_countries(X, top_n=number_countries)
    selected_genres = get_frequent_genres(
        X, min_occurrence=min_genre_occurrence
    )
    country_columns = [f"country_{country}" for country in top_countries]
    country_columns.append("country_Other")

    numeric_feature_columns = (
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
            "release_year",
        ]
        + selected_genres
        + country_columns
    )
    return {
        "top_countries": top_countries,
        "selected_genres": selected_genres,
        "feature_columns": numeric_feature_columns,
    }


def transform_preprocessor(X, config, state):
    """Transform data with feature choices learned from training data."""
    top_revenue_companies = config["top_revenue_companies"]
    star_actors = config["star_actors"]

    X = budget_missing_indicator(X)
    X = budget_0_to_nan(X)
    X = log_budget(X)

    X = collection_to_binary(X)

    X = log_popularity(X)

    X = english_to_binary(X)

    X = count_countries(X)
    X, _ = add_country_dummies(X, state["top_countries"])

    X = add_top_company_features(X, top_revenue_companies)

    X = encode_genres(X, state["selected_genres"])

    X = add_star_actor_features(X, star_actors)

    X = extract_release_year(X)

    # Keep only model-safe numeric columns in the transformed frame.
    # This removes non-numeric object/date text columns that XGBoost rejects.
    X_numeric = X.reindex(columns=state["feature_columns"], fill_value=0).copy()

    # Infer numeric columns if there are missing object columns not selected.
    # Also ensure all kept columns are numeric or boolean-compatible.
    for column in X_numeric.columns:
        if X_numeric[column].dtype == object:
            X_numeric[column] = pd.to_numeric(X_numeric[column], errors="coerce")

    return X_numeric


def run_preprocessing(X, config):
    """Fit and apply preprocessing to one standalone dataframe."""
    state = fit_preprocessor(X, config)
    return transform_preprocessor(X, config, state)
