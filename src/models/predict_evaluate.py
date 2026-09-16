from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.model_registry import load_model_by_alias


def predict(
    X: pd.DataFrame,
    model_name: str,
    model_alias: str,
) -> pd.DataFrame:
    """Return a copy of X with a prediction column."""
    model = load_model_by_alias(model_name, model_alias)
    result = X.copy()
    result["prediction"] = np.asarray(model.predict(X)).reshape(-1)
    return result


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",
) -> Path:
    """Save predictions and return the created file path."""
    output_path = Path(output_path)
    file_format = format.lower().lstrip(".")
    supported_formats = {"parquet", "csv", "json"}
    if file_format not in supported_formats:
        raise ValueError("format must be 'parquet', 'csv' or 'json'")

    if output_path.suffix.lower().lstrip(".") in supported_formats:
        file_path = output_path
    else:
        file_path = output_path / f"predictions.{file_format}"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "parquet":
        predictions.to_parquet(file_path, index=False)
    elif file_format == "csv":
        predictions.to_csv(file_path, index=False)
    else:
        predictions.to_json(file_path, orient="records", date_format="iso")
    return file_path
