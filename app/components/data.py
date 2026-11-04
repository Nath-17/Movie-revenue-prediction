from pathlib import Path

import pandas as pd
import yaml

from src.data.dataloader import run_dataloader
from src.data.preprocessing import extract_release_year

ROOT = Path(__file__).resolve().parents[2]


def load_explore_data() -> pd.DataFrame:
    with (ROOT / "config" / "config_paths.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["local_inputs_data_path"] = str(ROOT / config["local_inputs_data_path"])
    features, revenue, _ = run_dataloader(config)
    return prepare_explore_data(features.join(revenue))


def prepare_explore_data(df: pd.DataFrame) -> pd.DataFrame:
    result = extract_release_year(df)
    result["release_month"] = result["date_format"].dt.month
    result["budget_known"] = pd.to_numeric(result["budget"], errors="coerce").where(
        pd.to_numeric(result["budget"], errors="coerce") > 0
    )
    result["is_franchise"] = result["collection"].notna()
    result["primary_genre"] = result["genre"].map(_first_value)
    result["main_country"] = result["country"].map(_first_value)
    return result


def _first_value(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    first = value.split(",", maxsplit=1)[0].strip()
    return first or None
