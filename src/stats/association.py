from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.stats.bivariate import combine_rare_categories, eta_squared, spearman_abs
from src.stats.variable_registry import FeatureSpec

MIN_ASSOCIATION_SAMPLE_SIZE = 30


def driver_ranking(
    df: pd.DataFrame, registry: Iterable[FeatureSpec]
) -> pd.DataFrame:
    revenue = pd.to_numeric(df["revenue"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    log_revenue = np.log10(revenue.where(revenue > 0))
    rows = []
    for feature in registry:
        row = _rank_feature(df, feature, revenue, log_revenue)
        if row is not None:
            rows.append(row)
    columns = ["feature", "label", "type", "association", "n", "dropped"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        "association", ascending=False, ignore_index=True
    )


def _rank_feature(
    df: pd.DataFrame,
    feature: FeatureSpec,
    revenue: pd.Series,
    log_revenue: pd.Series,
) -> dict[str, str | float | int] | None:
    if feature.column not in df:
        return None
    values = df[feature.column]
    valid = values.notna() & revenue.notna() & (revenue > 0)
    if feature.kind == "categorical":
        categories = combine_rare_categories(values[valid], feature.min_group_size)
        score = eta_squared(categories, log_revenue[valid])
    else:
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid &= numeric.notna()
        score = spearman_abs(numeric[valid], log_revenue[valid])
    count = int(valid.sum())
    if count < MIN_ASSOCIATION_SAMPLE_SIZE:
        return None
    return {
        "feature": feature.column,
        "label": feature.label,
        "type": feature.kind,
        "association": score,
        "n": count,
        "dropped": int(len(df) - count),
    }


def describe_strength(value: float) -> str:
    if value < 0.1:
        return "very weak"
    if value < 0.3:
        return "weak"
    if value < 0.5:
        return "moderate"
    if value < 0.7:
        return "strong"
    return "very strong"
