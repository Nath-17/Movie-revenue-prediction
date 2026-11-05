"""Measures of the relationship between two variables."""

import numpy as np
import pandas as pd
from scipy import stats

OTHER_LABEL = "Other"
MIN_PAIR_SAMPLE_SIZE = 3


def spearman_abs(x: pd.Series, y: pd.Series) -> float:
    pairs = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if (
        len(pairs) < 2
        or pairs.iloc[:, 0].nunique() < 2
        or pairs.iloc[:, 1].nunique() < 2
    ):
        return 0.0
    value = pairs.iloc[:, 0].corr(pairs.iloc[:, 1], method="spearman")
    if pd.isna(value):
        return 0.0
    absolute_value = abs(float(value))
    return 1.0 if np.isclose(absolute_value, 1.0) else absolute_value


def eta_squared(cat: pd.Series, num: pd.Series) -> float:
    pairs = pd.DataFrame({"category": cat, "value": pd.to_numeric(num, errors="coerce")})
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 2 or pairs["category"].nunique() < 2:
        return 0.0
    overall_mean = pairs["value"].mean()
    total = ((pairs["value"] - overall_mean) ** 2).sum()
    if total == 0:
        return 0.0
    between = (
        pairs.groupby("category", observed=True)["value"]
        .apply(lambda values: len(values) * (values.mean() - overall_mean) ** 2)
        .sum()
    )
    return float(np.clip(between / total, 0.0, 1.0))


def combine_rare_categories(values: pd.Series, min_group_size: int) -> pd.Series:
    """Group categories with fewer than `min_group_size` rows under 'Other'."""
    values = values.astype("string").fillna("Unknown")
    counts = values.value_counts()
    rare = counts[counts < min_group_size].index
    return values.where(~values.isin(rare), OTHER_LABEL)


def clean_numeric(values: pd.Series, log_scale: bool = False) -> pd.Series:
    """Convert to float, drop infinities and, on a log scale, non-positive values."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if log_scale:
        numeric = np.log10(numeric.where(numeric > 0))
    return numeric


def numeric_pair_stats(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    """Signed Pearson and Spearman correlations with p-values, plus a linear fit."""
    pairs = pd.concat([x, y], axis=1).dropna()
    result: dict[str, float | int | None] = {
        "n": len(pairs),
        "pearson": None, "pearson_p": None,
        "spearman": None, "spearman_p": None,
        "slope": None, "intercept": None, "r_squared": None,
    }
    if (
        len(pairs) < MIN_PAIR_SAMPLE_SIZE
        or pairs.iloc[:, 0].nunique() < 2
        or pairs.iloc[:, 1].nunique() < 2
    ):
        return result
    a, b = pairs.iloc[:, 0], pairs.iloc[:, 1]
    pearson = stats.pearsonr(a, b)
    spearman = stats.spearmanr(a, b)
    fit = stats.linregress(a, b)
    result.update(
        pearson=float(pearson.statistic), pearson_p=float(pearson.pvalue),
        spearman=float(spearman.statistic), spearman_p=float(spearman.pvalue),
        slope=float(fit.slope), intercept=float(fit.intercept),
        r_squared=float(fit.rvalue**2),
    )
    return result


def group_summary(category: pd.Series, value: pd.Series) -> pd.DataFrame:
    """Count, median, mean and spread of a numeric variable for each category."""
    pairs = pd.DataFrame({"group": category, "value": value}).dropna()
    summary = pairs.groupby("group", observed=True)["value"].agg(
        n="count", median="median", mean="mean", std="std", min="min", max="max"
    )
    return summary.sort_values("median", ascending=False)


def categorical_numeric_stats(
    category: pd.Series, value: pd.Series
) -> dict[str, float | int | None]:
    """Eta squared (effect size) and Kruskal-Wallis test across groups."""
    pairs = pd.DataFrame({"group": category, "value": value}).dropna()
    groups = [
        part.to_numpy() for _, part in pairs.groupby("group", observed=True)["value"]
    ]
    result: dict[str, float | int | None] = {
        "n": len(pairs),
        "groups": len(groups),
        "eta_squared": eta_squared(pairs["group"], pairs["value"]),
        "kruskal_h": None,
        "kruskal_p": None,
    }
    if len(groups) >= 2 and pairs["value"].nunique() >= 2:
        test = stats.kruskal(*groups)
        result.update(kruskal_h=float(test.statistic), kruskal_p=float(test.pvalue))
    return result


def contingency_table(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    pairs = pd.DataFrame({"a": a, "b": b}).dropna()
    return pd.crosstab(pairs["a"], pairs["b"])


def categorical_pair_stats(table: pd.DataFrame) -> dict[str, float | int | None]:
    """Chi-squared test of independence and Cramér's V (0 = independent, 1 = linked)."""
    n = int(table.to_numpy().sum())
    rows, cols = table.shape
    result: dict[str, float | int | None] = {
        "n": n, "chi2": None, "chi2_p": None, "dof": None, "cramers_v": None,
    }
    if n == 0 or rows < 2 or cols < 2:
        return result
    test = stats.chi2_contingency(table)
    result.update(
        chi2=float(test.statistic),
        chi2_p=float(test.pvalue),
        dof=int(test.dof),
        cramers_v=float(
            stats.contingency.association(table.to_numpy(), method="cramer")
        ),
    )
    return result
