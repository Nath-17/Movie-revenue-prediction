from typing import TypedDict

import numpy as np
import pandas as pd

from src.stats.association import describe_strength

MONTH_LABELS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
MIN_GROUP_SIZE = 30
MIN_MONTH_GROUP_SIZE = 10
MIN_INSIGHT_SAMPLE_SIZE = 30
GENRE_CHART_LIMIT = 10
MIN_FRANCHISE_RATIO = 1.15


class KpiValues(TypedDict):
    """Typed summary values shown in the Explore KPI row."""

    films: int
    median_revenue: float | None
    median_budget: float | None
    profitable_share: float | None
    known_budgets: int


class ExploreInsights(TypedDict):
    """Typed data-derived findings displayed by the Explore page."""

    kpis: KpiValues
    year_scope: str
    budget: "BudgetStory | None"
    franchise: "StoryFinding | None"
    genre: "StoryFinding | None"
    seasonality: "StoryFinding | None"


class StoryFinding(TypedDict):
    """Computed title and takeaway for a supported story chart."""

    title: str
    caption: str


class BudgetStory(StoryFinding):
    """Budget finding with the elasticity used to draw its trend line."""

    elasticity: float


class DriverFinding(TypedDict):
    """Computed headline, chart title, and method caption for driver ranking."""

    headline: str
    title: str
    caption: str


def _numeric_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def budget_elasticity(df: pd.DataFrame) -> float | None:
    budget = _numeric_values(df["budget"])
    revenue = _numeric_values(df["revenue"])
    valid = (budget > 0) & (revenue > 0)
    if int(valid.sum()) < 2:
        return None
    slope = np.polyfit(np.log(budget[valid]), np.log(revenue[valid]), 1)[0]
    return float(slope) if np.isfinite(slope) else None


def franchise_median_ratio(df: pd.DataFrame) -> float | None:
    revenue = _numeric_values(df["revenue"])
    franchise = df["is_franchise"] if "is_franchise" in df else df["collection"].notna()
    valid = revenue.notna() & (revenue > 0)
    medians = revenue[valid].groupby(franchise[valid]).median()
    if not {False, True}.issubset(medians.index):
        return None
    if medians[False] <= 0:
        return None
    return float(medians[True] / medians[False])


def franchise_sample_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    revenue = _numeric_values(df["revenue"])
    franchise = df["is_franchise"] if "is_franchise" in df else df["collection"].notna()
    valid = revenue.notna() & (revenue > 0)
    return (
        int((valid & ~franchise).sum()),
        int((valid & franchise).sum()),
        int(len(df) - valid.sum()),
    )


def genre_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["primary_genre", "revenue"]].copy()
    data["revenue"] = _numeric_values(data["revenue"])
    data = data.dropna()
    data = data[data["revenue"] > 0]
    summary = data.groupby("primary_genre")["revenue"].agg(["median", "count"])
    summary = summary[summary["count"] >= MIN_GROUP_SIZE]
    return summary.sort_values("median", ascending=False)


def top_genre(df: pd.DataFrame) -> str | None:
    summary = genre_revenue_summary(df)
    return str(summary.index[0]) if not summary.empty else None


def genre_roi_rankings_differ(df: pd.DataFrame) -> bool:
    required = {"primary_genre", "budget", "revenue"}
    if not required.issubset(df.columns):
        return False
    data = df[list(required)].copy()
    data["budget"] = _numeric_values(data["budget"])
    data["revenue"] = _numeric_values(data["revenue"])
    revenue_data = data[data["revenue"] > 0]
    revenue_summary = revenue_data.groupby("primary_genre")["revenue"].agg(
        ["median", "count"]
    )
    budget_data = data[(data["budget"] > 0) & (data["revenue"] > 0)]
    roi = budget_data.assign(roi=budget_data["revenue"] / budget_data["budget"])
    roi_summary = roi.groupby("primary_genre")["roi"].median()
    eligible_revenue = revenue_summary[
        revenue_summary["count"] >= MIN_GROUP_SIZE
    ].index
    eligible_roi = budget_data.groupby("primary_genre").size()
    eligible = eligible_revenue.intersection(
        eligible_roi[eligible_roi >= MIN_GROUP_SIZE].index
    )
    if len(eligible) < 2:
        return False
    revenue_leader = revenue_summary.loc[eligible, "median"].idxmax()
    roi_leader = roi_summary.loc[eligible].idxmax()
    return revenue_leader != roi_leader


def monthly_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["release_month", "revenue"]].copy()
    data["revenue"] = _numeric_values(data["revenue"])
    data = data.dropna()
    data = data[(data["revenue"] > 0) & data["release_month"].between(1, 12)]
    summary = data.groupby("release_month")["revenue"].agg(
        median="median",
        lower=lambda values: values.quantile(0.25),
        upper=lambda values: values.quantile(0.75),
        count="count",
    )
    return summary[summary["count"] >= MIN_MONTH_GROUP_SIZE].reindex(
        range(1, 13)
    ).dropna(subset=["median"])


def best_months(df: pd.DataFrame) -> tuple[str, ...]:
    summary = monthly_revenue_summary(df)
    labels = [MONTH_LABELS[int(month) - 1] for month in summary.nlargest(2, "median").index]
    return tuple(labels)


def dataset_year_scope(df: pd.DataFrame) -> str:
    years = pd.to_numeric(df["release_year"], errors="coerce").dropna()
    if years.empty:
        return "unknown years"
    return f"{int(years.min())}–{int(years.max())}"


def budget_sample_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    budget = _numeric_values(df["budget"])
    revenue = _numeric_values(df["revenue"])
    valid = (budget > 0) & (revenue > 0)
    bad_budget = int((budget.isna() | (budget <= 0)).sum())
    bad_revenue = int((revenue.isna() | (revenue <= 0)).sum())
    return int(valid.sum()), bad_budget, bad_revenue


def kpi_values(df: pd.DataFrame) -> KpiValues:
    revenue = _numeric_values(df["revenue"])
    budget = _numeric_values(df["budget"])
    known = revenue.notna() & budget.notna() & (budget > 0)
    positive_revenue = revenue[revenue > 0]
    positive_budget = budget[budget > 0]
    return {
        "films": len(df),
        "median_revenue": (
            float(positive_revenue.median()) if not positive_revenue.empty else None
        ),
        "median_budget": (
            float(positive_budget.median()) if not positive_budget.empty else None
        ),
        "profitable_share": (
            float((revenue[known] > budget[known]).mean()) if known.any() else None
        ),
        "known_budgets": int(known.sum()),
    }


def budget_story(df: pd.DataFrame) -> BudgetStory | None:
    elasticity = budget_elasticity(df)
    n, bad_budget, bad_revenue = budget_sample_counts(df)
    if elasticity is None or n < MIN_INSIGHT_SAMPLE_SIZE:
        return None
    growth = (2**elasticity - 1) * 100
    direction = "more" if growth >= 0 else "less"
    percent = abs(growth)
    return {
        "title": f"Each budget doubling is associated with {percent:.0f}% {direction} revenue",
        "caption": (
            f"Each budget doubling is associated with about {percent:.0f}% "
            f"{direction} revenue; n = {n:,}. Dropped {bad_budget:,} films with "
            f"non-positive or missing budget and {bad_revenue:,} with "
            "non-positive or missing revenue."
        ),
        "elasticity": elasticity,
    }


def franchise_story(df: pd.DataFrame) -> StoryFinding | None:
    ratio = franchise_median_ratio(df)
    standalone_n, franchise_n, dropped = franchise_sample_counts(df)
    if (
        ratio is None
        or 1 / MIN_FRANCHISE_RATIO <= ratio <= MIN_FRANCHISE_RATIO
        or min(standalone_n, franchise_n) < MIN_INSIGHT_SAMPLE_SIZE
    ):
        return None
    if ratio >= 1:
        title = f"Franchise films earn {ratio:.1f}× standalone films at the median"
    else:
        title = f"Standalone films earn {1 / ratio:.1f}× franchise films at the median"
    return {
        "title": title,
        "caption": (
            f"Median revenue ratio: {ratio:.1f}× franchise to standalone; "
            f"n = {standalone_n:,} standalone and {franchise_n:,} franchise. "
            f"Dropped {dropped:,} films with missing or non-positive revenue."
        ),
    }


def genre_story(df: pd.DataFrame) -> StoryFinding | None:
    summary = genre_revenue_summary(df)
    if len(summary) < 2:
        return None
    genre = str(summary.index[0])
    top = summary.head(GENRE_CHART_LIMIT)
    n = int(top["count"].sum())
    roi_note = (
        "median revenue and ROI rankings differ."
        if genre_roi_rankings_differ(df)
        else "revenue and ROI leaders are consistent."
    )
    return {
        "title": f"{genre} has the highest median revenue among primary genres",
        "caption": (
            f"Primary genre only; {roi_note} n = {n:,} across the top "
            f"{GENRE_CHART_LIMIT}; {len(df) - n:,} films omitted for missing "
            "values, groups under 30, or falling outside the top 10."
        ),
    }


def seasonality_story(df: pd.DataFrame) -> StoryFinding | None:
    summary = monthly_revenue_summary(df)
    months = best_months(df)
    n = int(summary["count"].sum()) if not summary.empty else 0
    if len(summary) < 2 or n < MIN_INSIGHT_SAMPLE_SIZE or not months:
        return None
    return {
        "title": f"{' and '.join(months)} have the highest median revenues",
        "caption": (
            f"Best months: {', '.join(months)}. n = {n:,}; {len(df) - n:,} "
            "films omitted for missing/invalid month or revenue, or month "
            f"groups under {MIN_MONTH_GROUP_SIZE}."
        ),
    }


def driver_finding(ranking: pd.DataFrame) -> DriverFinding | None:
    if ranking.empty:
        return None
    labels = ranking["label"].head(3).tolist()
    top = ranking.iloc[0]
    strength = describe_strength(float(top["association"]))
    signals = " and ".join(labels[1:])
    headline = (
        f"{labels[0]} leads with a {strength} association; "
        f"{signals} are the next strongest signals."
        if signals
        else f"{labels[0]} has the strongest association with log revenue."
    )
    n_values = ", ".join(
        f"{row.label}: n = {row.n:,} (dropped {row.dropped:,})"
        for row in ranking.itertuples()
    )
    return {
        "headline": headline,
        "title": f"{labels[0]} has the strongest association with log revenue",
        "caption": (
            f"{n_values}. |Spearman rho| measures monotonic numeric association; "
            "eta squared measures how much revenue variation separates categories. "
            "Rows are excluded for missing features or invalid revenue; budget also "
            "excludes non-positive values."
        ),
    }


def explore_insights(df: pd.DataFrame) -> ExploreInsights:
    return {
        "kpis": kpi_values(df),
        "year_scope": dataset_year_scope(df),
        "budget": budget_story(df),
        "franchise": franchise_story(df),
        "genre": genre_story(df),
        "seasonality": seasonality_story(df),
    }
