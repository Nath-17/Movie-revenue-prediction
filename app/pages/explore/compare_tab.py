import numpy as np
import pandas as pd
import streamlit as st

from app.components.charts_bivariate import (
    crosstab_heatmap,
    group_box_chart,
    pair_scatter_chart,
)
from app.components.insights import MONTH_LABELS
from app.components.theme import MILLION
from src.stats.association import MIN_ASSOCIATION_SAMPLE_SIZE, describe_strength
from src.stats.bivariate import (
    categorical_numeric_stats,
    categorical_pair_stats,
    clean_numeric,
    combine_rare_categories,
    contingency_table,
    group_summary,
    numeric_pair_stats,
)
from src.stats.variable_registry import FEATURE_REGISTRY, FeatureSpec

COMPARE_VARIABLES = (
    FeatureSpec("revenue", "Revenue", "numeric"),
    *FEATURE_REGISTRY,
    FeatureSpec("release_month", "Release month", "categorical"),
)
MONEY_COLUMNS = frozenset({"revenue", "budget_known"})
SIGNIFICANCE_LEVEL = 0.05


def _display_label(spec: FeatureSpec) -> str:
    return f"{spec.label} ($M)" if spec.column in MONEY_COLUMNS else spec.label


def _numeric_values(df: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    values = clean_numeric(df[spec.column])
    if spec.column in MONEY_COLUMNS:
        values = values.where(values > 0) / MILLION
    return values


def _model_scale(values: pd.Series, spec: FeatureSpec) -> pd.Series:
    return np.log10(values) if spec.column in MONEY_COLUMNS else values


def _category_values(df: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    values = df[spec.column]
    if spec.column == "is_franchise":
        values = values.map({True: "Franchise", False: "Standalone"})
    elif spec.column == "release_month":
        values = values.map(lambda month: MONTH_LABELS[int(month) - 1], na_action="ignore")
    values = values.dropna()
    return combine_rare_categories(values, spec.min_group_size).reindex(df.index)


def _category_order(spec: FeatureSpec, present: pd.Index) -> list[str]:
    if spec.column == "release_month":
        return [month for month in MONTH_LABELS if month in present]
    return [str(value) for value in present]


def _format_p(value: float | None) -> str:
    if value is None:
        return "—"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _significance(p_value: float | None) -> str:
    if p_value is None:
        return "The test could not be computed."
    if p_value < SIGNIFICANCE_LEVEL:
        return f"Statistically significant at the {SIGNIFICANCE_LEVEL:.0%} level."
    return f"Not statistically significant at the {SIGNIFICANCE_LEVEL:.0%} level."


def _warn_if_small(n: int) -> bool:
    if n < MIN_ASSOCIATION_SAMPLE_SIZE:
        st.warning(
            f"Only {n} films have both values; at least "
            f"{MIN_ASSOCIATION_SAMPLE_SIZE} are needed for a reliable comparison."
        )
        return True
    return False


def _render_numeric_pair(df: pd.DataFrame, x_spec: FeatureSpec, y_spec: FeatureSpec) -> None:
    x = _numeric_values(df, x_spec)
    y = _numeric_values(df, y_spec)
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    stats = numeric_pair_stats(_model_scale(x, x_spec), _model_scale(y, y_spec))
    if _warn_if_small(stats["n"]) or stats["spearman"] is None:
        return

    columns = st.columns(4)
    columns[0].metric("Films", f"{stats['n']:,}")
    columns[1].metric("Pearson r", f"{stats['pearson']:+.2f}")
    columns[2].metric("Spearman ρ", f"{stats['spearman']:+.2f}")
    columns[3].metric("R² (linear fit)", f"{stats['r_squared']:.2f}")
    st.plotly_chart(
        pair_scatter_chart(
            x, y, _display_label(x_spec), _display_label(y_spec),
            x_spec.column in MONEY_COLUMNS, y_spec.column in MONEY_COLUMNS,
            stats["slope"], stats["intercept"],
        ),
        width="stretch",
        config={"displayModeBar": True},
    )
    direction = "positive" if stats["spearman"] > 0 else "negative"
    st.caption(
        f"{describe_strength(abs(stats['spearman'])).capitalize()} {direction} "
        f"monotonic association (Spearman ρ = {stats['spearman']:+.2f}, "
        f"p {_format_p(stats['spearman_p'])}). {_significance(stats['spearman_p'])} "
        "Money variables are analysed on a log10 scale."
    )
    with st.expander("Summary statistics"):
        summary = pd.DataFrame({
            _display_label(x_spec): x.describe(),
            _display_label(y_spec): y.describe(),
        })
        st.dataframe(summary.style.format("{:,.2f}"), width="stretch")


def _render_mixed_pair(
    df: pd.DataFrame, category_spec: FeatureSpec, value_spec: FeatureSpec
) -> None:
    category = _category_values(df, category_spec)
    value = _numeric_values(df, value_spec)
    valid = category.notna() & value.notna()
    category, value = category[valid], value[valid]
    stats = categorical_numeric_stats(category, _model_scale(value, value_spec))
    if _warn_if_small(stats["n"]):
        return
    summary = group_summary(category, value)

    columns = st.columns(4)
    columns[0].metric("Films", f"{stats['n']:,}")
    columns[1].metric("Groups", stats["groups"])
    columns[2].metric("Eta²", f"{stats['eta_squared']:.2f}")
    columns[3].metric("Kruskal-Wallis p", _format_p(stats["kruskal_p"]))
    st.plotly_chart(
        group_box_chart(
            category, value, category_spec.label, _display_label(value_spec),
            value_spec.column in MONEY_COLUMNS,
            _category_order(category_spec, summary.index),
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        f"{category_spec.label} explains {stats['eta_squared']:.0%} of the variance in "
        f"{value_spec.label.lower()} ({describe_strength(stats['eta_squared'])} effect). "
        f"Kruskal-Wallis test: {_significance(stats['kruskal_p'])} "
        f"Categories with fewer than {category_spec.min_group_size} films are grouped "
        "under 'Other'."
    )
    with st.expander("Statistics by group"):
        st.dataframe(
            summary.style.format({"n": "{:,}", **{
                column: "{:,.2f}" for column in summary.columns if column != "n"
            }}),
            width="stretch",
        )


def _render_categorical_pair(
    df: pd.DataFrame, row_spec: FeatureSpec, column_spec: FeatureSpec
) -> None:
    table = contingency_table(_category_values(df, row_spec), _category_values(df, column_spec))
    table = table.loc[
        _category_order(row_spec, table.index), _category_order(column_spec, table.columns)
    ]
    stats = categorical_pair_stats(table)
    if _warn_if_small(stats["n"]) or stats["cramers_v"] is None:
        return

    columns = st.columns(4)
    columns[0].metric("Films", f"{stats['n']:,}")
    columns[1].metric("Cramér's V", f"{stats['cramers_v']:.2f}")
    columns[2].metric("χ²", f"{stats['chi2']:,.1f}", help=f"{stats['dof']} degrees of freedom")
    columns[3].metric("χ² p-value", _format_p(stats["chi2_p"]))
    st.plotly_chart(
        crosstab_heatmap(table, row_spec.label, column_spec.label),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        f"{describe_strength(stats['cramers_v']).capitalize()} association "
        f"(Cramér's V = {stats['cramers_v']:.2f}). Chi-squared test of independence: "
        f"{_significance(stats['chi2_p'])}"
    )
    with st.expander("Contingency table (film counts)"):
        st.dataframe(table, width="stretch")


def render_compare_tab(df: pd.DataFrame) -> None:
    st.title("Compare two variables")
    st.write(
        "Pick two variables to see how they relate. The chart and statistics adapt "
        "to the variable types."
    )
    specs = {spec.label: spec for spec in COMPARE_VARIABLES if spec.column in df}
    labels = list(specs)
    columns = st.columns(2)
    first = columns[0].selectbox("First variable", labels, index=labels.index("Budget"))
    second = columns[1].selectbox("Second variable", labels, index=labels.index("Revenue"))
    if first == second:
        st.info("Choose two different variables.")
        return

    a, b = specs[first], specs[second]
    if a.kind == "numeric" and b.kind == "numeric":
        _render_numeric_pair(df, a, b)
    elif a.kind == "categorical" and b.kind == "categorical":
        _render_categorical_pair(df, a, b)
    elif a.kind == "categorical":
        _render_mixed_pair(df, a, b)
    else:
        _render_mixed_pair(df, b, a)
    st.info("These are associations, not causal effects.")
