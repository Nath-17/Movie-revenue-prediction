import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.insights import (
    GENRE_CHART_LIMIT,
    MONTH_LABELS,
    genre_revenue_summary,
    monthly_revenue_summary,
)
from app.components.theme import (
    ACCENT,
    ACCENT_LIGHT,
    ASSOCIATION_LABEL_DECIMALS,
    CATEGORY,
    GENRE_LABEL_DECIMALS,
    GREY,
    HIDDEN_LINE_WIDTH,
    LOG_TICK_TARGET_COUNT,
    MILLION,
    MONEY_LABEL_DECIMALS,
    RANKING_AXIS_MAX,
    RANKING_AXIS_TICKS,
    RANKING_CHART_HEIGHT,
    RANKING_CHART_MARGIN,
    SCATTER_MARKER_OPACITY,
    SCATTER_MARKER_SIZE,
    SEASONAL_LINE_WIDTH,
    SEASONAL_MARKER_SIZE,
    TREND_LINE_POINTS,
    TREND_LINE_WIDTH,
    apply_theme,
)


def _log_tick_labels(values: np.ndarray) -> tuple[list[float], list[str]]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    lower = int(np.floor(np.log10(minimum)))
    upper = int(np.ceil(np.log10(maximum)))
    step = max(1, int(np.ceil((upper - lower) / LOG_TICK_TARGET_COUNT)))
    exponents = list(range(lower, upper + 1, step))
    if upper not in exponents:
        exponents.append(upper)
    ticks = [float(10**exponent) for exponent in exponents]
    labels = []
    for tick in ticks:
        decimals = max(0, -int(np.floor(np.log10(tick))))
        label = f"{tick:,.{decimals}f}"
        labels.append(label.rstrip("0").rstrip(".") if "." in label else label)
    return ticks, labels


def driver_ranking_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for feature_type, color, label in (
        ("numeric", CATEGORY[0], "Numeric (|Spearman rho|)"),
        ("categorical", CATEGORY[1], "Categorical (eta squared)"),
    ):
        part = df[df["type"] == feature_type]
        fig.add_trace(go.Bar(
            x=part["association"], y=part["label"], orientation="h",
            marker_color=color,
            text=part["association"].map(
                lambda value: f"{value:.{ASSOCIATION_LABEL_DECIMALS}f}"
            ),
            textposition="outside", cliponaxis=False, name=label,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, barmode="overlay", showlegend=True,
        xaxis_title="Association strength with log revenue (0–1)",
        xaxis={
            "range": [0, RANKING_AXIS_MAX],
            "tickvals": RANKING_AXIS_TICKS,
        },
        height=RANKING_CHART_HEIGHT,
        margin=RANKING_CHART_MARGIN,
        yaxis={"title": "Feature", "categoryorder": "array", "categoryarray": df["label"].tolist(),
               "autorange": "reversed", "tickmode": "array",
               "tickvals": df["label"].tolist(), "ticktext": df["label"].tolist()},
    )
    return apply_theme(fig)


def budget_scatter_chart(df: pd.DataFrame, title: str, elasticity: float) -> go.Figure:
    data = df[["budget", "revenue"]].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["budget"] > 0) & (data["revenue"] > 0)]
    x = data["budget"].to_numpy() / MILLION
    y = data["revenue"].to_numpy() / MILLION
    x_ticks, x_labels = _log_tick_labels(x)
    y_ticks, y_labels = _log_tick_labels(y)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    fit_x = np.geomspace(x.min(), x.max(), TREND_LINE_POINTS)
    fit_y = np.exp(intercept + slope * np.log(fit_x))
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", name="Films",
        marker={
            "color": GREY,
            "opacity": SCATTER_MARKER_OPACITY,
            "size": SCATTER_MARKER_SIZE,
        },
        hovertemplate=(
            f"Budget: $%{{x:,.{MONEY_LABEL_DECIMALS}f}}M<br>"
            f"Revenue: $%{{y:,.{MONEY_LABEL_DECIMALS}f}}M<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=fit_x, y=fit_y, mode="lines",
        name=f"OLS fit (elasticity {elasticity:.2f})",
        line={"color": ACCENT, "width": TREND_LINE_WIDTH},
        hovertemplate=f"Trend: $%{{y:,.{MONEY_LABEL_DECIMALS}f}}M<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_type="log", yaxis_type="log",
        xaxis_title="Budget ($M, log scale)",
        yaxis_title="Revenue ($M, log scale)",
        xaxis={"tickmode": "array", "tickvals": x_ticks, "ticktext": x_labels},
        yaxis={"tickmode": "array", "tickvals": y_ticks, "ticktext": y_labels},
        showlegend=True,
    )
    return apply_theme(fig)


def franchise_box_chart(df: pd.DataFrame, title: str) -> go.Figure:
    data = df[["revenue", "is_franchise"]].copy()
    data["revenue"] = (
        pd.to_numeric(data["revenue"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        / MILLION
    )
    data = data.dropna()
    data = data[data["revenue"] > 0]
    ticks, labels = _log_tick_labels(data["revenue"].to_numpy())
    fig = go.Figure()
    for value, label, color in (
        (False, "Standalone", GREY),
        (True, "Franchise", ACCENT),
    ):
        group = data.loc[data["is_franchise"] == value, "revenue"]
        fig.add_trace(go.Box(
            y=group, name=label, marker_color=color, line_color=color,
            boxpoints=False,
            hovertemplate=(
                f"{label}<br>Revenue: "
                f"$%{{y:,.{MONEY_LABEL_DECIMALS}f}}M<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=title, showlegend=False,
        yaxis_title="Revenue ($M, log scale)", yaxis_type="log",
        yaxis={"tickmode": "array", "tickvals": ticks, "ticktext": labels},
        xaxis_title="Film type",
    )
    return apply_theme(fig)


def genre_median_chart(df: pd.DataFrame, title: str) -> go.Figure:
    summary = genre_revenue_summary(df).head(GENRE_CHART_LIMIT).sort_values("median")
    labels = summary.index.tolist()
    colors = [ACCENT if label == summary["median"].idxmax() else GREY for label in labels]
    fig = go.Figure(go.Bar(
        x=summary["median"] / MILLION, y=labels, orientation="h",
        marker_color=colors, text=(summary["median"] / MILLION).map(
            lambda value: f"${value:,.{GENRE_LABEL_DECIMALS}f}M"
        ),
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: $%{x:,.1f}M<extra></extra>",
    ))
    fig.update_layout(
        title=title, showlegend=False,
        xaxis_title="Median revenue ($M)",
        yaxis={"title": "Primary genre", "categoryorder": "array", "categoryarray": labels},
    )
    return apply_theme(fig)


def seasonality_line_chart(df: pd.DataFrame, title: str) -> go.Figure:
    summary = monthly_revenue_summary(df)
    labels = [MONTH_LABELS[int(month) - 1] for month in summary.index]
    values = summary["median"] / MILLION
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=summary["lower"] / MILLION, mode="lines",
        line={"width": HIDDEN_LINE_WIDTH}, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=summary["upper"] / MILLION, mode="lines",
        line={"width": HIDDEN_LINE_WIDTH}, fill="tonexty", fillcolor=ACCENT_LIGHT,
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers", name="Monthly median",
        line={"color": ACCENT, "width": SEASONAL_LINE_WIDTH},
        marker={"color": ACCENT, "size": SEASONAL_MARKER_SIZE},
        customdata=summary[["count"]].to_numpy(),
        hovertemplate=f"%{{x}}: $%{{y:,.{MONEY_LABEL_DECIMALS}f}}M<extra></extra>",
    ))
    fig.update_layout(
        title=title, showlegend=False,
        xaxis_title="Release month", yaxis_title="Median revenue ($M)",
        xaxis={"categoryorder": "array", "categoryarray":
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
    )
    return apply_theme(fig)
