import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.charts_drivers import _log_tick_labels
from app.components.theme import (
    ACCENT,
    GREY,
    SCATTER_MARKER_OPACITY,
    SCATTER_MARKER_SIZE,
    TREND_LINE_POINTS,
    TREND_LINE_WIDTH,
    apply_theme,
)

HEATMAP_SCALE = ((0.0, "#F4F7F8"), (1.0, ACCENT))
HEATMAP_DARK_TEXT_THRESHOLD = 0.55


def _axis(title: str, values: pd.Series, log_scale: bool) -> dict:
    if not log_scale:
        return {"title": title}
    ticks, labels = _log_tick_labels(values.to_numpy())
    return {
        "title": f"{title} (log scale)", "type": "log",
        "tickmode": "array", "tickvals": ticks, "ticktext": labels,
    }


def pair_scatter_chart(
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    x_log: bool,
    y_log: bool,
    slope: float | None,
    intercept: float | None,
) -> go.Figure:
    """Scatter of two numeric variables with the linear fit drawn on the plotted scale."""
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", name="Films",
        marker={
            "color": GREY,
            "opacity": SCATTER_MARKER_OPACITY,
            "size": SCATTER_MARKER_SIZE,
        },
        hovertemplate=f"{x_label}: %{{x:,.4g}}<br>{y_label}: %{{y:,.4g}}<extra></extra>",
    ))
    if slope is not None and intercept is not None:
        fit_x = (np.geomspace if x_log else np.linspace)(x.min(), x.max(), TREND_LINE_POINTS)
        fit_input = np.log10(fit_x) if x_log else fit_x
        fit_y = intercept + slope * fit_input
        fig.add_trace(go.Scatter(
            x=fit_x, y=10**fit_y if y_log else fit_y, mode="lines",
            name="Linear fit", line={"color": ACCENT, "width": TREND_LINE_WIDTH},
            hoverinfo="skip",
        ))
    fig.update_layout(
        title=f"{y_label} vs {x_label}", showlegend=True,
        xaxis=_axis(x_label, x, x_log), yaxis=_axis(y_label, y, y_log),
    )
    return apply_theme(fig)


def group_box_chart(
    category: pd.Series,
    value: pd.Series,
    category_label: str,
    value_label: str,
    value_log: bool,
    order: list[str],
) -> go.Figure:
    """One box per category, in the given order."""
    fig = go.Figure(go.Box(
        x=category, y=value, marker_color=ACCENT, line_color=ACCENT,
        boxpoints=False, name=value_label,
        hovertemplate=f"%{{x}}<br>{value_label}: %{{y:,.4g}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{value_label} by {category_label}", showlegend=False,
        xaxis={"title": category_label, "categoryorder": "array", "categoryarray": order},
        yaxis=_axis(value_label, value, value_log),
    )
    return apply_theme(fig)


def crosstab_heatmap(table: pd.DataFrame, row_label: str, column_label: str) -> go.Figure:
    """Heatmap of row percentages: each row shows how one category splits."""
    shares = table.div(table.sum(axis=1), axis=0)
    rows = [str(value) for value in shares.index]
    columns = [str(value) for value in shares.columns]
    fig = go.Figure(go.Heatmap(
        z=shares.to_numpy(), x=columns, y=rows,
        customdata=table.to_numpy(), colorscale=HEATMAP_SCALE, zmin=0, zmax=1,
        colorbar={"title": "Row share", "tickformat": ".0%"},
        hovertemplate=(
            f"{row_label}: %{{y}}<br>{column_label}: %{{x}}<br>"
            "Share of row: %{z:.0%}<br>Films: %{customdata}<extra></extra>"
        ),
        xgap=2, ygap=2,
    ))
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            share = shares.iat[i, j]
            fig.add_annotation(
                x=column, y=row, text=f"{share:.0%}", showarrow=False,
                font={"color": "white" if share > HEATMAP_DARK_TEXT_THRESHOLD else "#333"},
            )
    fig.update_layout(
        title=f"{column_label} split within each {row_label}",
        xaxis={"title": column_label, "type": "category", "showgrid": False},
        yaxis={"title": row_label, "type": "category", "autorange": "reversed",
               "showgrid": False},
    )
    return apply_theme(fig)
