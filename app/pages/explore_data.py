import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts_drivers import (
    budget_scatter_chart,
    driver_ranking_chart,
    franchise_box_chart,
    genre_median_chart,
    seasonality_line_chart,
)
from app.components.data import load_explore_data
from app.components.insights import (
    MIN_INSIGHT_SAMPLE_SIZE,
    BudgetStory,
    DriverFinding,
    ExploreInsights,
    StoryFinding,
    driver_finding,
    explore_insights,
)
from app.components.theme import (
    MILLION,
    RANKING_CHART_HEIGHT,
    STORY_CHART_HEIGHT,
)
from src.stats.association import driver_ranking
from src.stats.variable_registry import FEATURE_REGISTRY

from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="What drives revenue?",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0.75rem; }
    .block-container [data-testid="stVerticalBlock"] { gap: 0.35rem; }
    .block-container h1 { font-size: 1.8rem; margin-bottom: 0; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _cached_ranking(df: pd.DataFrame) -> pd.DataFrame:
    return driver_ranking(df, FEATURE_REGISTRY)


@st.cache_data(show_spinner=False)
def _cached_insights(df: pd.DataFrame) -> ExploreInsights:
    return explore_insights(df)


@st.cache_data(show_spinner=False)
def _cached_driver_finding(ranking: pd.DataFrame) -> DriverFinding | None:
    return driver_finding(ranking)


def _money_millions(value: float | None) -> str:
    if value is None:
        return "—"
    amount = value / MILLION
    return f"${amount:,.0f}M" if amount >= 100 else f"${amount:,.1f}M"


def _render_kpis(values: ExploreInsights) -> None:
    kpis = values["kpis"]
    columns = st.columns(4)
    columns[0].metric("Films", f"{kpis['films']:,}")
    columns[1].metric("Median revenue", _money_millions(kpis["median_revenue"]))
    columns[2].metric("Median budget", _money_millions(kpis["median_budget"]))
    profitable = (
        f"{kpis['profitable_share']:.0%}"
        if kpis["profitable_share"] is not None
        else "—"
    )
    columns[3].metric("Profitable", profitable)
    st.caption(
        f"Budget and profitability metrics use the {kpis['known_budgets']:,} "
        "films with a reported positive budget."
    )


def _render_budget_story(df: pd.DataFrame, finding: BudgetStory | None) -> None:
    if finding is None:
        return
    st.divider()
    st.subheader("Budget")
    st.plotly_chart(
        budget_scatter_chart(df, finding["title"], finding["elasticity"]),
        width="stretch",
        height=STORY_CHART_HEIGHT,
        config={"displayModeBar": True},
    )
    st.caption(finding["caption"])


def _render_franchise_story(df: pd.DataFrame, finding: StoryFinding | None) -> None:
    if finding is None:
        return
    st.divider()
    st.subheader("Franchise")
    st.plotly_chart(
        franchise_box_chart(df, finding["title"]),
        width="stretch",
        height=STORY_CHART_HEIGHT,
        config={"displayModeBar": False},
    )
    st.caption(finding["caption"])


def _render_genre_story(df: pd.DataFrame, finding: StoryFinding | None) -> None:
    if finding is None:
        return
    st.divider()
    st.subheader("Genre")
    st.plotly_chart(
        genre_median_chart(df, finding["title"]),
        width="stretch",
        height=STORY_CHART_HEIGHT,
        config={"displayModeBar": False},
    )
    st.caption(finding["caption"])


def _render_seasonality_story(
    df: pd.DataFrame, finding: StoryFinding | None
) -> None:
    if finding is None:
        return
    st.divider()
    st.subheader("Seasonality")
    st.plotly_chart(
        seasonality_line_chart(df, finding["title"]),
        width="stretch",
        height=STORY_CHART_HEIGHT,
        config={"displayModeBar": False},
    )
    st.caption(finding["caption"])


def _render_ranking(ranking: pd.DataFrame, finding: DriverFinding) -> None:
    st.plotly_chart(
        driver_ranking_chart(ranking, finding["title"]),
        width="stretch",
        height=RANKING_CHART_HEIGHT,
        config={"displayModeBar": False},
    )
    st.caption(finding["caption"])


def _render_stories(df: pd.DataFrame, insights: ExploreInsights) -> None:
    if insights["kpis"]["films"] < MIN_INSIGHT_SAMPLE_SIZE:
        st.warning(
            f"Only {insights['kpis']['films']} films are available; story charts require at least "
            f"{MIN_INSIGHT_SAMPLE_SIZE} complete rows."
        )
        return

    _render_budget_story(df, insights["budget"])
    _render_franchise_story(df, insights["franchise"])
    _render_genre_story(df, insights["genre"])
    _render_seasonality_story(df, insights["seasonality"])

    st.divider()
    st.info(
        "These are associations, not causal effects; budget, franchise status "
        f"and genre overlap. Training data covers {insights['year_scope']}. "
        "Revenue is not adjusted for inflation."
    )


def render_drivers_tab(df: pd.DataFrame) -> None:
    ranking = _cached_ranking(df)
    insights = _cached_insights(df)
    finding = _cached_driver_finding(ranking)
    st.title("What drives revenue?")
    if finding is None:
        st.warning(
            f"Fewer than {MIN_INSIGHT_SAMPLE_SIZE} complete films are available "
            "for driver analysis."
        )
        return
    st.write(finding["headline"])
    _render_kpis(insights)
    _render_ranking(ranking, finding)
    _render_stories(df, insights)


@st.cache_data(show_spinner=False)
def _cached_explore_data() -> pd.DataFrame:
    return load_explore_data()

render_sidebar()
tab, = st.tabs(["Drivers"])
with tab:
    render_drivers_tab(_cached_explore_data())
    
    