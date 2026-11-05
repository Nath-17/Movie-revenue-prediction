import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.data import load_explore_data
from app.pages.explore.compare_tab import render_compare_tab
from app.pages.explore.drivers_tab import render_drivers_tab
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="What drives revenue?",
    layout="wide",
    initial_sidebar_state="expanded",
)


HEADER_HEIGHT = "3.75rem"  # height of Streamlit's fixed top header
TAB_BAR_BACKGROUND = {"light": "#FFFFFF", "dark": "#0E1117"}
tab_bar_background = TAB_BAR_BACKGROUND.get(st.context.theme.type, "#FFFFFF")

st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: calc({HEADER_HEIGHT} + 0.5rem);
        padding-bottom: 0.75rem;
    }}
    .block-container [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
    .block-container h1 {{ font-size: 1.8rem; margin-bottom: 0; }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    [data-testid="stMetricLabel"] {{ font-size: 0.8rem; }}
    /* Keep the tab bar visible while scrolling, just below Streamlit's header. */
    .stTabs [data-baseweb="tab-list"] {{
        position: sticky;
        top: {HEADER_HEIGHT};
        z-index: 999;
        background-color: {tab_bar_background};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _cached_explore_data() -> pd.DataFrame:
    return load_explore_data()


render_sidebar()
drivers_tab, compare_tab = st.tabs(["Drivers", "Compare"])
explore_df = _cached_explore_data()
with drivers_tab:
    render_drivers_tab(explore_df)
with compare_tab:
    render_compare_tab(explore_df)
