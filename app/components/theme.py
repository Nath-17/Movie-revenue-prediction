import plotly.graph_objects as go

ACCENT = "#2A788E"
ACCENT_LIGHT = "rgba(42,120,142,0.18)"
GREY = "#858C91"
GREY_LIGHT = "#D9DEE1"
CATEGORY = ("#0072B2", "#D55E00")

FONT_FAMILY = "Arial, sans-serif"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 13
TICK_SIZE = 12
MILLION = 1_000_000
STORY_CHART_HEIGHT = 370
RANKING_CHART_HEIGHT = 220
RANKING_CHART_MARGIN = {"l": 72, "r": 18, "t": 48, "b": 42}
RANKING_AXIS_MAX = 1.12
RANKING_AXIS_TICKS = (0, 0.2, 0.4, 0.6, 0.8, 1)
LOG_TICK_TARGET_COUNT = 6
ASSOCIATION_LABEL_DECIMALS = 2
GENRE_LABEL_DECIMALS = 1
MONEY_LABEL_DECIMALS = 1
SCATTER_MARKER_SIZE = 6
SCATTER_MARKER_OPACITY = 0.38
TREND_LINE_WIDTH = 3
TREND_LINE_POINTS = 100
SEASONAL_LINE_WIDTH = 3
SEASONAL_MARKER_SIZE = 7
HIDDEN_LINE_WIDTH = 0

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY},
        title={"font": {"size": TITLE_SIZE}},
        xaxis={
            "showline": False,
            "zeroline": False,
            "gridcolor": GREY_LIGHT,
            "title": {"font": {"size": AXIS_LABEL_SIZE}},
            "tickfont": {"size": TICK_SIZE},
        },
        yaxis={
            "showline": False,
            "zeroline": False,
            "gridcolor": GREY_LIGHT,
            "title": {"font": {"size": AXIS_LABEL_SIZE}},
            "tickfont": {"size": TICK_SIZE},
        },
        margin={"l": 72, "r": 28, "t": 58, "b": 48},
        height=STORY_CHART_HEIGHT,
        hoverlabel={"font": {"family": FONT_FAMILY}},
    )
)


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply transparent backgrounds and consistent typography to a figure."""
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig
