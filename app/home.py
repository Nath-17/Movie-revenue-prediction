import streamlit as st


st.set_page_config(
    page_title="Movie Revenue ML",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def navigate_to(page: str) -> None:
    st.switch_page(page)


st.title("🎬 Movie Revenue ML")

st.markdown(
    """
    ## Can we predict how much a movie will earn?

    This project explores the use of **Machine Learning** to predict
    movie revenue from characteristics such as budget, popularity,
    ratings and other movie metadata.

    The application brings together:

    - 📊 **Data exploration and visualization**
    - 🤖 **Movie revenue prediction**
    - 🧪 **Model evaluation and comparison**
    - 📦 **Experiment tracking with MLflow**
    """
)

st.divider()

st.subheader("Explore the project")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        ### 🎬 Explore

        Explore the movie dataset through interactive visualizations.

        Analyze relationships between:

        - Budget
        - Revenue
        - Genres
        - Ratings
        - Popularity
        """
    )

    if st.button(
        "Explore the data →",
        key="explore",
        use_container_width=True,
    ):
        navigate_to("pages/explore_data.py")


with col2:
    st.markdown(
        """
        ### 🤖 Predict

        Enter the characteristics of a movie and let the
        trained Machine Learning model estimate its revenue.

        Try different scenarios and see how the prediction changes.
        """
    )

    if st.button(
        "Predict revenue →",
        key="predict",
        use_container_width=True,
    ):
        navigate_to("pages/predict_revenue.py")


with col3:
    st.markdown(
        """
        ### 🧪 Evaluate

        Compare different experiments and models using metrics
        tracked with MLflow.

        Analyze predictions and model errors.
        """
    )

    if st.button(
        "Evaluate models →",
        key="evaluate",
        use_container_width=True,
    ):
        navigate_to("pages/model_performance.py")


st.divider()

with st.sidebar:
    st.title("🎬 Movie Revenue ML")

    st.markdown(
        """
        **Machine Learning project**

        Predicting movie revenue with Python,
        Machine Learning and MLflow.
        """
    )

    st.divider()

    st.caption("Explore")

    st.page_link(
        "pages/explore_data.py",
        label="Explore the data",
        icon="🎬",
    )

    st.page_link(
        "pages/predict_revenue.py",
        label="Predict revenue",
        icon="🤖",
    )

    st.page_link(
        "pages/model_performance.py",
        label="Model performance",
        icon="🧪",
    )

    st.divider()

    st.caption("Built with Python · Streamlit · MLflow")
