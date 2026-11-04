import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <style>
            .st-key-home-title-link a,
            .st-key-home-title-link p {
                color: inherit;
                font-size: 1.4rem;
                font-weight: 700;
                text-decoration: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="home-title-link"):
            st.page_link("home.py", label="🎬 Movie Revenue ML")
        st.markdown(
            """
            **Machine Learning project**

            Predicting movie revenue with Python,
            Machine Learning and MLflow.
            """
        )
        st.divider()
        st.caption("Explore")
        st.page_link("pages/explore_data.py", label="Explore the data", icon="🎬")
        st.page_link("pages/predict_revenue.py", label="Predict revenue", icon="🤖")
        st.page_link(
            "pages/model_performance.py",
            label="Model performance",
            icon="🧪",
        )
        st.divider()
        st.caption("Built with Python · Streamlit · MLflow")