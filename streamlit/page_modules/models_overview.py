import streamlit as st
import pandas as pd
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features


@st.cache_data(show_spinner="Preparing model-ready data...")
def _prepare_data():
    X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
    X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
    X_train_genus = filter_genus_features(X_train_clr)
    X_test_genus = filter_genus_features(X_test_clr)
    return X_train_genus, X_test_genus, y_train, y_test


def app():
    st.title("Models")
    st.markdown("## Summary of trained and alternative models from the notebooks")

    X_train_genus, X_test_genus, y_train, y_test = _prepare_data()
    st.markdown("### Distribution of age groups in training vs test")
    dist_df = pd.concat(
        [
            pd.DataFrame({'Split': 'Train', 'Age Group': y_train}),
            pd.DataFrame({'Split': 'Test', 'Age Group': y_test}),
        ],
        ignore_index=True,
    )
    counts = (
        dist_df.groupby(['Age Group', 'Split'])
        .size()
        .reset_index(name='count')
        .pivot(index='Age Group', columns='Split', values='count')
        .fillna(0)
    )
    st.bar_chart(data=counts)

    st.markdown(
        """
        The notebook evaluates several regression models to predict age group from microbiome profiles:
        - **Random Forest** (baseline and tuned)
        - **XGBoost**
        - **Gradient Boosting**
        - **LightGBM**
        - **Neural Networks** (as an explored alternative)

        Modeling steps follow the FAIR principle by documenting preprocessing, splits, and hyperparameters.
        """
    )

    st.markdown("### Data shapes used for training")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train samples", len(X_train_genus))
        st.metric("Features (genus)", X_train_genus.shape[1])
    with col2:
        st.metric("Test samples", len(X_test_genus))
        st.metric("Age groups", len(pd.unique(y_train)))

    st.markdown("### Feature variance (top 15 genera)")
    feature_variance = X_train_genus.var().sort_values(ascending=False).head(15)
    st.bar_chart(feature_variance)

    st.markdown("---")
    st.markdown("### Key modeling notes")
    st.markdown(
        """
        - Train/test split: 80/20 stratified by age group (random_state=42).
        - Features: CLR-transformed abundances filtered to genus level to reduce dimensionality.
        - Metrics: RMSE, R², and MAE reported in the notebooks for baseline vs tuned models.
        - Interpretability: SHAP/LIME and feature importance highlight influential genera.
        """
    )

    st.info(
        "Training data and transformations are cached for responsiveness. "
        "Use the Conclusions page for a concise takeaway of model performance."
    )
