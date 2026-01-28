import streamlit as st


def app():
    st.title("Conclusions")
    st.markdown("## Key findings and next steps")

    st.markdown(
        """
        **Summary of insights**
        - The LucKi cohort subset comprises 930 samples with ~6,900 features; data are sparse with ~300 taxa per sample.
        - Age-related signal exists but explains a limited share of variance; PCA shows a gradient rather than distinct clusters.
        - Model performance improves with CLR normalization and genus-level feature filtering.
        - Tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting, LightGBM) provide competitive baselines.
        - Feature importance and SHAP/LIME highlight prevalent genera as key contributors.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **FAIR and accessibility notes**
        - Findable & Accessible: Data paths and preprocessing steps are documented; navigation uses clear labels.
        - Interoperable: Standard CSV inputs, encoded labels, and reproducible splits.
        - Reusable: Cached preprocessing, documented transformations, and model summaries enable re-analysis.
        - Accessibility: Descriptive text accompanies metrics and tables; no emojis are used.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **Potential next steps**
        - Explore domain-adapted neural architectures for microbiome data.
        - Incorporate uncertainty estimates (e.g., quantile regression forests).
        - Extend interpretability with cohort-specific subgroup analyses.
        """
    )

    st.success("This page concludes the summarized insights from the notebooks.")
