import streamlit as st


def app():
    st.title("FAIRness")
    st.markdown(
        """
        This study follows the **FAIR principles** to ensure data and models remain
        findable, accessible, interoperable, and reusable.

        - **Findable:** Clear naming, stable repository links, and versioned releases.
        - **Accessible:** Open-source code and documentation with public issue tracking.
        - **Interoperable:** Standard CSV/tabular formats and taxonomic naming conventions.
        - **Reusable:** Documented preprocessing (encoding, CLR), genus-level feature set, and cached splits.
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        """
        - Single-cohort data; external validation is limited.
        - Microbiome composition can drift over time and geography.
        - Predictions can be biased by class imbalance or unobserved confounders.
        - Model performance may degrade outside the observed age ranges.
        """
    )

    st.markdown("### Possible Improvements")
    st.markdown(
        """
        - Add external validation cohorts and longitudinal follow-up.
        - Calibrate models for out-of-distribution detection.
        - Incorporate dietary and clinical covariates to reduce confounding.
        - Expand interpretability with cohort-specific SHAP summaries.
        """
    )
