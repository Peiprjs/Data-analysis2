import streamlit as st


def app():
    st.title("Introduction")
    st.markdown("## LucKi Cohort Microbiome Study Overview")

    st.markdown(
        """
        This app summarizes findings from the project notebooks **Finalized Models** and **data_analysis**.
        The dataset is a subset of the LucKi cohort, consisting of **930 stool samples** with
        approximately **6,900 microbiome features** derived from MetaPhlAn 4.1.1 taxonomic profiles.

        We summarize key steps of the analysis, keeping the information **FAIR**:
        - **Findable:** Clear naming, stable links, and identifiers for data sources.
        - **Accessible:** Public repository links and concise summaries for non-specialists.
        - **Interoperable:** Standard tabular formats (CSV), taxonomic feature names, and label encodings.
        - **Reusable:** Documented preprocessing (encoding, CLR transformation), splits, and model choices.
        """
    )

    st.markdown(
        """
        **How to cite this work**

        - Title: *Data Analysis Pipeline for Microbial Community Profiling*
        - Authors: Mar Roca Cugat; Lucien Santiago; Jacob Záboj; Tom Einhaus; Julie Kretzers
        - Version: 1.0.0 (released 2026-01-19)
        - DOI: [10.5281/zenodo.18302927](https://doi.org/10.5281/zenodo.18302927)
        """
    )

    st.markdown("---")
    st.subheader("Cohort and Data Source")
    st.markdown(
        """
        The cohort contains multiple families and age groups. Metadata includes sample ID, family ID,
        sex, and age group at sample collection. See the related cohort description in
        [Lutz et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC4578419/).
        """
    )

    st.markdown("---")
    st.subheader("Analysis Workflow Recap")
    st.markdown(
        """
        1. **Data loading:** Merge MetaPhlAn abundance table with sample metadata (6903 × 932 -> 930 × 6903 after transpose).
        2. **Preprocessing:** Encode categorical variables (sex, family, age group), drop missing age group rows.
        3. **Train/test split:** 80/20 stratified by age group for reproducibility.
        4. **Normalization:** Apply Centered Log-Ratio (CLR) transformation to handle compositional data.
        5. **Feature filtering:** Restrict to genus-level features for modeling efficiency.
        6. **Modeling:** Evaluate Random Forest, XGBoost, Gradient Boosting, LightGBM, and Neural Networks.
        7. **Interpretation:** Use SHAP/LIME and feature importance to understand drivers.
        """
    )

    st.info(
        "Use the sidebar to navigate to Exploratory Data Analysis, Models, and Conclusions pages. "
        "Alt text is included for plots when relevant to improve accessibility."
    )
