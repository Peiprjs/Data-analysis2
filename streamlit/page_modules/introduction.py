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

    st.markdown("---")
    st.subheader("Cohort and Data Source")
    st.markdown(
        """
        The cohort contains multiple families and age groups. Metadata includes sample ID, family ID,
        sex, and age group at sample collection. See the related cohort description in
        [Lutz et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC4578419/).
        """
    )

    st.markdown("### Intended Use")
    st.markdown(
        """
        This software is intended to **predict a child's age from gut microbiome composition** and
        compare the predicted age with the actual age. A notable deviation between predicted and real
        age could flag potential gut health issues for further investigation. It is not a diagnostic
        tool and should be used to generate hypotheses in conjunction with clinical expertise.
        """
    )

    st.markdown("### Cohort Description")
    st.markdown(
        """
        - **Population:** LucKi cohort subset with 930 stool samples from children across multiple families.
        - **Features:** ~6,900 microbial abundance features (MetaPhlAn 4.1.1), reduced to genus level for modeling.
        - **Metadata:** Age group labels, sex, family identifiers, and sample identifiers.
        - **Split:** Stratified 80/20 train-test split by age group to preserve distribution.
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

    st.markdown("---")
    st.subheader("Possible Limitations and Improvements")
    st.markdown(
        """
        - **Data scope:** Current models are trained on a single cohort; performance may vary on new populations.
        - **Temporal drift:** Microbiome profiles can change over time; periodic re-training is recommended.
        - **Confounding factors:** Diet, antibiotics, and environment are not fully captured and may bias predictions.
        - **Future work:** Add external validation, longitudinal modeling, and uncertainty estimates for predictions.
        """
    )
    st.markdown("---")
    st.subheader("How to cite this work")
    st.code(
        """        
        - Title: Data Analysis Pipeline for Microbial Community Profiling
        - Authors: Tom Einhaus; Julie Kretzers; Mar Roca Cugat; Lucien Santiago; Jacob Záboj
        - Version: 1.0.0
        - DOI: [10.5281/zenodo.18302927](https://doi.org/10.5281/zenodo.18302927)
        """)
    with st.expander("Bibtex"):
        st.code("""
            @software{Lucki_Data_Analysis_Pipeline_2026,
            author = {Roca Cugat, Mar and Santiago, Lucien and Záboj, Jacob  and Einhaus, Tom  and Kretzers, Julie},
            license = {GNU AGPL v3},
            month = jan,
            title = {{Data Analysis Pipeline for Microbial Community Profiling}},
            url = {https://github.com/MAI-David/Data-analysis},
            version = {1.0.0},
            year = {2026}
            }
        """,
        language="latex"
        )
