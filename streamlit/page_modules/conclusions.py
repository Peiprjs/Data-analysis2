import streamlit as st


def app():
    st.title("Conclusions")
    st.markdown("## Key findings and next steps")

    st.markdown(
        """
        **Summary of insights**
        - The LucKi cohort subset comprises 930 samples with ~6,900 features; data are sparse with ~300 taxa per sample.
        - An age-related signal exists but explains a limited share of variance; PCA shows a gradient rather than distinct clusters.
        - Model performance improves with CLR normalization and genus-level feature filtering.
        - Tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting, LightGBM) provide competitive baselines.
        - Feature importance and SHAP analysis consistently highlight several key bacterial genera as major contributors to age prediction.
        
        **Significant bacteria affecting age prediction:**
        
        Examples of genera that consistently show high feature importance in the models include:
        - Staphylococcus: Helps stabilise the gut microbiota.
        - Citrobacter: May aid in digesting certain compounds.
        - Bifidobacterium: 
        - Cornyebacterium: 
        - Lacrimispora: 
            
        These genera represent the most influential features in predicting age groups and reflect 
        known biological patterns of microbiome development and aging.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **FAIR and accessibility notes**
        - Findable & Accessible: Data paths and preprocessing steps are documented; navigation uses clear labels.
        - Interoperable: Standard CSV inputs, encoded labels, and reproducible splits. Known metagenomic format.
        - Reusable: Cached preprocessing, documented transformations, and model summaries enable re-analysis. Code is open source, allowing for easy validation.
        - Accessibility: Descriptive text accompanies metrics and tables; no emojis are used.
        """
    )
    st.markdown("---")

    st.subheader("Possible Limitations and Improvements")
    st.markdown(
        """
        - **Data scope:** Current models are trained on a single cohort; performance may vary on new populations, based on geography, culture, religion...
        - **Temporal drift:** Microbiome profiles can change over time; periodic re-training could be useful.
        - **Confounding factors:** Diet, antibiotics, and environment are not fully captured and may bias predictions.
        - **Future work:** Performing external validation, longitudinal modeling, and uncertainty estimates for predictions would improve its applicability in clinical practice.
        """
    )