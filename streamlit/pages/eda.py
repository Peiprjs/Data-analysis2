import streamlit as st
import pandas as pd
from utils.data_loader import load_raw_data, preprocess_data


@st.cache_data(show_spinner="Loading EDA summaries...")
def _eda_summaries():
    data, metadata = load_raw_data()
    encoded_samples, _, _, _, merged = preprocess_data()

    samples_per_family = merged['family_id'].value_counts().rename_axis("Family").reset_index(name="Samples")
    age_groups = merged['age_group_at_sample'].value_counts().rename_axis("Age Group").reset_index(name="Count")
    taxa_per_sample = (data.filter(regex="^mpa411_").astype(bool).sum()).describe()
    feature_prevalence = (data.filter(regex="^mpa411_") > 0).mean(axis=1)

    return {
        "data_shape": data.shape,
        "metadata_shape": metadata.shape,
        "samples_per_family": samples_per_family,
        "age_groups": age_groups,
        "taxa_per_sample_stats": taxa_per_sample,
        "feature_prevalence": feature_prevalence.describe()
    }


def app():
    st.title("Exploratory Data Analysis")
    st.markdown("## Dataset overview and key patterns")

    summaries = _eda_summaries()

    st.markdown(
        """
        The dataset is **high-dimensional and sparse**, with most taxa being rare.
        Metadata covers family, sex, and age group. Following the notebook, key observations include:
        - Stable total abundance per sample (consistent sequencing depth).
        - ~300 taxa detected on average per sample.
        - Non-zero abundances follow an approximately log-normal distribution.
        - PCA on prevalent taxa shows a gradual age-related gradient rather than sharp clusters.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Abundance table shape", f"{summaries['data_shape'][0]} x {summaries['data_shape'][1]}")
        st.metric("Metadata rows", f"{summaries['metadata_shape'][0]}")
        st.metric("Features", f"{summaries['data_shape'][0]}")
    with col2:
        st.markdown("### Samples per family")
        st.dataframe(summaries["samples_per_family"], use_container_width=True)

    st.markdown("---")
    st.markdown("### Age group distribution")
    st.dataframe(summaries["age_groups"], use_container_width=True)

    st.markdown("---")
    st.markdown("### Taxa per sample (count stats)")
    st.dataframe(
        pd.DataFrame(summaries["taxa_per_sample_stats"]).reset_index().rename(columns={"index": "Statistic", 0: "Value"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Feature prevalence across samples")
    st.dataframe(
        pd.DataFrame(summaries["feature_prevalence"]).reset_index().rename(columns={"index": "Statistic", 0: "Value"}),
        use_container_width=True,
    )

    st.info(
        "EDA summary is cached for performance. Values are computed directly from the raw data for reproducibility."
    )
