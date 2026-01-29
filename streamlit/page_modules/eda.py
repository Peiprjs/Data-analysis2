import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.functions import get_eda_summaries


def app():
    st.title("Exploratory Data Analysis")
    st.markdown("## Dataset overview and key patterns")

    summaries = get_eda_summaries()

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

    # Interactive section for dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{summaries['data_shape'][0]}")
    with col2:
        st.metric("Total Features", f"{summaries['data_shape'][1]}")
    with col3:
        st.metric("Metadata Columns", f"{summaries['metadata_shape'][1]}")

    # Interactive visualization for samples per family
    st.markdown("---")
    st.markdown("###Samples per Family")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Create interactive bar chart
        fig = px.bar(
            summaries["samples_per_family"].head(20), 
            x="Family", 
            y="Samples",
            title="Top 20 Families by Sample Count",
            color="Samples",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            xaxis_title="Family ID",
            yaxis_title="Number of Samples",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Family Statistics**")
        st.dataframe(
            summaries["samples_per_family"].describe().reset_index().rename(columns={"index": "Stat", "Samples": "Value"}),
            use_container_width=True
        )

    # Interactive age group distribution
    st.markdown("---")
    st.markdown("###Age Group Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for age groups
        fig = px.pie(
            summaries["age_groups"], 
            values="Count", 
            names="Age Group",
            title="Age Group Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart for age groups
        fig = px.bar(
            summaries["age_groups"], 
            x="Age Group", 
            y="Count",
            title="Sample Count by Age Group",
            color="Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Interactive taxa analysis
    st.markdown("---")
    st.markdown("###Taxa Analysis")
    
    # Add filter for viewing raw data
    show_details = st.checkbox("Show detailed statistics", value=False)
    
    if show_details:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Taxa per Sample Statistics**")
            st.dataframe(
                pd.DataFrame(summaries["taxa_per_sample_stats"]).reset_index().rename(columns={"index": "Statistic", 0: "Value"}),
                use_container_width=True,
            )
        
        with col2:
            st.markdown("**Feature Prevalence Across Samples**")
            st.dataframe(
                pd.DataFrame(summaries["feature_prevalence"]).reset_index().rename(columns={"index": "Statistic", 0: "Value"}),
                use_container_width=True,
            )
    
    # Sample distribution visualization
    st.markdown("###Sample Metadata Exploration")
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Samples by Sex", "Samples by Age and Sex", "Family Size Distribution"]
    )
    
    merged_data = summaries["merged"]
    
    if viz_type == "Samples by Sex":
        sex_counts = merged_data['sex'].value_counts().reset_index()
        sex_counts.columns = ['Sex', 'Count']
        fig = px.bar(
            sex_counts,
            x='Sex',
            y='Count',
            title='Sample Distribution by Sex',
            color='Sex',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Samples by Age and Sex":
        age_sex_counts = merged_data.groupby(['age_group_at_sample', 'sex']).size().reset_index(name='Count')
        fig = px.bar(
            age_sex_counts,
            x='age_group_at_sample',
            y='Count',
            color='sex',
            title='Sample Distribution by Age Group and Sex',
            barmode='group',
            labels={'age_group_at_sample': 'Age Group', 'sex': 'Sex'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Family Size Distribution":
        family_sizes = merged_data['family_id'].value_counts()
        size_distribution = family_sizes.value_counts().sort_index().reset_index()
        size_distribution.columns = ['Samples per Family', 'Number of Families']
        fig = px.bar(
            size_distribution,
            x='Samples per Family',
            y='Number of Families',
            title='Distribution of Family Sizes',
            color='Number of Families',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The EDA summary is cached for performance. Values are computed directly from the raw data for reproducibility. "
        "Use the interactive controls above to explore different aspects of the dataset."
    )
