import streamlit as st
import pandas as pd

def app():
    st.title("Microbiome Data Analysis")
    st.markdown("## LucKi Cohort Analysis Platform")
    
    st.markdown("""
    ### Overview
    
    This application analyzes and predicts microbiome data on a selected subset from the LucKi cohort. 
    The analysis focuses on bacterial abundance patterns across different age groups and samples, 
    using MetaPhlAn 4.1.1 taxonomic profiles.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", "930")
        st.metric("Features", "~6,900")
    
    with col2:
        st.metric("Age Groups", "Multiple")
        st.metric("Taxa per Sample", "~300")
    
    with col3:
        st.metric("Test Split", "20%")
        st.metric("Models Tested", "5+")
    
    st.markdown("---")
    
    st.markdown("""
    ### Key Features
    
    #### Data Integration
    - Merges abundance tables with sample metadata
    - Handles high-dimensional microbiome data
    
    #### Preprocessing Pipeline
    - CLR transformation for compositional data
    - Label encoding for categorical variables
    - Missing value handling and outlier detection
    
    #### Machine Learning
    - Random Forest regression models
    - XGBoost and LightGBM implementations
    - Neural network-based feature selection
    
    #### Model Interpretability
    - LIME (Local Interpretable Model-agnostic Explanations)
    - SHAP (SHapley Additive exPlanations)
    - Feature importance analysis
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Dataset Characteristics
    
    The LucKi cohort subdataset consists of **930 stool samples** collected from multiple individuals 
    across different families. Key characteristics include:
    
    - **High-dimensional data**: Approximately 6,900 microbiome features
    - **Sparse dataset**: Each sample contains on average ~1,200 genera and ~300 taxa
    - **Consistent sequencing depth**: Total microbial abundance per sample is relatively stable
    - **Rare taxa dominance**: Most taxa occur in only a small fraction of samples
    - **Log-normal distribution**: Non-zero abundances follow an approximately log-normal distribution
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Analysis Workflow
    
    1. **Data Loading**: Import raw abundance data and metadata
    2. **Data Merging**: Combine abundance profiles with sample metadata
    3. **Preprocessing**: Label encoding, missing value handling, outlier detection
    4. **Train-Test Split**: Separate data before further processing
    5. **Normalization**: Apply CLR transformation to abundance data
    6. **Exploratory Data Analysis**: Analyze distributions and patterns
    7. **Feature Selection**: Filter features at the genus level
    8. **Model Training**: Train and evaluate multiple ML models
    9. **Interpretability**: Explain model predictions using LIME and SHAP
    """)
    
    st.markdown("---")
    
    st.info("""
    **Navigation**: Use the sidebar to explore different sections of the analysis:
    - **Data Preprocessing**: View data transformation steps
    - **Model Training**: Explore different ML models and their performance
    - **Model Interpretability**: Understand model predictions
    - **Results Comparison**: Compare model performance across different configurations
    """)
