import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_raw_data, preprocess_data, get_train_test_split, apply_clr_transformation

def app():
    st.title("Data Preprocessing")
    
    st.markdown("""
    This section demonstrates the data preprocessing pipeline applied to the microbiome data.
    """)
    
    tabs = st.tabs(["Raw Data", "Encoding", "Missing Values", "Train-Test Split", "CLR Transformation"])
    
    with tabs[0]:
        st.header("Raw Data Overview")
        
        with st.spinner("Loading raw data..."):
            data, metadata = load_raw_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Abundance Data")
            st.write(f"Shape: {data.shape}")
            st.dataframe(data.head(10), use_container_width=True)
            
            st.metric("Total Features (Clades)", data.shape[0])
            st.metric("Total Samples", len([c for c in data.columns if c.startswith('mpa411_')]))
        
        with col2:
            st.subheader("Metadata")
            st.write(f"Shape: {metadata.shape}")
            st.dataframe(metadata.head(10), use_container_width=True)
            
            st.metric("Total Metadata Records", metadata.shape[0])
            st.metric("Metadata Columns", metadata.shape[1])
    
    with tabs[1]:
        st.header("Label Encoding")
        
        st.markdown("""
        Categorical variables are encoded into numeric values for machine learning models.
        """)
        
        with st.spinner("Preprocessing data..."):
            encoded_samples, le_age, le_sex, le_family, merged_samples = preprocess_data()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Sex Encoding")
            sex_mapping = pd.DataFrame({
                'Original': le_sex.classes_,
                'Encoded': range(len(le_sex.classes_))
            })
            st.dataframe(sex_mapping, use_container_width=True)
        
        with col2:
            st.subheader("Age Group Encoding")
            age_mapping = pd.DataFrame({
                'Original': le_age.classes_,
                'Encoded': range(len(le_age.classes_))
            })
            st.dataframe(age_mapping, use_container_width=True)
        
        with col3:
            st.subheader("Family ID")
            st.metric("Unique Families", len(le_family.classes_))
            st.write("Families are encoded numerically")
        
        st.subheader("Encoded Dataset Preview")
        preview_cols = ['sample_id', 'family_id', 'sex', 'age_group_at_sample', 'age_group_encoded']
        available_cols = [col for col in preview_cols if col in encoded_samples.columns]
        st.dataframe(encoded_samples[available_cols].head(10), use_container_width=True)
    
    with tabs[2]:
        st.header("Missing Value Analysis")
        
        with st.spinner("Analyzing missing values..."):
            encoded_samples, le_age, le_sex, le_family, merged_samples = preprocess_data()
        
        missing_counts = encoded_samples.isnull().sum()
        missing_pct = (missing_counts / len(encoded_samples)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(missing_df)), missing_df['Missing %'].values)
            ax.set_xlabel('Columns')
            ax.set_ylabel('Missing %')
            ax.set_title('Missing Values by Column')
            st.pyplot(fig)
        else:
            st.success("No missing values found in the dataset after preprocessing!")
        
        st.metric("Total Samples After Cleaning", len(encoded_samples))
    
    with tabs[3]:
        st.header("Train-Test Split")
        
        st.markdown("""
        Data is split into training and testing sets before any transformation to prevent data leakage.
        """)
        
        with st.spinner("Splitting data..."):
            X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Set")
            st.metric("Samples", len(X_train))
            st.metric("Features", X_train.shape[1])
            
            st.write("Class Distribution:")
            train_dist = pd.DataFrame(y_train.value_counts()).reset_index()
            train_dist.columns = ['Age Group', 'Count']
            st.dataframe(train_dist, use_container_width=True)
        
        with col2:
            st.subheader("Test Set")
            st.metric("Samples", len(X_test))
            st.metric("Features", X_test.shape[1])
            
            st.write("Class Distribution:")
            test_dist = pd.DataFrame(y_test.value_counts()).reset_index()
            test_dist.columns = ['Age Group', 'Count']
            st.dataframe(test_dist, use_container_width=True)
        
        st.info(f"Split ratio: 80% training, 20% testing (stratified by age group)")
    
    with tabs[4]:
        st.header("CLR Transformation")
        
        st.markdown("""
        Centered Log-Ratio (CLR) transformation is applied to handle compositional nature of microbiome data.
        
        **Formula**: CLR(x) = log(x / geometric_mean(x))
        """)
        
        with st.spinner("Applying CLR transformation..."):
            X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
            X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before CLR")
            sample_feature = X_train.iloc[:, 0]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sample_feature[sample_feature > 0], bins=50, edgecolor='black')
            ax.set_xlabel('Abundance')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution Before CLR')
            st.pyplot(fig)
            
            st.metric("Mean", f"{sample_feature.mean():.4f}")
            st.metric("Std Dev", f"{sample_feature.std():.4f}")
        
        with col2:
            st.subheader("After CLR")
            sample_feature_clr = X_train_clr.iloc[:, 0]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sample_feature_clr, bins=50, edgecolor='black')
            ax.set_xlabel('CLR-transformed Abundance')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution After CLR')
            st.pyplot(fig)
            
            st.metric("Mean", f"{sample_feature_clr.mean():.4f}")
            st.metric("Std Dev", f"{sample_feature_clr.std():.4f}")
        
        st.success("CLR transformation helps normalize the data and accounts for the compositional nature of microbiome data.")
