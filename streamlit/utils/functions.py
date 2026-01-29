"""
Utility functions for data analysis and model training.
This module contains helper functions used across multiple pages.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .data_loader import load_raw_data, preprocess_data


@st.cache_data(show_spinner="Loading EDA summaries...")
def get_eda_summaries():
    """
    Load and compute summary statistics for exploratory data analysis.
    
    Returns
    -------
    dict
        Dictionary containing data shapes, family distributions, age groups,
        taxa statistics, and feature prevalence information.
    """
    data, metadata = load_raw_data()
    encoded_samples, _, _, _, merged = preprocess_data()

    samples_per_family = merged['family_id'].value_counts().rename_axis("Family").reset_index(name="Samples")
    age_groups = merged['age_group_at_sample'].value_counts().rename_axis("Age Group").reset_index(name="Count")
    taxa_per_sample = (data.filter(regex="^mpa411_").astype(bool).sum()).describe()
    feature_prevalence = (data.filter(regex="^mpa411_") > 0).mean(axis=1)

    return {
        "data": data,
        "metadata": metadata,
        "merged": merged,
        "data_shape": data.shape,
        "metadata_shape": metadata.shape,
        "samples_per_family": samples_per_family,
        "age_groups": age_groups,
        "taxa_per_sample_stats": taxa_per_sample,
        "feature_prevalence": feature_prevalence.describe()
    }


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train a machine learning model and compute evaluation metrics.
    
    Parameters
    ----------
    model : sklearn-compatible model
        Machine learning model with fit and predict methods.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.
    model_name : str
        Name of the model for reporting.
    
    Returns
    -------
    dict
        Dictionary containing model name and performance metrics (RMSE, R2, MAE)
        for both training and test sets.
    """
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return {
        'Model': model_name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R2': train_r2,
        'Test R2': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    }
