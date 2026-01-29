"""
Data loading and preprocessing utilities for microbiome analysis.

This module provides cached functions for loading, preprocessing, and transforming
microbiome data from the LucKi cohort. All functions use Streamlit caching for
improved performance.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


@st.cache_data(show_spinner="Loading raw data...")
def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw microbiome abundance data and sample metadata.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - data : pd.DataFrame
            MetaPhlAn 4.1.1 abundance table (6903 rows × 932 columns)
            Rows are taxonomic clades, columns are samples
        - metadata : pd.DataFrame
            Sample metadata (930 rows × 6 columns)
            Contains sample_id, family_id, sex, age_group_at_sample, etc.
    
    Notes
    -----
    Data files are expected to be in the repository-level ``data/raw/`` directory.
    The abundance table was manually converted from TSV to CSV for proper pandas
    loading.
    
    Examples
    --------
    >>> data, metadata = load_raw_data()
    >>> print(data.shape)
    (6903, 932)
    >>> print(metadata.shape)
    (930, 6)
    """
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    
    data_path = os.path.join(base_path, 'MAI3004_lucki_mpa411.csv')
    metadata_path = os.path.join(base_path, 'MAI3004_lucki_metadata_safe.csv')
    
    data = pd.read_csv(data_path, index_col=0)
    metadata = pd.read_csv(metadata_path)
    
    return data, metadata


@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data() -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, LabelEncoder, pd.DataFrame]:
    """
    Preprocess microbiome data with encoding and cleaning steps.
    
    Performs the following operations:
    1. Transpose abundance table to sample × feature format
    2. Merge with metadata
    3. Remove unnecessary columns (year_of_birth, body_product)
    4. Label encode categorical variables (sex, family_id, age_group)
    5. Remove samples with missing age group information
    
    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder, LabelEncoder, LabelEncoder, pd.DataFrame]
        A tuple containing:
        - encoded_samples : pd.DataFrame
            Preprocessed data with encoded categorical variables
        - le_age : LabelEncoder
            Fitted encoder for age groups
        - le_sex : LabelEncoder
            Fitted encoder for sex
        - le_family : LabelEncoder
            Fitted encoder for family IDs
        - merged_samples : pd.DataFrame
            Merged data before encoding (for reference)
    
    Notes
    -----
    Label encoding is used instead of one-hot encoding to preserve ordinal
    relationships and reduce dimensionality. Missing age group samples are
    dropped as they cannot be used for supervised learning.
    
    Examples
    --------
    >>> encoded_samples, le_age, le_sex, le_family, merged = preprocess_data()
    >>> print(encoded_samples['age_group_encoded'].unique())
    [0 1 2 3 4]
    """
    data, metadata = load_raw_data()
    
    sample_cols = [col for col in data.columns if col.startswith('mpa411_')]
    sample_abundances = data[sample_cols].T
    sample_abundances.columns = data.index
    sample_abundances.index = sample_abundances.index.str.replace('mpa411_', '')
    sample_abundances.index.name = 'sample_id'
    
    metadata_common = metadata[metadata['sample_id'].isin(sample_abundances.index)].copy()
    
    merged_samples = pd.merge(
        metadata_common,
        sample_abundances,
        left_on='sample_id',
        right_index=True,
        how='inner'
    )
    
    merged_samples = merged_samples.drop(columns=['year_of_birth', 'body_product'])
    
    encoded_samples = merged_samples.copy()
    
    le_sex = LabelEncoder()
    encoded_samples['sex'] = le_sex.fit_transform(encoded_samples['sex'])
    
    le_family = LabelEncoder()
    encoded_samples['family_id'] = le_family.fit_transform(encoded_samples['family_id'])
    
    encoded_samples = encoded_samples.dropna(subset=['age_group_at_sample'])
    
    le_age = LabelEncoder()
    encoded_samples['age_group_encoded'] = le_age.fit_transform(
        encoded_samples['age_group_at_sample']
    )
    
    return encoded_samples, le_age, le_sex, le_family, merged_samples


@st.cache_data(show_spinner="Splitting data...")
def get_train_test_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Split data into stratified training and testing sets.
    
    Creates an 80/20 train-test split with stratification by age group to
    maintain class balance. Features exclude metadata columns.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]
        A tuple containing:
        - X_train : pd.DataFrame
            Training features (80% of data)
        - X_test : pd.DataFrame
            Test features (20% of data)
        - y_train : pd.Series
            Training labels (encoded age groups)
        - y_test : pd.Series
            Test labels (encoded age groups)
        - feature_cols : List[str]
            List of feature column names
    
    Notes
    -----
    Random state is fixed at 42 for reproducibility. Stratification ensures
    that all age groups are represented proportionally in both train and test sets.
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test, features = get_train_test_split()
    >>> print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    Train size: 744, Test size: 186
    >>> print(f"Number of features: {len(features)}")
    Number of features: 6903
    """
    encoded_samples, le_age, le_sex, le_family, merged_samples = preprocess_data()
    
    feature_cols = [
        col for col in encoded_samples.columns 
        if col not in ['sample_id', 'family_id', 'sex', 'age_group_at_sample', 'age_group_encoded']
    ]
    
    X = encoded_samples[feature_cols]
    y = encoded_samples['age_group_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


@st.cache_data(show_spinner="Applying CLR transformation...")
def apply_clr_transformation(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Centered Log-Ratio (CLR) transformation to microbiome abundance data.
    
    The CLR transformation is used to account for the compositional nature of
    microbiome relative abundance data. It normalizes each sample by its
    geometric mean before taking the log.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix with abundance values (samples × features)
    X_test : pd.DataFrame
        Test feature matrix with abundance values (samples × features)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        CLR-transformed training and test datasets
    
    Notes
    -----
    The CLR transformation formula is:
        CLR(x) = log(x / geometric_mean(x))
    
    A small pseudocount (1e-6) is added to avoid log(0). This transformation:
    - Accounts for compositional constraints (values sum to 100%)
    - Makes data more normally distributed
    - Reduces the influence of sampling depth variation
    
    References
    ----------
    Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
    Chapman and Hall.
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test, _ = get_train_test_split()
    >>> X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
    >>> print(f"CLR mean: {X_train_clr.mean().mean():.4f}")
    CLR mean: 0.0000
    """
    def clr_transform(X: pd.DataFrame) -> pd.DataFrame:
        """
        Internal function to apply CLR transformation to a single dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input abundance data
        
        Returns
        -------
        pd.DataFrame
            CLR-transformed data
        """
        X_clr = X.copy()
        X_clr = X_clr + 1e-6
        geo_mean = np.exp(np.log(X_clr).mean(axis=1))
        X_clr = np.log(X_clr.div(geo_mean, axis=0))
        return X_clr
    
    X_train_clr = clr_transform(X_train)
    X_test_clr = clr_transform(X_test)
    
    return X_train_clr, X_test_clr


@st.cache_data(show_spinner="Filtering genus-level features...")
def filter_genus_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only genus-level taxonomic features.
    
    Selects features that contain the genus-level marker '|g__' but exclude
    species-level marker '|s__'. This reduces dimensionality while maintaining
    biological interpretability.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix with all taxonomic levels
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset containing only genus-level features
    
    Notes
    -----
    MetaPhlAn 4 taxonomic format uses pipe-delimited levels:
    - k__ : Kingdom
    - p__ : Phylum
    - c__ : Class
    - o__ : Order
    - f__ : Family
    - g__ : Genus
    - s__ : Species
    
    Genus-level features provide a good balance between:
    - Feature dimensionality (manageable for ML)
    - Biological resolution (meaningful taxonomic groups)
    - Data availability (more prevalent than species)
    
    Examples
    --------
    >>> X_train, X_test, _, _, _ = get_train_test_split()
    >>> X_train_genus = filter_genus_features(X_train)
    >>> print(f"Original features: {X_train.shape[1]}")
    Original features: 6903
    >>> print(f"Genus features: {X_train_genus.shape[1]}")
    Genus features: 412
    """
    genus_cols = [col for col in X.columns if '|g__' in col and '|s__' not in col]
    return X[genus_cols]
