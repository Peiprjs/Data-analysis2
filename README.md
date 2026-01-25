# Microbiome Data Analysis Platform

![image](https://img.shields.io/badge/python-3.8%2B-blue) 
![image](https://img.shields.io/badge/license-AGPL%203.0-green) 
![image](https://img.shields.io/badge/code%20style-black-black)

## Overview

A comprehensive platform for analyzing microbiome data from the LucKi cohort,
featuring machine learning models for age group prediction from gut microbiome
taxonomic profiles. The platform includes both interactive Streamlit web
application and Jupyter notebook-based analysis.

### Keywords

`microbiome` `metagenomics` `machine-learning` `bioinformatics` `MetaPhlAn` `
age-prediction` `taxonomic-profiling` `compositional-data` `CLR-transformation` `
feature-selection` `model-interpretability` `LIME` `SHAP` `Random-Forest` `
XGBoost` `neural-networks` `streamlit` `python` `data-science`

### Identifiers

- **Repository**: <https://github.com/Peiprjs/Data-analysis2>
- **Project Name**: Microbiome Data Analysis Platform
- **Version**: 1.0.0
- **DOI**: 10.5281/zenodo.18302927
- **Data Source**: LucKi Cohort
- **License**: AGPL-3.0

- - -
## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

- - -
## Features

### Data Processing

- **CLR Transformation**: Handles compositional nature of microbiome data
- **Label Encoding**: Automatic encoding of categorical variables
- **Missing Value Handling**: Robust preprocessing pipeline
- **Train-Test Split**: Stratified splitting to maintain class balance

### Machine Learning Models

- **Random Forest**: Ensemble learning with decision trees
- **XGBoost**: Gradient boosting with regularization
- **Gradient Boosting**: Sequential ensemble learning
- **LightGBM**: High-efficiency gradient boosting
- **AdaBoost**: Adaptive boosting algorithm
- **Neural Networks**: Feature selection with gatekeeper layers

### Model Interpretability

- **LIME**: Local interpretable model-agnostic explanations
- **SHAP**: SHapley additive explanations
- **Feature Importance**: Analysis and visualization
- **Cross-Validation**: K-fold validation for robustness

### Interactive Platform

- **Streamlit Application**: Web-based interactive interface
- **Real-time Analysis**: Dynamic model training and evaluation
- **Visualization**: Comprehensive plotting and comparison tools
- **User-Friendly**: No coding required for basic analysis

- - -
## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU support for TensorFlow
- Minimum 8GB RAM (16GB recommended for neural network feature selection)

### Step-by-Step Installation Guide

#### 1\. Clone the Repository

```bash
git clone https://github.com/Peiprjs/Data-analysis2.git
cd Data-analysis2
```
#### 2\. Create Virtual Environment (Recommended)

**Using venv:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**Using conda:**

```bash
conda create -n microbiome python=3.10
conda activate microbiome
```
#### 3\. Install Dependencies

**For Streamlit Application:**

```bash
pip install -r requirements.txt
```
**For Jupyter Notebook Analysis:**

```bash
cd notebooks
pip install -r requirements.txt
```
**For Development (with code quality tools):**

```bash
pip install -e ".[dev]"
```
#### 4\. Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; import xgboost; print('Installation successful!')"
```
- - -
## Hardware Requirements

### Minimum Requirements


|Component|Specification            |
|---------|-------------------------|
|CPU      |2 cores, 2.0 GHz         |
|RAM      |8 GB                     |
|Storage  |2 GB available space     |
|OS       |Linux, macOS, Windows 10+|

### Recommended Requirements


|Component|Specification                                                   |
|---------|----------------------------------------------------------------|
|CPU      |4+ cores, 3.0+ GHz                                              |
|RAM      |16 GB                                                           |
|GPU      |NVIDIA GPU with 4GB+ VRAM (for neural network feature selection)|
|Storage  |5 GB available space                                            |
|OS       |Linux (Ubuntu 20.04+), macOS 11+, Windows 10+                   |

### GPU Support

- **NVIDIA GPUs**: Requires CUDA 11.2+ and cuDNN 8.1+
- **AMD GPUs**: ROCm support (experimental)
- **Apple Silicon (M1/M2)**: TensorFlow Metal plugin

**Note**: GPU is optional. All models can run on CPU, though neural network
feature selection will be slower.

- - -
## Quick Start

### Using Streamlit Application

```bash
streamlit run app.py
```
Then open your browser to http://localhost:8501

### Using Jupyter Notebook

```bash
cd notebooks
jupyter notebook data-pipeline.ipynb
```
### Command Line Analysis (Quick Demo)

```python
from utils.data_loader import get_train_test_split, apply_clr_transformation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load and preprocess data
X_train, X_test, y_train, y_test, _ = get_train_test_split()
X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_clr, y_train)

# Evaluate
score = r2_score(y_test, model.predict(X_test_clr))
print(f"Test R² Score: {score:.4f}")
```
- - -
## Usage

### Streamlit Application

The web application provides five main sections:

1.  **Home**: Project overview and dataset statistics
2.  **Data Preprocessing**: Interactive data transformation and visualization
3.  **Model Training**: Train and compare multiple ML models
4.  **Model Interpretability**: Understand model predictions with LIME/SHAP
5.  **Results Comparison**: Cross-validation and ensemble analysis

### Jupyter Notebook Workflow

The notebook is organized into sections:

1.  **Housekeeping**: Library imports and settings
2.  **Data Preprocessing**: Loading, merging, and cleaning
3.  **Exploratory Data Analysis**: Visualizations and statistics
4.  **Model Training**: Multiple ML algorithms
5.  **Feature Selection**: Neural network-based selection
6.  **Model Interpretability**: LIME and SHAP analysis
7.  **Cross-Validation**: Taxonomic level comparison

- - -
## Data Description

### Dataset Characteristics

The LucKi cohort subdataset consists of:

- **930 stool samples** from multiple individuals across different families
- **~6,900 microbiome features** (taxonomic clades)
- **MetaPhlAn 4.1.1** taxonomic profiling
- **Age groups** as target variable for prediction

### Data Files

Located in `data/raw/`:

#### MAI3004_lucki_mpa411.csv

- **Format**: CSV (converted from TSV)
- **Dimensions**: 6903 rows × 932 columns
- **Content**: Taxonomic profiles with relative abundances
- **Row Index**: Taxonomic clade names (species to kingdom level)
- **Columns**: Sample IDs prefixed with `mpa411_`
- **Values**: Relative abundance (0-100%)

**MetaPhlAn 4 Taxonomic Format**:

```
k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Lachnospiraceae|g__Blautia|s__Blautia_obeum
```
Taxonomic levels:

- `k__`: Kingdom
- `p__`: Phylum
- `c__`: Class
- `o__`: Order
- `f__`: Family
- `g__`: Genus
- `s__`: Species

#### MAI3004_lucki_metadata_safe.csv

- **Format**: CSV
- **Dimensions**: 930 rows × 6 columns
- **Content**: Sample metadata and demographics

**Columns**:

- `sample_id`: Unique sample identifier
- `family_id`: Family grouping identifier
- `sex`: Biological sex (categorical)
- `age_group_at_sample`: Age group category (target variable)
- `year_of_birth`: Birth year (removed during preprocessing)
- `body_product`: Sample type (removed during preprocessing)

### Data Quality Metrics


|Metric                    |Value                              |
|--------------------------|-----------------------------------|
|Total samples             |930                                |
|Total features            |~6,900                             |
|Average genera per sample |~1,200                             |
|Average species per sample|~300                               |
|Missing values            |Minimal (\<1%)                     |
|Data sparsity             |High (~80% zeros)                  |
|Distribution              |Log-normal (typical for microbiome)|

- - -
## Methodology

### Preprocessing Pipeline

1.  **Data Integration**
- Merge abundance table with metadata
- Filter for common samples
- Remove unnecessary columns
2.  **Encoding**
- Label encoding for categorical variables (family_id, sex, age_group)
- Preserve ordinal relationships where applicable
3.  **Quality Control**
- Missing value detection and removal
- Outlier analysis using IQR method
- Normality testing with Shapiro-Wilk
4.  **Normalization**
- CLR (Centered Log-Ratio) transformation
- Accounts for compositional nature of microbiome data
- Formula: `CLR(x) = log(x / geometric_mean(x))`
5.  **Feature Selection**
- Genus-level filtering
- Neural network-based selection (optional)
- Prevalence and variance filtering

### Machine Learning Pipeline

```
Raw Data → Preprocessing → Train/Test Split → CLR Transform → 
Feature Selection → Model Training → Evaluation → Interpretation
```
### Model Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Prediction error magnitude
- **R² Score**: Proportion of variance explained
- **MAE** (Mean Absolute Error): Average prediction error
- **Cross-Validation**: K-fold validation for robustness

- - -
## Reproducibility

### Setting Random Seeds

All analyses use fixed random seeds for reproducibility:

```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```
### Environment Specification

Full environment captured in:

- `requirements.txt`: Pinned package versions
- `pyproject.toml`: Development dependencies
- Python version: 3.8+

### Reproducing Results

1.  **Clone exact repository version**:

    ```bash
    git clone https://github.com/Peiprjs/Data-analysis2.git
    git checkout <commit-hash>  # Use specific commit for exact reproduction
    ```
2.  **Install exact dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
3.  **Run analysis**:

    ```bash
    jupyter notebook notebooks/data-pipeline.ipynb
    ```
    Or:

    ```bash
    streamlit run app.py
    ```
### Version Control

- All code changes tracked in Git
- See `CHANGELOG.md` for version history
- Tagged releases for major versions

- - -
## Project Structure

```
Data-analysis2/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── pyproject.toml              # Project configuration and dependencies
├── requirements.txt            # Python package requirements
├── LICENSE                     # MIT License
├── app.py                      # Streamlit application entry point
│
├── data/                       # Data directory
│   └── raw/                    # Raw data files
│       ├── MAI3004_lucki_mpa411.csv           # Abundance data
│       ├── MAI3004_lucki_metadata_safe.csv    # Sample metadata
│       └── metaphlan411_data_description.md   # Data format docs
│
├── notebooks/                  # Jupyter notebooks
│   ├── data-pipeline.ipynb    # Main analysis notebook
│   ├── functions.py           # Helper functions
│   └── requirements.txt       # Notebook-specific dependencies
│
├── pages/                      # Streamlit pages
│   ├── __init__.py
│   ├── home.py                # Home page
│   ├── preprocessing.py       # Data preprocessing page
│   ├── models.py              # Model training page
│   ├── interpretability.py    # Model interpretability page
│   └── results.py             # Results comparison page
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── data_loader.py         # Data loading and caching functions
│
└── outputs/                    # Analysis outputs (generated)
    └── data-pipeline-1150-2001.ipynb  # Example output
```
- - -
## Documentation

### Function Documentation

All critical functions include NumPy-style docstrings:

```python
def apply_clr_transformation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Centered Log-Ratio transformation to microbiome abundance data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix with abundance values.
    X_test : pd.DataFrame
        Test feature matrix with abundance values.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        CLR-transformed training and test datasets.
    
    Notes
    -----
    The CLR transformation accounts for the compositional nature of microbiome data.
    A small pseudocount (1e-6) is added to avoid log(0).
    """
```
### API Documentation

For detailed API documentation, see individual module docstrings:

- `utils/data_loader.py`: Data loading functions
- `notebooks/functions.py`: Analysis functions
- `pages/*.py`: Streamlit page implementations

### External Resources

- [MetaPhlAn 4 Documentation](https://github.com/biobakery/MetaPhlAn)
- [Compositional Data Analysis](https://doi.org/10.1080/02664763.2017.1389862)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [SHAP Documentation](https://github.com/slundberg/shap)

- - -
### Related Publications

The LucKi cohort is described in:

- Luckey et al. (2015). "2015 LucKi cohort description." BMC Public Health.
  DOI: 10.1186/s12889-015-2255-7

- - -
**Last Updated**: 2024-01-25 **Version**: 1.0.0

