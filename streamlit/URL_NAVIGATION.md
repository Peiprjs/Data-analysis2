# URL Navigation Guide

## Overview

The Streamlit application now supports direct links to specific sections using URL query parameters. This allows users to share links that open the application directly to a particular page.

## Usage

### Basic Syntax

To navigate directly to a specific page, append `?page=<page-identifier>` to the application URL:

```
http://localhost:8501/?page=<page-identifier>
```

### Supported Page Identifiers

The following URL-friendly identifiers can be used to navigate to different sections:

| URL Parameter | Target Page | Example |
|--------------|-------------|---------|
| `introduction` | Introduction | `http://localhost:8501/?page=introduction` |
| `fairness` | FAIRness | `http://localhost:8501/?page=fairness` |
| `eda` | Exploratory Data Analysis | `http://localhost:8501/?page=eda` |
| `exploratory-data-analysis` | Exploratory Data Analysis | `http://localhost:8501/?page=exploratory-data-analysis` |
| `models-overview` | Models Overview | `http://localhost:8501/?page=models-overview` |
| `model-training` | Model Training | `http://localhost:8501/?page=model-training` |
| `models` | Model Training | `http://localhost:8501/?page=models` |
| `interpretability` | Model Interpretability | `http://localhost:8501/?page=interpretability` |
| `model-interpretability` | Model Interpretability | `http://localhost:8501/?page=model-interpretability` |
| `conclusions` | Conclusions | `http://localhost:8501/?page=conclusions` |

## Examples

### Direct Link to Model Interpretability

```
http://localhost:8501/?page=interpretability
```

This will open the application directly to the "Model Interpretability" section, allowing users to immediately access SHAP analysis, feature importance, and sample exploration tools.

### Direct Link to Exploratory Data Analysis

```
http://localhost:8501/?page=eda
```

This will open the application directly to the "Exploratory Data Analysis" section.

### Multiple Aliases

Some pages have multiple URL identifiers for convenience:

- **Exploratory Data Analysis**: Can be accessed via `eda` or `exploratory-data-analysis`
- **Model Training**: Can be accessed via `models` or `model-training`
- **Model Interpretability**: Can be accessed via `interpretability` or `model-interpretability`

## Behavior

### Default Page

If no `page` parameter is provided or if an invalid parameter is specified, the application will default to the "Introduction" page.

### Case Insensitivity

Page identifiers are case-insensitive. The following URLs are equivalent:

```
http://localhost:8501/?page=interpretability
http://localhost:8501/?page=Interpretability
http://localhost:8501/?page=INTERPRETABILITY
```

### Error Handling

If an unknown page identifier is provided, the application will:
1. Log a debug message indicating the unknown parameter
2. Fall back to the default "Introduction" page
3. Continue to function normally

## Use Cases

1. **Sharing Specific Sections**: Share a direct link to a specific analysis section with colleagues
2. **Documentation Links**: Create documentation that points directly to relevant sections
3. **Bookmarking**: Bookmark frequently accessed sections for quick navigation
4. **Presentations**: Include direct links in presentations or reports

## Technical Implementation

The URL parameter is read using Streamlit's `st.query_params` API and mapped to the corresponding page name through the `PAGE_URL_MAPPING` dictionary in `app.py`. The sidebar radio button is then initialized to the correct page based on this mapping.
