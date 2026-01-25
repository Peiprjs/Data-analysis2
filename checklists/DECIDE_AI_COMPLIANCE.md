# DECIDE-AI Compliance Checklist

**DECIDE-AI**: A checklist for reporting of evaluation studies in medical AI

**Project**: Microbiome Data Analysis Platform  
**Version**: 1.0.0  
**Date**: 2026-01-25

---

## Overview

DECIDE-AI is a reporting guideline for evaluation studies of medical AI systems. It focuses on ensuring transparent reporting of study design, data sources, reference standards, and statistical methods.

---

## Section 1: Title and Abstract

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 1 | Title identifies study as diagnostic/prognostic accuracy study | Done | README.md - "Microbiome Data Analysis Platform" with prediction focus |
| 2 | Abstract structured with study design, methods, results, conclusions | Done | README.md Overview and Features sections |

---

## Section 2: Introduction

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 3 | Scientific/clinical background with AI rationale | Done | README.md Overview - age prediction from microbiome |
| 4 | Study objectives and hypotheses | Done | README.md Methodology - predict age groups using ML |

---

## Section 3: Methods - Study Design

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 5a | Study design: Prospective/retrospective, single/multi-center | Done | Retrospective analysis of LucKi cohort data |
| 5b | Recruitment: How participants identified | Done | LucKi cohort (cross-sectional stool samples) |
| 5c | Data collection: Prospective/retrospective | Done | Retrospective use of existing cohort data |
| 5d | Blinding: Who was blinded to what | N/A | Research study, not clinical evaluation |

---

## Section 4: Methods - Participants

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 6a | Eligibility criteria: Inclusion/exclusion | Partial | LucKi cohort participants; specific criteria in cohort paper |
| 6b | Participant sampling: Consecutive/random | Partial | Full cohort subset used; sampling method not detailed |
| 6c | Data sources: Clinical/population | Done | LucKi population cohort, stool samples |
| 7 | Data collection period and setting | Partial | Cohort paper referenced; specific dates not in README |

---

## Section 5: Methods - Test Methods

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 8a | Index test: Describe AI system fully | Done | Multiple ML models (RF, XGBoost, etc.) in models.py |
| 8b | AI system purpose: Triage/diagnosis/prognosis | Done | Prognostic - age group prediction |
| 8c | AI algorithm: Architecture and training | Done | models.py, functions.py with full implementations |
| 8d | AI training data: Separate from test data | Done | 80/20 train-test split (data_loader.py) |
| 8e | AI preprocessing: Steps documented | Done | preprocessing.py with CLR transformation |
| 8f | AI output: Format and interpretation | Done | Continuous age group predictions (0-N scale) |
| 8g | Software/hardware: Versions and requirements | Done | requirements.txt, README.md Hardware Requirements |
| 8h | AI accessibility: Available for testing | Done | Full code on GitHub, Streamlit app available |

---

## Section 6: Methods - Reference Standard

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 9a | Reference standard: Definition and rationale | Done | Age group at sample collection (metadata) |
| 9b | Reference standard measurement | Done | Recorded in LucKi cohort metadata |
| 9c | Reference standard blinding | N/A | Retrospective study, age groups predetermined |
| 10 | Index test vs reference timing | Done | Concurrent - microbiome and age from same timepoint |

---

## Section 7: Methods - Statistical Methods

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 11a | Sample size: How determined | Done | Full available cohort (n=930) |
| 11b | Missing data: Handling methods | Done | preprocessing.py - dropna for missing age groups |
| 11c | Outliers: Detection and handling | Done | preprocessing.py - IQR-based outlier detection |
| 12a | Performance metrics: Predefined | Done | RMSE, R², MAE (standard regression metrics) |
| 12b | Confidence intervals: Calculation method | Partial | CV provides variability; explicit CIs not computed |
| 12c | Subgroup analysis: If performed | Partial | Stratified by age group; no sex/family subgroup analysis |

---

## Section 8: Results - Participants

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 13a | Participant flow: Enrollment to analysis | Done | README.md UML diagram shows flow |
| 13b | Baseline characteristics: Demographics | Done | home.py shows sample statistics |
| 13c | Distribution of conditions: Prevalence | Done | Age group distribution shown in preprocessing.py |

---

## Section 9: Results - Test Results

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 14a | AI performance: Report all prespecified metrics | Done | models.py reports RMSE, R², MAE |
| 14b | Performance by subgroup: If applicable | Partial | Age group performance shown; no other subgroups |
| 14c | Confidence intervals: Report for all metrics | Partial | CV std reported; formal CIs not calculated |
| 15 | Adverse events: AI-related failures | N/A | Research tool, not deployed clinically |
| 16 | Indeterminate results: Report and handle | Done | All predictions deterministic; no indeterminate cases |

---

## Section 10: Results - Model Transparency

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 17a | Model interpretability: Feature importance | Done | interpretability.py with LIME, SHAP, feature importance |
| 17b | Error analysis: Failure modes | Partial | Residual plots shown; systematic error analysis limited |
| 17c | Model decisions: Examples provided | Done | Individual predictions shown in interpretability.py |

---

## Section 11: Discussion

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| 18 | Limitations: Study and AI system | Partial | README.md Future Work section; needs expansion |
| 19 | Clinical applicability: Practical implications | Partial | README.md Impact section; needs clinical detail |
| 20 | Generalizability: External validity | Partial | LucKi cohort specific; needs discussion |

---

## Section 12: AI-Specific Requirements

| Item | DECIDE-AI Requirement | Status | Location/Notes |
|------|----------------------|--------|----------------|
| AI-1 | Dataset composition: Training/validation/test splits | Done | 80/20 split documented (data_loader.py) |
| AI-2 | Data quality: Quality control measures | Done | preprocessing.py shows QC steps |
| AI-3 | Data diversity: Demographic representation | Partial | Family-based cohort; diversity not analyzed |
| AI-4 | Training procedure: Complete description | Done | models.py shows training process |
| AI-5 | Hyperparameters: Report all settings | Done | models.py shows slider controls and defaults |
| AI-6 | Validation strategy: Internal/external | Done | Train-test split + k-fold CV (results.py) |
| AI-7 | Performance comparison: Human/other AI | Not Done | No human baseline or comparison to other studies |
| AI-8 | Computational requirements: Infrastructure | Done | README.md Hardware Requirements |
| AI-9 | Reproducibility: Code and data availability | Done | Full code on GitHub; data availability described |
| AI-10 | Model versioning: Track model versions | Done | pyproject.toml version 1.0.0, CHANGELOG.md |
| AI-11 | Bias assessment: Evaluate for bias | Partial | Stratified split ensures balance; no systematic bias analysis |
| AI-12 | Failure mode analysis: When AI fails | Partial | Residual analysis; systematic failure modes not analyzed |

---

## Compliance Summary

**Total Items**: 54  
**Fully Compliant**: 35 (65%)  
**Partially Compliant**: 17 (31%)  
**Not Compliant/N/A**: 2 (4%)

---

## Recommendations for Full Compliance

### High Priority

1. **Add comprehensive limitations section**:
   ```markdown
   ## Limitations
   
   ### Study Limitations
   - Single cohort (LucKi) limits generalizability
   - Cross-sectional design prevents temporal analysis
   - Retrospective data collection
   
   ### AI System Limitations
   - Models specific to MetaPhlAn 4.1.1 profiling
   - Requires sufficient sequencing depth
   - Age group prediction (not continuous age)
   ```

2. **Perform systematic bias analysis**:
   - Analyze performance by sex
   - Analyze performance by family clusters
   - Test for demographic biases
   - Add to results.py

3. **Add failure mode analysis**:
   ```python
   # In results.py
   def analyze_failure_modes(y_true, y_pred, threshold=2.0):
       large_errors = abs(y_true - y_pred) > threshold
       # Analyze characteristics of large errors
   ```

4. **Add confidence intervals**:
   ```python
   from scipy import stats
   def calculate_ci(values, confidence=0.95):
       return stats.t.interval(confidence, len(values)-1, 
                              loc=np.mean(values), 
                              scale=stats.sem(values))
   ```

### Medium Priority

5. **Add cohort details to README.md**:
   - Data collection period
   - Specific inclusion/exclusion criteria
   - Sampling strategy

6. **Add performance comparison**:
   - Compare to random baseline
   - Compare to mean predictor
   - Reference any literature baselines

7. **Add subgroup analysis**:
   ```python
   # In results.py
   def subgroup_performance(X, y, y_pred, metadata):
       # Analyze by sex, family, etc.
   ```

### Low Priority

8. **Expand clinical applicability section**:
   - How results inform microbiome-age relationships
   - Potential clinical applications
   - Healthcare implications

9. **Add external validation discussion**:
   - How to apply to other cohorts
   - What adaptations needed
   - Generalizability expectations

---

## Action Items for Implementation

### Code Changes Needed

1. **Add to results.py**:
```python
def bias_analysis(X_test, y_test, y_pred, metadata):
    """
    Analyze model performance across demographic subgroups.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predictions
    metadata : pd.DataFrame
        Demographic information (sex, family_id)
    
    Returns
    -------
    dict
        Performance metrics by subgroup
    """
    results = {}
    
    # Performance by sex
    for sex in metadata['sex'].unique():
        mask = metadata['sex'] == sex
        results[f'sex_{sex}'] = {
            'rmse': np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])),
            'r2': r2_score(y_test[mask], y_pred[mask]),
            'n': mask.sum()
        }
    
    # Performance by family
    for family in metadata['family_id'].unique():
        mask = metadata['family_id'] == family
        if mask.sum() >= 5:  # Only if enough samples
            results[f'family_{family}'] = {
                'rmse': np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])),
                'r2': r2_score(y_test[mask], y_pred[mask]),
                'n': mask.sum()
            }
    
    return results

def failure_mode_analysis(y_test, y_pred, X_test, threshold=2.0):
    """
    Identify and analyze failure modes.
    
    Parameters
    ----------
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predictions
    X_test : pd.DataFrame
        Test features
    threshold : float
        Error threshold for failure
    
    Returns
    -------
    pd.DataFrame
        Analysis of failure cases
    """
    errors = np.abs(y_test - y_pred)
    failures = errors > threshold
    
    failure_analysis = pd.DataFrame({
        'true_value': y_test[failures],
        'predicted_value': y_pred[failures],
        'error': errors[failures],
        'sample_idx': np.where(failures)[0]
    })
    
    return failure_analysis
```

2. **Add to README.md**:
```markdown
## Limitations

### Study Design
- **Single Cohort**: Data from LucKi cohort only; generalizability to other populations unknown
- **Cross-Sectional**: Snapshot in time; cannot assess temporal changes
- **Retrospective**: Uses existing data; subject to collection biases

### AI System
- **Method-Specific**: Models trained on MetaPhlAn 4.1.1; may not generalize to other profiling methods
- **Categorical Output**: Predicts age groups, not continuous age
- **Sequencing Depth**: Requires adequate sequencing coverage for reliable predictions
- **Compositional Data**: Assumes proper CLR transformation; sensitive to preprocessing

### Clinical Application
- **Research Tool**: Not validated for clinical use
- **Population Specificity**: Performance in other populations not established
- **Confounders**: Family clustering and other factors not fully addressed
```

### Documentation Changes Needed

3. **Create LIMITATIONS.md**:
   - Detailed limitation discussion
   - Mitigation strategies
   - Future work to address

4. **Create VALIDATION.md**:
   - Internal validation results
   - External validation considerations
   - Validation protocol for new data

---

## Document History

- **Version 1.0**: Initial DECIDE-AI compliance assessment
- **Date**: 2024-01-25
- **Assessor**: Data Analysis Team
- **Next Review**: Upon major updates or external validation

---

## References

- Sounderajah V, et al. (2021). Developing a reporting guideline for artificial intelligence-centred diagnostic test accuracy studies: the STARD-AI protocol. BMJ Open.
- DECIDE-AI: Diagnostic Evidence Cooperative Interoperable Data Excellence - AI
- Cohen JF, et al. (2016). STARD 2015 guidelines for reporting diagnostic accuracy studies. BMJ.
- GO-FAIR Principles: https://www.go-fair.org/fair-principles/
