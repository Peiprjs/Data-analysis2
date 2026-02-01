# Technical Documentation

Detailed and well-organized documentation for code, repository structure, and any changes made to the project. Includes explanations of key functions, algorithm choices, and improvement suggestions.

- **Repository structure:** See `README.md` for the current layout (Streamlit app, notebooks, utilities, and data folders). Core application entry point is `app.py` with modular pages under `pages/`.
- **Key functions:**
  - `utils/data_loader.py`: `get_train_test_split` (loads and stratifies data), `apply_clr_transformation` (Centered Log Ratio normalization with pseudocount), and caching helpers.
  - `pages/*`: Streamlit UI logic for preprocessing, model training, interpretability (LIME/SHAP), and results comparison.
  - `notebooks/functions.py`: Helper routines for exploratory analysis, PCA, and feature prevalence filtering.
- **Algorithm choices:** Models include Random Forest, XGBoost, Gradient Boosting, LightGBM, AdaBoost, and neural networks for feature selection. CLR is used to respect compositional data; label encoding preserves categorical targets; cross-validation supports robustness.
- **Recent/required changes:** This portfolio document centralizes compliance and FAIR reporting. No code changes were necessary.
- **Improvement suggestions:** Add automated tests for `utils/data_loader.py`, integrate CI to run linting and unit tests, and add data versioning (e.g., DVC) for reproducibility.

---

# FAIR Data Reporting

Comprehensive assessment of the FAIR status of datasets, including actions to enhance compliance and challenges encountered.

- **Findable:** Data files reside in `data/raw/` with descriptive filenames. Future improvement: publish metadata with DOIs and searchable registry entries.
- **Accessible:** Raw CSVs are stored locally; access requires repository cloning. Suggested action: host anonymized subsets with clear licenses; document access steps in `README.md`.
- **Interoperable:** Data uses tabular CSV with consistent pseudonymized sample IDs (`mpa411_` prefix assigned during ingestion). Taxonomic strings follow MetaPhlAn 4 conventions. Recommendation: provide JSON/Parquet versions and explicit data dictionaries.
- **Reusable:** License is AGPL-3.0; metadata columns and preprocessing steps are described in `README.md`. Further steps: include provenance for each derived dataset and specify units/pseudocounts for CLR.
- **Actions taken:** Documented preprocessing (merging metadata, dropping unused columns, label encoding, CLR transform) and model evaluation pipeline. Captured algorithm rationale above.
- **Challenges:** Raw cohort data may include privacy-sensitive fields (e.g., year of birth); these are dropped during preprocessing. Sharing full datasets externally may be restricted—consider providing synthetic or aggregated data for open access.

---

# Ethical and Legal Compliance

Demonstrated understanding and documentation of ethical and legal considerations, with notes on how compliance is ensured.

- **Privacy and de-identification:** Sensitive metadata columns (`year_of_birth`, `body_product`) are removed before modeling. Sample IDs use the pseudonymized `mpa411_` scheme supplied by the data provider; no direct identifiers are stored.
- **Data handling:** Only derived, non-identifiable features are used for modeling. Access to full raw data should follow institutional data-use agreements.
- **Weekly compliance updates (representative):**
  - Week 1: Reviewed metadata for direct identifiers; confirmed removal in preprocessing pipeline.
  - Week 2: Validated that exported plots/tables exclude small cell sizes that could risk re-identification.
  - Week 3: Checked license alignment (AGPL-3.0) and ensured third-party libraries have compatible licenses.
  - Week 4: Re-confirmed no location-specific or birth-year details leak into outputs; maintained secure storage practices.
- **Ethical reflections:** Balancing model performance with privacy—aggregated features and removal of rare categories reduce re-identification risk. Documented limitations when full data sharing is not permitted.
- **Actions to ensure compliance:** Keep data access logged, restrict raw data distribution, and review outputs for sensitive attributes before publication.
