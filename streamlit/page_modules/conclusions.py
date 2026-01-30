import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features

# Prediction quality thresholds (in days) - same as in interpretability.py
EXCELLENT_ERROR_THRESHOLD = 7.0
GOOD_ERROR_THRESHOLD = 21.0


def app():
    st.title("Conclusions")
    st.markdown("## Key findings and next steps")

    st.markdown(
        """
        **Summary of insights**
        - The LucKi cohort subset comprises 930 samples with ~6,900 features; data are sparse with ~300 taxa per sample.
        - An age-related signal exists but explains a limited share of variance; PCA shows a gradient rather than distinct clusters.
        - Model performance improves with CLR normalization and genus-level feature filtering.
        - Tree-based ensemble methods (Random Forest, XGBoost, Gradient Boosting, LightGBM) provide competitive baselines.
        - Feature importance and SHAP analysis consistently highlight several key bacterial genera as major contributors to age prediction.
        
        **Significant bacteria affecting age prediction:**
        
        Examples of genera that consistently show high feature importance in the models include:
        - *Staphylococcus*, *Citrobacter*, *Bifidobacterium*, *Corynebacterium*, and *Lacrimispora* 
            
        These genera represent the most influential features in predicting age groups and reflect 
        known biological patterns of microbiome development and aging.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **FAIR and accessibility notes**
        - Findable & Accessible: Data paths and preprocessing steps are documented; navigation uses clear labels.
        - Interoperable: Standard CSV inputs, encoded labels, and reproducible splits. Known metagenomic format.
        - Reusable: Cached preprocessing, documented transformations, and model summaries enable re-analysis. Code is open source, allowing for easy validation.
        """
    )
    st.markdown("---")

    st.subheader("Possible Limitations and Improvements")
    st.markdown(
        """
        - **Data scope:** Current models are trained on a single cohort; performance may vary on new populations, based on geography, culture, religion...
        - **Temporal drift:** Microbiome profiles can change over time; periodic re-training could be useful.
        - **Confounding factors:** Diet, antibiotics, and environment are not fully captured and may bias predictions.
        - **Future work:** Performing external validation, longitudinal modeling, and uncertainty estimates for predictions would improve its applicability in clinical practice.
        """
    )
    
    st.markdown("---")
    st.subheader("Prediction Quality Analysis")
    st.markdown("""
    To assess the overall performance of our best model, we analyze the distribution of prediction quality 
    on the test set. Predictions are categorized as:
    - **Excellent**: Error < 7 days
    - **Good**: Error between 7 and 21 days
    - **Bad**: Error ≥ 21 days
    """)
    
    # Add a button to generate the pie chart
    if st.button("Generate Prediction Quality Pie Chart", key='pred_quality_pie'):
        with st.spinner("Training model and analyzing predictions..."):
            # Load and preprocess data
            X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
            X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
            
            # Use genus-level features (best performing configuration)
            X_train_genus = filter_genus_features(X_train_clr)
            X_test_genus = filter_genus_features(X_test_clr)
            
            # Train the best model (Random Forest with tuned parameters)
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
            model.fit(X_train_genus, y_train)
            
            # Generate predictions
            predictions = model.predict(X_test_genus)
            
            # Calculate errors
            errors = np.abs(y_test.values - predictions)
            
            # Categorize predictions
            excellent = np.sum(errors < EXCELLENT_ERROR_THRESHOLD)
            good = np.sum((errors >= EXCELLENT_ERROR_THRESHOLD) & (errors < GOOD_ERROR_THRESHOLD))
            bad = np.sum(errors >= GOOD_ERROR_THRESHOLD)
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 7))
            
            labels = ['Excellent\n(< 7 days)', 'Good\n(7-21 days)', 'Bad\n(≥ 21 days)']
            sizes = [excellent, good, bad]
            colors = ['#2E7D32', '#FFA726', '#D32F2F']  # Green, Orange, Red
            explode = (0.05, 0.05, 0.05)  # Slightly separate all slices
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            
            # Enhance the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_weight('bold')
            
            ax.set_title('Distribution of Prediction Quality on Test Set', 
                        fontsize=16, weight='bold', pad=20)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Display the pie chart
            st.pyplot(fig)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Test Samples", len(y_test))
            with col2:
                st.metric("Excellent Predictions", f"{excellent} ({excellent/len(y_test)*100:.1f}%)")
            with col3:
                st.metric("Good Predictions", f"{good} ({good/len(y_test)*100:.1f}%)")
            with col4:
                st.metric("Bad Predictions", f"{bad} ({bad/len(y_test)*100:.1f}%)")
            
            # Additional insights
            st.markdown("---")
            st.markdown("### Insights")
            
            total_acceptable = excellent + good
            acceptable_pct = total_acceptable / len(y_test) * 100
            
            st.markdown(f"""
            - **{acceptable_pct:.1f}%** of predictions are within acceptable range (error < 21 days)
            - **{excellent/len(y_test)*100:.1f}%** of predictions are highly accurate (error < 7 days)
            - The model demonstrates {'strong' if excellent/len(y_test) > 0.3 else 'moderate' if excellent/len(y_test) > 0.15 else 'reasonable'} performance on the test set
            """)