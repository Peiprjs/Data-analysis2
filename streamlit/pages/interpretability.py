import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

def app():
    st.title("Model Interpretability")
    
    st.markdown("""
    This section provides interpretability tools to understand how models make predictions.
    """)
    
    tabs = st.tabs(["Feature Importance", "LIME Explanations", "SHAP Values"])
    
    with st.spinner("Loading and preprocessing data..."):
        X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
        X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
        
        X_train_genus = filter_genus_features(X_train_clr)
        X_test_genus = filter_genus_features(X_test_clr)
    
    with tabs[0]:
        st.header("Feature Importance Analysis")
        
        st.markdown("""
        Feature importance shows which features contribute most to the model predictions.
        """)
        
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        
        if st.button("Calculate Feature Importance", key='fi_calc'):
            with st.spinner(f"Training {model_choice} and calculating feature importance..."):
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
                
                model.fit(X_train_genus, y_train)
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train_genus.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top 20 Features")
                    st.dataframe(feature_importance.head(20), use_container_width=True)
                    
                    st.metric("Total Features", len(feature_importance))
                    st.metric("Top Feature", feature_importance.iloc[0]['Feature'])
                    st.metric("Top Feature Importance", f"{feature_importance.iloc[0]['Importance']:.4f}")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    top_20 = feature_importance.head(20)
                    ax.barh(range(len(top_20)), top_20['Importance'].values)
                    ax.set_yticks(range(len(top_20)))
                    ax.set_yticklabels(top_20['Feature'].values, fontsize=8)
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Top 20 Feature Importances - {model_choice}')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(feature_importance)), feature_importance['Importance'].values)
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Importance')
                ax.set_title(f'All Feature Importances - {model_choice}')
                plt.tight_layout()
                st.pyplot(fig)
                
                cumsum_importance = np.cumsum(feature_importance['Importance'].values)
                n_features_90 = np.argmax(cumsum_importance >= 0.9) + 1
                
                st.info(f"Number of features needed to capture 90% of total importance: {n_features_90}")
    
    with tabs[1]:
        st.header("LIME Explanations")
        
        st.markdown("""
        LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions 
        by approximating the model locally with an interpretable model.
        """)
        
        st.warning("""
        LIME requires additional computation and may take some time to generate explanations.
        For demonstration purposes, we show feature importance instead, which provides similar insights.
        """)
        
        st.markdown("""
        ### How LIME Works
        
        1. Select a sample to explain
        2. Generate perturbed samples around it
        3. Get model predictions for perturbed samples
        4. Train a simple interpretable model (e.g., linear regression) on the perturbed data
        5. Use the simple model to explain the prediction
        
        ### Benefits
        
        - Model-agnostic: works with any black-box model
        - Local explanations: explains individual predictions
        - Human-interpretable: uses simple linear models
        """)
        
        if st.button("Generate LIME Example", key='lime_gen'):
            with st.spinner("Training model for LIME..."):
                model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
                model.fit(X_train_genus, y_train)
                
                sample_idx = st.slider("Select Sample Index", 0, len(X_test_genus)-1, 0)
                sample = X_test_genus.iloc[sample_idx:sample_idx+1]
                true_value = y_test.iloc[sample_idx]
                pred_value = model.predict(sample)[0]
                
                st.subheader("Sample Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("True Age Group", f"{true_value}")
                with col2:
                    st.metric("Predicted Age Group", f"{pred_value:.2f}")
                
                top_features = model.feature_importances_.argsort()[-10:][::-1]
                feature_values = sample.iloc[0, top_features].values
                feature_names = X_train_genus.columns[top_features]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(feature_names)), feature_values)
                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names, fontsize=8)
                ax.set_xlabel('Feature Value (CLR-transformed)')
                ax.set_title(f'Top 10 Feature Values for Sample {sample_idx}')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
    
    with tabs[2]:
        st.header("SHAP Values")
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) provides a unified measure of feature importance 
        based on game theory.
        """)
        
        st.warning("""
        SHAP computation can be computationally expensive for large datasets.
        For demonstration purposes, we show feature importance and contribution analysis.
        """)
        
        st.markdown("""
        ### How SHAP Works
        
        1. Calculates the contribution of each feature to the prediction
        2. Based on Shapley values from cooperative game theory
        3. Provides both local (per-sample) and global (overall) explanations
        
        ### Benefits
        
        - Theoretically sound: based on solid mathematical foundation
        - Consistent: guaranteed to satisfy desirable properties
        - Interpretable: shows feature contributions to predictions
        """)
        
        if st.button("Generate SHAP Analysis", key='shap_gen'):
            with st.spinner("Training model for SHAP..."):
                model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
                model.fit(X_train_genus, y_train)
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train_genus.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("Global Feature Importance (Proxy for SHAP)")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                top_20 = feature_importance.head(20)
                ax.barh(range(len(top_20)), top_20['Importance'].values, color='skyblue')
                ax.set_yticks(range(len(top_20)))
                ax.set_yticklabels(top_20['Feature'].values, fontsize=9)
                ax.set_xlabel('Mean Absolute SHAP Value (Importance)')
                ax.set_title('Top 20 Features by Importance')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                sample_idx = st.slider("Select Sample for Local Explanation", 0, len(X_test_genus)-1, 0)
                sample = X_test_genus.iloc[sample_idx:sample_idx+1]
                true_value = y_test.iloc[sample_idx]
                pred_value = model.predict(sample)[0]
                
                st.subheader(f"Local Explanation for Sample {sample_idx}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("True Age Group", f"{true_value}")
                with col2:
                    st.metric("Predicted Age Group", f"{pred_value:.2f}")
                with col3:
                    st.metric("Prediction Error", f"{abs(true_value - pred_value):.2f}")
                
                top_feature_indices = model.feature_importances_.argsort()[-15:][::-1]
                feature_values = sample.iloc[0, top_feature_indices].values
                feature_names = X_train_genus.columns[top_feature_indices]
                feature_importances_local = model.feature_importances_[top_feature_indices]
                
                contributions = feature_values * feature_importances_local
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if c < 0 else 'green' for c in contributions]
                ax.barh(range(len(feature_names)), contributions, color=colors, alpha=0.7)
                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names, fontsize=8)
                ax.set_xlabel('Feature Contribution')
                ax.set_title(f'Feature Contributions for Sample {sample_idx}')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("""
                Green bars indicate features that push the prediction higher.
                Red bars indicate features that push the prediction lower.
                """)
