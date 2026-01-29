import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features
from utils.functions import train_and_evaluate_model
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

def app():
    st.title("Model Interpretability")
    
    st.markdown("""
    This section provides interpretability tools to understand how models make predictions.
    Explore feature importance, SHAP values, and individual prediction explanations interactively.
    """)
    
    tabs = st.tabs(["Feature Importance", "Sample Explorer", "SHAP Analysis"])
    
    with st.spinner("Loading and preprocessing data..."):
        X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
        X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
        
        X_train_genus = filter_genus_features(X_train_clr)
        X_test_genus = filter_genus_features(X_test_clr)
    
    with tabs[0]:
        st.header("🔍 Feature Importance Analysis")
        
        st.markdown("""
        Feature importance shows which features contribute most to the model predictions.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
            n_features = st.slider("Number of top features to display", 10, 50, 20, 5)
        
        with col2:
            viz_type = st.radio("Visualization Type", ["Bar Chart", "Interactive Bar", "Heatmap"])
        
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
                
                # Display top features
                top_features_df = feature_importance.head(n_features)
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.subheader(f"Top {n_features} Features")
                    st.dataframe(top_features_df, use_container_width=True, height=400)
                    
                    st.metric("Total Features", len(feature_importance))
                    st.metric("Top Feature", feature_importance.iloc[0]['Feature'])
                    st.metric("Top Feature Importance", f"{feature_importance.iloc[0]['Importance']:.4f}")
                
                with col1:
                    if viz_type == "Bar Chart":
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(range(len(top_features_df)), top_features_df['Importance'].values, color='steelblue')
                        ax.set_yticks(range(len(top_features_df)))
                        ax.set_yticklabels(top_features_df['Feature'].values, fontsize=8)
                        ax.set_xlabel('Importance')
                        ax.set_title(f'Top {n_features} Feature Importances - {model_choice}')
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                    elif viz_type == "Interactive Bar":
                        fig = px.bar(
                            top_features_df,
                            y='Feature',
                            x='Importance',
                            orientation='h',
                            title=f'Top {n_features} Feature Importances - {model_choice}',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Heatmap
                        importance_matrix = top_features_df['Importance'].values.reshape(-1, 1)
                        fig, ax = plt.subplots(figsize=(3, 10))
                        sns.heatmap(importance_matrix, 
                                    yticklabels=top_features_df['Feature'].values,
                                    xticklabels=['Importance'],
                                    cmap='YlOrRd', annot=False, cbar=True, ax=ax)
                        ax.set_title(f'Top {n_features} Features Heatmap')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Cumulative importance
                st.subheader("📈 Cumulative Importance Analysis")
                cumsum_importance = np.cumsum(feature_importance['Importance'].values)
                n_features_90 = np.argmax(cumsum_importance >= 0.9) + 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(cumsum_importance) + 1)),
                    y=cumsum_importance,
                    mode='lines',
                    name='Cumulative Importance',
                    line=dict(color='blue', width=2)
                ))
                fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                             annotation_text="90% threshold")
                fig.update_layout(
                    title='Cumulative Feature Importance',
                    xaxis_title='Number of Features',
                    yaxis_title='Cumulative Importance',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 Number of features needed to capture 90% of total importance: **{n_features_90}** out of {len(feature_importance)}")
    
    with tabs[1]:
        st.header("🔬 Sample Explorer")
        
        st.markdown("""
        Explore individual sample predictions and understand which features contribute to each prediction.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"], key='sample_model')
        
        with col2:
            sample_source = st.radio("Sample Source", ["Test Set", "Training Set"])
        
        if st.button("Train Model for Sample Analysis", key='sample_train'):
            with st.spinner(f"Training {model_choice}..."):
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
                
                model.fit(X_train_genus, y_train)
                
                # Choose dataset
                if sample_source == "Test Set":
                    X_samples = X_test_genus
                    y_samples = y_test
                else:
                    X_samples = X_train_genus
                    y_samples = y_train
                
                sample_idx = st.slider("Select Sample Index", 0, len(X_samples)-1, 0, key='sample_idx')
                sample = X_samples.iloc[sample_idx:sample_idx+1]
                true_value = y_samples.iloc[sample_idx]
                pred_value = model.predict(sample)[0]
                
                # Display sample information
                st.subheader("📊 Sample Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sample Index", f"{sample_idx}")
                with col2:
                    st.metric("True Age Group", f"{true_value}")
                with col3:
                    st.metric("Predicted Age Group", f"{pred_value:.2f}")
                
                error = abs(true_value - pred_value)
                if error < 0.5:
                    st.success(f"✅ Excellent prediction! Error: {error:.3f}")
                elif error < 1.0:
                    st.info(f"ℹ️ Good prediction! Error: {error:.3f}")
                else:
                    st.warning(f"⚠️ Significant error: {error:.3f}")
                
                # Top feature contributions
                st.subheader("🎯 Feature Contributions for This Sample")
                
                top_n = st.slider("Number of features to show", 5, 20, 10, key='sample_features_slider')
                
                top_feature_indices = model.feature_importances_.argsort()[-top_n:][::-1]
                feature_values = sample.iloc[0, top_feature_indices].values
                feature_names = X_train_genus.columns[top_feature_indices]
                feature_importances_local = model.feature_importances_[top_feature_indices]
                
                # Note: This is a rough approximation. For accurate local explanations, use SHAP or LIME.
                # This calculation combines feature values with global importance as a proxy for contribution.
                contributions = feature_values * feature_importances_local
                
                # Interactive plotly chart
                fig = go.Figure()
                colors = ['red' if c < 0 else 'green' for c in contributions]
                fig.add_trace(go.Bar(
                    y=feature_names,
                    x=contributions,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{c:.3f}" for c in contributions],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=f'Top {top_n} Feature Contributions for Sample {sample_idx}',
                    xaxis_title='Contribution',
                    yaxis_title='Feature',
                    height=max(400, top_n * 30),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                🟢 **Green bars** indicate features that push the prediction higher.
                🔴 **Red bars** indicate features that push the prediction lower.
                
                ⚠️ **Note**: This is a simplified approximation combining feature values with global importance. 
                For accurate local explanations, consider using SHAP or LIME libraries.
                """)
                
                # Feature values table
                with st.expander("📋 View Feature Values"):
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value (CLR)': feature_values,
                        'Importance': feature_importances_local,
                        'Contribution': contributions
                    })
                    st.dataframe(feature_df, use_container_width=True)
    
    with tabs[2]:
        st.header("🎲 SHAP-like Analysis")
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) provides a unified measure of feature importance 
        based on game theory. This demonstrates the concept with feature importance approximations.
        """)
        
        st.info("""
        💡 **Note**: Full SHAP computation can be computationally expensive for large datasets.
        This demonstration uses feature importance and contribution analysis as a proxy.
        """)
        
        if st.button("Generate SHAP-like Analysis", key='shap_gen'):
            with st.spinner("Training model for SHAP analysis..."):
                model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
                model.fit(X_train_genus, y_train)
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train_genus.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("🌍 Global Feature Importance (Proxy for SHAP)")
                
                top_n_global = st.slider("Number of top features", 10, 30, 20, key='shap_top')
                
                # Interactive plotly chart
                top_features = feature_importance.head(top_n_global)
                fig = px.bar(
                    top_features,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    title=f'Top {top_n_global} Features by Global Importance',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=max(400, top_n_global * 25), yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Local explanation
                st.subheader("🔍 Local Explanation")
                sample_idx = st.slider("Select Sample for Local Explanation", 0, len(X_test_genus)-1, 0, key='shap_sample')
                sample = X_test_genus.iloc[sample_idx:sample_idx+1]
                true_value = y_test.iloc[sample_idx]
                pred_value = model.predict(sample)[0]
                
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
                
                # Note: This is a rough approximation. For accurate local explanations, use SHAP or LIME.
                # This calculation combines feature values with global importance as a proxy for contribution.
                contributions = feature_values * feature_importances_local
                
                fig = go.Figure()
                colors = ['rgba(255,0,0,0.7)' if c < 0 else 'rgba(0,128,0,0.7)' for c in contributions]
                fig.add_trace(go.Bar(
                    y=feature_names,
                    x=contributions,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{c:.3f}" for c in contributions],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=f'Feature Contributions for Sample {sample_idx}',
                    xaxis_title='Contribution (SHAP-like)',
                    yaxis_title='Feature',
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **How to interpret:**
                - 🟢 Green bars push the prediction toward higher age groups
                - 🔴 Red bars push the prediction toward lower age groups
                - Longer bars have stronger impact on the prediction
                
                ⚠️ **Note**: This is a simplified approximation combining feature values with global importance. 
                For accurate local explanations and true SHAP values, consider using the SHAP library.
                """)
