import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features, extract_genus_name
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
        st.header("Feature Importance Analysis")
        
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
                    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=3004, n_jobs=-1)
                
                model.fit(X_train_genus, y_train)
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train_genus.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Extract genus names for display
                feature_importance['Genus'] = feature_importance['Feature'].apply(extract_genus_name)
                
                # Display top features
                top_features_df = feature_importance.head(n_features)
                
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.subheader(f"Top {n_features} Features")
                    display_df = top_features_df[['Genus', 'Importance']].copy()
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    st.metric("Total Features", len(feature_importance))
                    st.metric("Top Feature", feature_importance.iloc[0]['Genus'])
                    st.metric("Top Feature Importance", f"{feature_importance.iloc[0]['Importance']:.4f}")
                
                with col1:
                    if viz_type == "Bar Chart":
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(range(len(top_features_df)), top_features_df['Importance'].values, color='steelblue')
                        ax.set_yticks(range(len(top_features_df)))
                        ax.set_yticklabels(top_features_df['Genus'].values, fontsize=8)
                        ax.set_xlabel('Importance')
                        ax.set_title(f'Top {n_features} Feature Importances - {model_choice}')
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                    elif viz_type == "Interactive Bar":
                        fig = px.bar(
                            top_features_df,
                            y='Genus',
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
                                    yticklabels=top_features_df['Genus'].values,
                                    xticklabels=['Importance'],
                                    cmap='YlOrRd', annot=False, cbar=True, ax=ax)
                        ax.set_title(f'Top {n_features} Features Heatmap')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Cumulative importance
                st.subheader("Cumulative Importance Analysis")
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
                
                st.info(f"Number of features needed to capture 90% of total importance: **{n_features_90}** out of {len(feature_importance)}")
    
    with tabs[1]:
        st.header("Sample Explorer")
        
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
                    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=3004, n_jobs=-1)
                
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
                true_value = float(y_samples.iloc[sample_idx])
                pred_value = float(model.predict(sample)[0])
                
                # Display sample information
                st.subheader("Sample Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sample Index", f"{sample_idx}")
                with col2:
                    st.metric("True Age (days)", f"{true_value:.1f}")
                with col3:
                    st.metric("Predicted Age (days)", f"{pred_value:.1f}")
                
                error = abs(true_value - pred_value)
                if error < 3.5:
                    st.success(f"Excellent prediction! Error: {error:.1f} days")
                elif error < 7.0:
                    st.info(f"Good prediction! Error: {error:.1f} days")
                else:
                    st.warning(f"Significant error: {error:.1f} days")
                
                # Top feature contributions
                st.subheader("Feature Contributions for This Sample")
                
                top_n = st.slider("Number of features to show", 5, 20, 10, key='sample_features_slider')
                
                top_feature_indices = model.feature_importances_.argsort()[-top_n:][::-1]
                feature_values = sample.iloc[0, top_feature_indices].values
                feature_names = X_train_genus.columns[top_feature_indices]
                feature_genus_names = [extract_genus_name(name) for name in feature_names]
                feature_importances_local = model.feature_importances_[top_feature_indices]
                
                # Note: This is a rough approximation. For accurate local explanations, use SHAP or LIME.
                # This calculation combines feature values with global importance as a proxy for contribution.
                contributions = feature_values * feature_importances_local
                
                # Interactive plotly chart
                fig = go.Figure()
                colors = ['red' if c < 0 else 'green' for c in contributions]
                fig.add_trace(go.Bar(
                    y=feature_genus_names,
                    x=contributions,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{c:.3f}" for c in contributions],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=f'Top {top_n} Feature Contributions for Sample {sample_idx}',
                    xaxis_title='Contribution',
                    yaxis_title='Genus',
                    height=max(400, top_n * 30),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Green bars** indicate features that push the prediction higher.
                **Red bars** indicate features that push the prediction lower. 
               """)
                
                # Feature values table
                with st.expander("View Feature Values"):
                    feature_df = pd.DataFrame({
                        'Genus': feature_genus_names,
                        'Value (CLR)': feature_values,
                        'Importance': feature_importances_local,
                        'Contribution': contributions
                    })
                    st.dataframe(feature_df, use_container_width=True)
    
    with tabs[2]:
        st.header("SHAP Analysis")
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) provides a unified measure of feature importance 
        based on game theory. This tool uses the official SHAP library to provide accurate
        explanations of model predictions.
        """)

        col1, col2 = st.columns([1, 1])
        
        with col1:
            analysis_type = st.radio("Analysis Type", ["Global Feature Importance", "Local Sample Explanation"])
        
        with col2:
            if analysis_type == "Local Sample Explanation":
                sample_idx = st.slider("Select Sample for Local Explanation", 0, len(X_test_genus)-1, 0, key='shap_sample')

        if st.button("Generate SHAP Analysis", key='shap_gen'):
            with st.spinner("Training model and computing SHAP values..."):
                model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
                model.fit(X_train_genus, y_train)
                
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                
                if analysis_type == "Global Feature Importance":
                    st.subheader("Global Feature Importance")
                    
                    # Calculate SHAP values for test set
                    shap_values = explainer.shap_values(X_test_genus)
                    
                    # Calculate mean absolute SHAP values for global importance
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X_train_genus.columns,
                        'Mean |SHAP|': mean_shap
                    }).sort_values('Mean |SHAP|', ascending=False)
                    
                    # Extract genus names
                    feature_importance['Genus'] = feature_importance['Feature'].apply(extract_genus_name)
                    
                    top_n_global = st.slider("Number of top features", 10, 30, 20, key='shap_top')
                    top_features = feature_importance.head(top_n_global)
                    
                    # Interactive plotly chart
                    fig = px.bar(
                        top_features,
                        y='Genus',
                        x='Mean |SHAP|',
                        orientation='h',
                        title=f'Top {top_n_global} Features by Mean Absolute SHAP Value',
                        color='Mean |SHAP|',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=max(400, top_n_global * 25), yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display summary statistics
                    st.subheader("SHAP Summary Statistics")
                    display_df = top_features[['Genus', 'Mean |SHAP|']].copy()
                    st.dataframe(display_df, use_container_width=True)
                    
                    # SHAP summary plot
                    st.subheader("SHAP Summary Plot")
                    st.info("The summary plot shows the distribution of SHAP values for each feature across all samples.")
                    
                    # Create renamed columns for display
                    X_test_genus_display = X_test_genus.copy()
                    X_test_genus_display.columns = [extract_genus_name(col) for col in X_test_genus_display.columns]
                    
                    shap.summary_plot(shap_values, X_test_genus_display, show=False, max_display=top_n_global)
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
                else:  # Local Sample Explanation
                    st.subheader("Local Sample Explanation")
                    
                    # Calculate SHAP values for the selected sample
                    sample = X_test_genus.iloc[sample_idx:sample_idx+1]
                    shap_values_sample = explainer.shap_values(sample)
                    
                    true_value = float(y_test.iloc[sample_idx])
                    pred_value = float(model.predict(sample)[0])
                    # Handle expected_value which may be an array for multi-output models
                    expected_val = explainer.expected_value
                    if isinstance(expected_val, np.ndarray):
                        base_value = float(expected_val.flatten()[0])
                    else:
                        base_value = float(expected_val)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sample Index", f"{sample_idx}")
                    with col2:
                        st.metric("True Age (days)", f"{true_value:.1f}")
                    with col3:
                        st.metric("Predicted Age (days)", f"{pred_value:.1f}")
                    with col4:
                        st.metric("Base Value", f"{base_value:.1f}")
                    
                    error = abs(true_value - pred_value)
                    if error < 3.5:
                        st.success(f"Excellent prediction! Error: {error:.1f} days")
                    elif error < 7.0:
                        st.info(f"Good prediction! Error: {error:.1f} days")
                    else:
                        st.warning(f"Significant error: {error:.1f} days")
                    
                    # Create waterfall plot
                    st.subheader("SHAP Waterfall Plot")
                    st.info("The waterfall plot shows how each feature contributes to push the prediction from the base value to the final prediction.")
                    
                    # Create a renamed version for display
                    feature_names_display = [extract_genus_name(col) for col in X_test_genus.columns]
                    
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values_sample[0],
                            base_values=base_value,
                            data=sample.values[0],
                            feature_names=feature_names_display
                        ),
                        show=False,
                        max_display=15
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
                    # Force plot
                    st.subheader("SHAP Force Plot")
                    st.info("The force plot visualizes which features push the prediction higher (red) or lower (blue).")
                    
                    # Create force plot
                    force_plot = shap.force_plot(
                        base_value,
                        shap_values_sample[0],
                        sample.values[0],
                        feature_names=feature_names_display,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(force_plot, bbox_inches='tight')
                    plt.close()
                    
                    # Top contributing features
                    st.subheader("Top Contributing Features")
                    
                    # Get top features by absolute SHAP value
                    shap_df = pd.DataFrame({
                        'Feature': X_test_genus.columns,
                        'Genus': [extract_genus_name(col) for col in X_test_genus.columns],
                        'SHAP Value': shap_values_sample[0],
                        'Feature Value': sample.values[0]
                    })
                    shap_df['Abs SHAP'] = np.abs(shap_df['SHAP Value'])
                    shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(15)
                    
                    fig = go.Figure()
                    colors = ['red' if c < 0 else 'green' for c in shap_df['SHAP Value']]
                    fig.add_trace(go.Bar(
                        y=shap_df['Genus'],
                        x=shap_df['SHAP Value'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{c:.3f}" for c in shap_df['SHAP Value']],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title=f'Top 15 SHAP Values for Sample {sample_idx}',
                        xaxis_title='SHAP Value',
                        yaxis_title='Genus',
                        height=500,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    **How to interpret:**
                    - Green bars (positive SHAP) push the prediction toward higher age (more days)
                    - Red bars (negative SHAP) push the prediction toward lower age (fewer days)
                    - Longer bars have stronger impact on the prediction
                    - The sum of all SHAP values plus the base value equals the final prediction
                    """)
