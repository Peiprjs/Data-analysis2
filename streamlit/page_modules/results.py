import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

def app():
    st.title("Results Comparison")
    
    st.markdown("""
    This section compares model performance across different configurations and feature sets. 
    Below we summarize results from the modeling notebooks, including held-out test scores and 
    visual diagnostics for the strongest models.
    """)
    
    tabs = st.tabs(["Feature Set Comparison", "Cross-Validation", "Prediction Analysis", "Notebook Results"])
    
    with st.spinner("Loading and preprocessing data..."):
        X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
        X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
        
        X_train_genus = filter_genus_features(X_train_clr)
        X_test_genus = filter_genus_features(X_test_clr)
    
    with tabs[0]:
        st.header("Feature Set Comparison")
        
        st.markdown("""
        Compare model performance using different feature sets:
        - All features (all taxonomic levels)
        - Genus-level features only
        """)
        
        if st.button("Compare Feature Sets", key='fs_compare'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
            
            status_text.text("Training on all features... (1/2)")
            progress_bar.progress(0.25)
            model_all = model.fit(X_train_clr, y_train)
            progress_bar.progress(0.5)
            y_pred_all = model_all.predict(X_test_clr)
            rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))
            r2_all = r2_score(y_test, y_pred_all)
            mae_all = mean_absolute_error(y_test, y_pred_all)
            
            status_text.text("Training on genus features... (2/2)")
            progress_bar.progress(0.75)
            model_genus = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1)
            model_genus = model_genus.fit(X_train_genus, y_train)
            progress_bar.progress(0.9)
            y_pred_genus = model_genus.predict(X_test_genus)
            rmse_genus = np.sqrt(mean_squared_error(y_test, y_pred_genus))
            r2_genus = r2_score(y_test, y_pred_genus)
            mae_genus = mean_absolute_error(y_test, y_pred_genus)
            
            progress_bar.progress(1.0)
            status_text.text("Comparison complete!")
            progress_bar.empty()
            status_text.empty()
            
            results_df = pd.DataFrame({
                'Feature Set': ['All Features', 'Genus Only'],
                'Num Features': [X_train_clr.shape[1], X_train_genus.shape[1]],
                'RMSE': [rmse_all, rmse_genus],
                'R2 Score': [r2_all, r2_genus],
                'MAE': [mae_all, mae_genus]
            })
            
            st.subheader("Performance Comparison")
            st.dataframe(results_df, use_container_width=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].bar(results_df['Feature Set'], results_df['RMSE'])
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('RMSE by Feature Set')
            axes[0].tick_params(axis='x', rotation=15)
            
            axes[1].bar(results_df['Feature Set'], results_df['R2 Score'])
            axes[1].set_ylabel('R2 Score')
            axes[1].set_title('R2 Score by Feature Set')
            axes[1].tick_params(axis='x', rotation=15)
            
            axes[2].bar(results_df['Feature Set'], results_df['MAE'])
            axes[2].set_ylabel('MAE')
            axes[2].set_title('MAE by Feature Set')
            axes[2].tick_params(axis='x', rotation=15)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            if r2_genus > r2_all:
                st.success("Genus-level features provide better performance with fewer features!")
            else:
                st.info("All features provide slightly better performance, but genus features are more interpretable.")
    
    with tabs[1]:
        st.header("Cross-Validation Analysis")
        
        st.markdown("""
        Evaluate model robustness using k-fold cross-validation.
        """)
        
        cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
        
        if st.button("Run Cross-Validation", key='cv_run'):
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=3004, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=3004),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=3004)
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cv_results = []
            total_models = len(models)
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Cross-validating {name}... ({idx + 1}/{total_models})")
                progress_bar.progress((idx + 1) / total_models)
                
                scores = cross_val_score(model, X_train_genus, y_train, cv=cv_folds, 
                                       scoring='r2', n_jobs=-1)
                cv_results.append({
                    'Model': name,
                    'Mean R2': scores.mean(),
                    'Std R2': scores.std(),
                    'Min R2': scores.min(),
                    'Max R2': scores.max()
                })
            
            status_text.text("Cross-validation completed!")
            progress_bar.empty()
            status_text.empty()
            
            cv_df = pd.DataFrame(cv_results)
            
            st.subheader(f"Cross-Validation Results ({cv_folds}-Fold)")
            st.dataframe(cv_df, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x_pos = np.arange(len(cv_df))
            ax.bar(x_pos, cv_df['Mean R2'], yerr=cv_df['Std R2'], 
                  capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cv_df['Model'], rotation=15)
            ax.set_ylabel('R2 Score')
            ax.set_title(f'Cross-Validation R2 Scores ({cv_folds}-Fold)')
            ax.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
            best_model = cv_df.loc[cv_df['Mean R2'].idxmax(), 'Model']
            best_score = cv_df.loc[cv_df['Mean R2'].idxmax(), 'Mean R2']
            st.success(f"Best Model: {best_model} with Mean R2 = {best_score:.4f}")
    
    with tabs[2]:
        st.header("Prediction Analysis")
        
        st.markdown("""
        Analyze prediction patterns and errors across different models.
        """)
        
        if st.button("Generate Prediction Analysis", key='pred_analysis'):
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=3004, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=3004, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=3004),
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            predictions = {}
            total_models = len(models)
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Training and predicting with {name}... ({idx + 1}/{total_models})")
                progress_bar.progress((idx + 1) / total_models)
                
                model.fit(X_train_genus, y_train)
                predictions[name] = model.predict(X_test_genus)
            
            status_text.text("All predictions generated!")
            progress_bar.empty()
            status_text.empty()
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.ravel()
            
            for idx, (name, y_pred) in enumerate(predictions.items()):
                ax = axes[idx]
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel('True Age (days)')
                ax.set_ylabel('Predicted Age (days)')
                ax.set_title(f'{name}\nR2 = {r2_score(y_test, y_pred):.3f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes[3].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

    with tabs[3]:
        st.header("Summary of Notebook Results")

        st.markdown(
            """
            Key findings extracted from the modeling notebooks (predicting_models.ipynb and Finalized Models.ipynb):
            """
        )

        summary_df = pd.DataFrame(_NOTEBOOK_RESULTS)

        st.dataframe(summary_df, use_container_width=True)

        st.markdown("### Visualizations from evaluation")

        col1, col2 = st.columns(2)
        with col1:
            image_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "battle_results.png")
            st.image(image_path, caption="Model comparison heatmap (from notebooks)", use_column_width=True)
        with col2:
            st.pyplot(_notebook_metric_barplot())


def _notebook_metric_barplot():
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered = [item for item in _NOTEBOOK_RESULTS if "Neural" not in item["Model"]]
    models = [item["Model"] for item in filtered]
    r2_scores = [item["Test R2"] for item in filtered]
    rmse_scores = [item["Test RMSE"] for item in filtered]

    ax.bar(models, r2_scores, color="#2E7D32")
    ax.set_ylabel("Test R2")
    ax.set_title("Notebook R2 by model")
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(models, rmse_scores, color="#1565C0", marker="o", label="RMSE")
    ax2.set_ylabel("Test RMSE")
    ax2.legend(loc="lower right")
    plt.close(fig)
    return fig


_NOTEBOOK_RESULTS = [
    {"Model": "Random Forest (tuned)", "Test R2": 0.42, "Test RMSE": 10.8, "Notes": "Strong baseline with feature importance"},
    {"Model": "XGBoost", "Test R2": 0.44, "Test RMSE": 10.5, "Notes": "Best overall generalization"},
    {"Model": "Gradient Boosting", "Test R2": 0.38, "Test RMSE": 11.4, "Notes": "Stable but slightly weaker"},
    {"Model": "LightGBM", "Test R2": 0.40, "Test RMSE": 11.0, "Notes": "Competitive with fast training"},
    {"Model": "Neural Network (trial)", "Test R2": 0.30, "Test RMSE": 12.6, "Notes": "Exploratory; not selected"},
]
# Results extracted from notebooks on 2026-01-29. Update if notebook metrics change.
