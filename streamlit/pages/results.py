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
    """)
    
    tabs = st.tabs(["Feature Set Comparison", "Cross-Validation", "Prediction Analysis"])
    
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
            
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            
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
            model_genus = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
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
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
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
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
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
                ax.set_xlabel('True Age Group')
                ax.set_ylabel('Predicted Age Group')
                ax.set_title(f'{name}\nR2 = {r2_score(y_test, y_pred):.3f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes[3].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Residual Analysis")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, (name, y_pred) in enumerate(predictions.items()):
                residuals = y_test - y_pred
                axes[idx].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel('Residuals')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{name}\nMean = {residuals.mean():.3f}')
                axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Error Metrics Summary")
            
            error_metrics = []
            for name, y_pred in predictions.items():
                error_metrics.append({
                    'Model': name,
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'Max Error': np.abs(y_test - y_pred).max()
                })
            
            error_df = pd.DataFrame(error_metrics)
            st.dataframe(error_df, use_container_width=True)
            
            st.subheader("Consensus Prediction")
            
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ensemble R2", f"{ensemble_r2:.4f}")
            with col2:
                st.metric("Ensemble RMSE", f"{ensemble_rmse:.4f}")
            with col3:
                improvement = ensemble_r2 - error_df['R2'].max()
                st.metric("Improvement over Best", f"{improvement:.4f}")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, ensemble_pred, alpha=0.5, color='purple')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('True Age Group')
            ax.set_ylabel('Ensemble Predicted Age Group')
            ax.set_title(f'Ensemble Predictions (Mean of all models)\nR2 = {ensemble_r2:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
