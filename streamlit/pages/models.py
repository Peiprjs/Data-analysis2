import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from utils.data_loader import get_train_test_split, apply_clr_transformation, filter_genus_features
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return {
        'Model': model_name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R2': train_r2,
        'Test R2': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    }

def app():
    st.title("Model Training")
    
    st.markdown("""
    This section trains and evaluates multiple machine learning models on the microbiome data.
    """)
    
    tabs = st.tabs(["Model Selection", "Random Forest", "XGBoost", "Gradient Boosting", "LightGBM", "Model Comparison"])
    
    with st.spinner("Loading and preprocessing data..."):
        X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()
        X_train_clr, X_test_clr = apply_clr_transformation(X_train, X_test)
        
        X_train_genus = filter_genus_features(X_train_clr)
        X_test_genus = filter_genus_features(X_test_clr)
    
    with tabs[0]:
        st.header("Model Selection Overview")
        
        st.markdown("""
        Multiple regression models are evaluated to predict age group from microbiome data:
        
        - **Random Forest**: Ensemble of decision trees with bagging
        - **XGBoost**: Gradient boosting with advanced regularization
        - **Gradient Boosting**: Sequential ensemble learning
        - **LightGBM**: Fast gradient boosting framework
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Dimensions")
            st.metric("Training Samples", len(X_train_genus))
            st.metric("Test Samples", len(X_test_genus))
            st.metric("Features (Genus Level)", X_train_genus.shape[1])
        
        with col2:
            st.subheader("Target Variable")
            st.metric("Unique Age Groups", len(np.unique(y_train)))
            st.write("Distribution:")
            dist_df = pd.DataFrame({
                'Age Group': np.unique(y_train),
                'Count': [np.sum(y_train == i) for i in np.unique(y_train)]
            })
            st.dataframe(dist_df, use_container_width=True)
    
    with tabs[1]:
        st.header("Random Forest Regressor")
        
        st.markdown("""
        Random Forest is an ensemble learning method that builds multiple decision trees 
        and merges them together to get a more accurate and stable prediction.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Hyperparameters")
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth", 5, 50, 20, 5)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
        
        if st.button("Train Random Forest", key='rf_train'):
            with st.spinner("Training Random Forest..."):
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1
                )
                
                results = train_and_evaluate_model(model, X_train_genus, X_test_genus, y_train, y_test, "Random Forest")
                
                with col2:
                    st.subheader("Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'R2 Score', 'MAE'],
                        'Training': [results['Train RMSE'], results['Train R2'], results['Train MAE']],
                        'Testing': [results['Test RMSE'], results['Test R2'], results['Test MAE']]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    ax1.scatter(y_train, model.predict(X_train_genus), alpha=0.5, label='Train')
                    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
                    ax1.set_xlabel('True Age Group')
                    ax1.set_ylabel('Predicted Age Group')
                    ax1.set_title('Training Set Predictions')
                    ax1.legend()
                    
                    ax2.scatter(y_test, model.predict(X_test_genus), alpha=0.5, label='Test', color='orange')
                    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax2.set_xlabel('True Age Group')
                    ax2.set_ylabel('Predicted Age Group')
                    ax2.set_title('Test Set Predictions')
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X_train_genus.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(20)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(feature_importance)), feature_importance['Importance'].values)
                    ax.set_yticks(range(len(feature_importance)))
                    ax.set_yticklabels(feature_importance['Feature'].values, fontsize=8)
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 20 Feature Importances')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tabs[2]:
        st.header("XGBoost Regressor")
        
        st.markdown("""
        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient and flexible.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Hyperparameters")
            xgb_n_estimators = st.slider("Number of Estimators", 50, 300, 100, 50, key='xgb_n')
            xgb_max_depth = st.slider("Max Depth", 3, 15, 6, key='xgb_depth')
            xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key='xgb_lr')
        
        if st.button("Train XGBoost", key='xgb_train'):
            with st.spinner("Training XGBoost..."):
                model = xgb.XGBRegressor(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate,
                    random_state=42,
                    n_jobs=-1
                )
                
                results = train_and_evaluate_model(model, X_train_genus, X_test_genus, y_train, y_test, "XGBoost")
                
                with col2:
                    st.subheader("Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'R2 Score', 'MAE'],
                        'Training': [results['Train RMSE'], results['Train R2'], results['Train MAE']],
                        'Testing': [results['Test RMSE'], results['Test R2'], results['Test MAE']]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    ax1.scatter(y_train, model.predict(X_train_genus), alpha=0.5, label='Train')
                    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
                    ax1.set_xlabel('True Age Group')
                    ax1.set_ylabel('Predicted Age Group')
                    ax1.set_title('Training Set Predictions')
                    ax1.legend()
                    
                    ax2.scatter(y_test, model.predict(X_test_genus), alpha=0.5, label='Test', color='orange')
                    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax2.set_xlabel('True Age Group')
                    ax2.set_ylabel('Predicted Age Group')
                    ax2.set_title('Test Set Predictions')
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tabs[3]:
        st.header("Gradient Boosting Regressor")
        
        st.markdown("""
        Gradient Boosting builds an additive model in a forward stage-wise fashion.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Hyperparameters")
            gb_n_estimators = st.slider("Number of Estimators", 50, 300, 100, 50, key='gb_n')
            gb_max_depth = st.slider("Max Depth", 3, 10, 5, key='gb_depth')
            gb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key='gb_lr')
        
        if st.button("Train Gradient Boosting", key='gb_train'):
            with st.spinner("Training Gradient Boosting..."):
                model = GradientBoostingRegressor(
                    n_estimators=gb_n_estimators,
                    max_depth=gb_max_depth,
                    learning_rate=gb_learning_rate,
                    random_state=42
                )
                
                results = train_and_evaluate_model(model, X_train_genus, X_test_genus, y_train, y_test, "Gradient Boosting")
                
                with col2:
                    st.subheader("Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'R2 Score', 'MAE'],
                        'Training': [results['Train RMSE'], results['Train R2'], results['Train MAE']],
                        'Testing': [results['Test RMSE'], results['Test R2'], results['Test MAE']]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
    
    with tabs[4]:
        st.header("LightGBM Regressor")
        
        st.markdown("""
        LightGBM is a gradient boosting framework that uses tree-based learning algorithms with high efficiency.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Hyperparameters")
            lgb_n_estimators = st.slider("Number of Estimators", 50, 300, 100, 50, key='lgb_n')
            lgb_max_depth = st.slider("Max Depth", 3, 15, 6, key='lgb_depth')
            lgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key='lgb_lr')
        
        if st.button("Train LightGBM", key='lgb_train'):
            with st.spinner("Training LightGBM..."):
                model = lgb.LGBMRegressor(
                    n_estimators=lgb_n_estimators,
                    max_depth=lgb_max_depth,
                    learning_rate=lgb_learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                
                results = train_and_evaluate_model(model, X_train_genus, X_test_genus, y_train, y_test, "LightGBM")
                
                with col2:
                    st.subheader("Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'R2 Score', 'MAE'],
                        'Training': [results['Train RMSE'], results['Train R2'], results['Train MAE']],
                        'Testing': [results['Test RMSE'], results['Test R2'], results['Test MAE']]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
    
    with tabs[5]:
        st.header("Model Comparison")
        
        st.markdown("""
        Compare performance across all trained models.
        """)
        
        if st.button("Train All Models", key='train_all'):
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_list = []
            total_models = len(models)
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}... ({idx + 1}/{total_models})")
                progress_bar.progress((idx + 1) / total_models)
                
                result = train_and_evaluate_model(model, X_train_genus, X_test_genus, y_train, y_test, name)
                results_list.append(result)
            
            status_text.text("All models trained successfully!")
            progress_bar.empty()
            status_text.empty()
            
            results_df = pd.DataFrame(results_list)
            
            st.subheader("Performance Metrics")
            st.dataframe(results_df, use_container_width=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].bar(results_df['Model'], results_df['Test RMSE'])
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].set_title('Test RMSE by Model')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            axes[0, 1].bar(results_df['Model'], results_df['Test R2'])
            axes[0, 1].set_ylabel('R2 Score')
            axes[0, 1].set_title('Test R2 Score by Model')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            axes[1, 0].bar(results_df['Model'], results_df['Test MAE'])
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_title('Test MAE by Model')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            x = np.arange(len(results_df))
            width = 0.35
            axes[1, 1].bar(x - width/2, results_df['Train RMSE'], width, label='Train')
            axes[1, 1].bar(x + width/2, results_df['Test RMSE'], width, label='Test')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].set_title('Train vs Test RMSE')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(results_df['Model'], rotation=45)
            axes[1, 1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            best_model_idx = results_df['Test R2'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'Model']
            best_r2 = results_df.loc[best_model_idx, 'Test R2']
            
            st.success(f"Best Model: {best_model_name} with Test R2 = {best_r2:.4f}")
