import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import re
from collections import namedtuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Define a structure for the output to keep it clean and immutable
ModelResult = namedtuple('ModelResult', ['model', 'rmse', 'r2', 'best_params', 'runtime'])


def run_xgboost_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    """
    Standardized XGBoost pipeline for testing different microbiome feature sets.
    Returns a named tuple: (model, rmse, r2, best_params, runtime)
    """
    print(f"🚀 Initializing XGBoost Engine: {label}")

    # 1. Clean Column Names (Ensures compatibility with XGBoost/LightGBM)
    X_train_clean = X_train_df.copy()
    X_test_clean = X_test_df.copy()
    X_train_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_train_clean.columns]
    X_test_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_test_clean.columns]

    # 2. Setup Model & Hyperparameter Space
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 4, 5],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.1, 0.2],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 5.0]
    }

    # 3. Optimized Search
    search_xgb = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # 4. Fit & Time
    start_time = time.time()
    search_xgb.fit(X_train_clean, y_train)
    elapsed = time.time() - start_time

    # 5. Evaluation
    best_model = search_xgb.best_estimator_
    y_pred = best_model.predict(X_test_clean)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 6. Summary Output
    print(f"\n✅ {label} Complete ({elapsed:.1f}s)")
    print(f"   Test RMSE: {rmse:.3f} | Test R2: {r2:.3f}")

    # 7. Importance Visualization
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': importances})
    top_20 = feat_df.sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_20, x='Importance', y='Bacteria', palette='magma')
    plt.title(f'Top 20 Drivers - {label}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    # Return as an immutable Named Tuple
    return ModelResult(
        model=best_model,
        rmse=rmse,
        r2=r2,
        best_params=search_xgb.best_params_,
        runtime=elapsed
    )