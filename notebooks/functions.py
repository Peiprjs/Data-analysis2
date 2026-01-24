import pandas as pd
import numpy as np
import time
import re
import random
import os
import gc
from collections import namedtuple
from tqdm import tqdm

# Machine Learning & Stats
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, callbacks, initializers, regularizers
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, f_classif, SelectKBest, RFE
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline

# Model Interpretability
import lime
import lime.lime_tabular
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================================================
# 1. GLOBAL CONFIG & SYSTEM-AGNOSTIC DEVICE SETUP
# =========================================================
ModelResult = namedtuple('ModelResult', ['model', 'rmse', 'r2', 'best_params', 'runtime', 'top_features'])
NNResult = namedtuple('NNResult', ['X_train_elite', 'X_test_elite', 'feature_names', 'rmse', 'n_features'])

# CRITICAL FOR ROCm: These must be set BEFORE any TF operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# This helps with the '0 MB memory' error by using the system allocator
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Detect Device
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Re-verify visibility and growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Test if GPU is actually functional with a small tensor
        with tf.device('/GPU:0'):
            _ = tf.constant([1.0])

        DEVICE = '/GPU:0'
        print(f"GPU Active: {gpus[0].name}")
    except Exception as e:
        print(f"GPU Initialization Failed: {e}. Switching to CPU for stability.")
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    print("No GPU found. Running on CPU mode.")


def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Global seeds set to {seed}")


# =========================================================
# 2. NEURAL NETWORK SELECTOR (Robust Version)
# =========================================================
class GatekeeperLayer(layers.Layer):
    def __init__(self, num_features, l1_penalty=0.01, **kwargs):
        super(GatekeeperLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l1_penalty = float(l1_penalty)  # Cast early to avoid tensor issues

    def build(self, input_shape):
        # We use a lambda for the initializer to ensure it's created on the correct device
        self.w = self.add_weight(
            name="gate_weights",
            shape=(self.num_features,),
            initializer=initializers.Ones(),
            trainable=True,
            constraint=constraints.NonNeg(),
            regularizer=regularizers.l1(self.l1_penalty)
        )

    def call(self, inputs):
        return inputs * self.w


def nn_feature_search(
        X_train, X_test, Y_train,
        target_range=(50, 1250),
        consensus_threshold=0.7,
        # Hardware/Speed Parameters
        batch_size=64,  # 1024 for Radeon 7900 / 64 for CPU
        jit_compile=False,  # True for 7900 (XLA) / False for CPU
        mixed_precision=False,  # True for 7900 (FP16) / False for CPU
        device="/CPU:0"  # "/GPU:0" for GPU or Mac M-chips
):
    base_lr = 0.01 if batch_size >= 4096 else (0.004 if batch_size > 512 else 0.001)
    repeats = 10
    epochs = 400
    patience = 50
    penalties = [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]

    # Global Mixed Precision Setup
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('mixed_float16')
    else:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('float32')

    scaler_x = StandardScaler()
    X_train_tf = scaler_x.fit_transform(X_train).astype('float32')
    y_train_tf = Y_train.values.astype('float32')

    penalty_results = []
    print(f"🧬 Search Started | Device: {device} | Batch: {batch_size}")
    print("-" * 85)
    print(f"{'Penalty':<8} | {'Features':<10} | {'RMSE (Avg)':<12} | {'R2 (Avg)':<10} | {'Efficiency'}")
    print("-" * 85)

    for p in tqdm(penalties, desc="Testing penalties"):
        batch_weights, batch_rmse, batch_r2 = [], [], []
        tf.keras.backend.clear_session()
        gc.collect()

        for i in tqdm(range(repeats), desc=f"Penalty {p}", leave=False):
            with tf.device(device):
                # ARCHITECTURE: Fixed logic to ensure scientific consistency
                inputs = layers.Input(shape=(X_train_tf.shape[1],))
                gate = GatekeeperLayer(X_train_tf.shape[1], l1_penalty=p)(inputs)

                # L2 Regularization and Dropout are "Hardcoded" to prevent overfitting
                x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(gate)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
                outputs = layers.Dense(1, activation='linear', dtype="float32")(x)

                model = models.Model(inputs=inputs, outputs=outputs)
                steps_per_epoch = int(np.ceil(len(X_train_tf) / batch_size))

                # LEARNING RATE: Adaptive schedule to handle large batches safely
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.0,
                    decay_steps=epochs*steps_per_epoch,
                    warmup_target=base_lr,
                    warmup_steps=20 * steps_per_epoch
                )

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss='mse',
                    jit_compile=jit_compile
                )

                model.fit(
                    X_train_tf, y_train_tf,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
                )

            # Record weights and predict
            w = model.layers[1].get_weights()[0]
            batch_weights.append(w > 1e-2)
            y_pred = model.predict(X_train_tf, batch_size=batch_size, verbose=0)
            batch_rmse.append(np.sqrt(mean_squared_error(y_train_tf, y_pred)))
            batch_r2.append(r2_score(y_train_tf, y_pred))

        avg_rmse = np.mean(batch_rmse)
        avg_r2 = np.mean(batch_r2)
        selection_frequency = np.mean(batch_weights, axis=0)
        consensus_feats = np.sum(selection_frequency >= consensus_threshold)
        eff = avg_r2 / np.log1p(consensus_feats) if consensus_feats > 0 else 0
        print(f"{p:<8} | {consensus_feats:<10} | {avg_rmse:<12.2f} | {avg_r2:<10.2f} | {eff:.4f}")

        penalty_results.append({
            'penalty': p,
            'n_features': consensus_feats,
            'rmse': avg_rmse,
            'r2': avg_r2,
            'efficiency': eff,  # Store this so champion logic can find it
            'freq_mask': selection_frequency
        })

    # Champion Selection...
    valid_results = [r for r in penalty_results if target_range[0] < r['n_features'] < target_range[1]]
    if not valid_results: return None
    champion = max(valid_results, key=lambda x: x['efficiency'])

    elite_mask = champion['freq_mask'] >= consensus_threshold
    elite_names = X_train.columns[elite_mask].tolist()

    return NNResult(X_train[elite_names], X_test[elite_names], elite_names, champion['rmse'], champion['n_features'])

# =========================================================
# 3. CLASSICAL ML BENCHMARKS (XGB & RF)
# =========================================================

def run_model_benchmark(estimator, param_distributions, X_train_df, X_test_df, y_train, y_test, 
                        label="Dataset", n_iter=15, cv=5, task='regression', clean_columns=False):
    """
    Generic function to run model benchmarking with hyperparameter search.
    
    Parameters:
    - estimator: Model estimator
    - param_distributions: Dictionary of hyperparameter distributions
    - X_train_df, X_test_df: Training and test features
    - y_train, y_test: Training and test targets
    - label: Dataset label for logging
    - n_iter: Number of iterations for RandomizedSearchCV
    - cv: Number of cross-validation folds
    - task: 'regression' or 'classification'
    - clean_columns: Whether to clean column names
    
    Returns:
    - ModelResult for regression or dict for classification
    """
    X_train_clean = X_train_df.copy()
    X_test_clean = X_test_df.copy()
    
    if clean_columns:
        X_train_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_train_clean.columns]
        X_test_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_test_clean.columns]
    
    scoring = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'
    
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train_clean, y_train)
    elapsed = time.time() - start_time
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_clean)
    
    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        if hasattr(best_model, 'feature_importances_'):
            feat_df = pd.DataFrame({
                'Bacteria': X_train_df.columns,
                'Importance': best_model.feature_importances_
            })
            top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
        else:
            top_drivers = None
        
        print(f"\n{label} Complete ({elapsed:.1f}s) | R2: {r2:.3f}")
        return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)
    
    else:
        accuracy = accuracy_score(y_test, y_pred)
        
        if hasattr(best_model, 'feature_importances_'):
            feat_df = pd.DataFrame({
                'Bacteria': X_train_df.columns,
                'Importance': best_model.feature_importances_
            })
            top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
        else:
            top_drivers = None
        
        print(f"\n{label} Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
        return {
            'model': best_model,
            'accuracy': accuracy,
            'best_params': search.best_params_,
            'runtime': elapsed,
            'top_features': top_drivers
        }


def xgboost_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing XGBoost Engine: {label}")
    X_train_clean = X_train_df.copy()
    X_test_clean = X_test_df.copy()
    X_train_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_train_clean.columns]
    X_test_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_test_clean.columns]

    search_xgb = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        param_distributions={"n_estimators": [500, 1000], "learning_rate": [0.01, 0.05], "max_depth": [3, 4, 5],
                             "subsample": [0.7, 0.8], "colsample_bytree": [0.1, 0.2], "reg_alpha": [0.1, 0.5, 1.0],
                             "reg_lambda": [1.0, 5.0]},
        n_iter=15, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )

    start_time = time.time()
    search_xgb.fit(X_train_clean, y_train)
    elapsed = time.time() - start_time

    best_model = search_xgb.best_estimator_
    y_pred = best_model.predict(X_test_clean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search_xgb.best_params_, elapsed, top_drivers)


def random_forest_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Random Forest Engine: {label}")
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={"n_estimators": [300, 500, 800, 1000], "max_depth": [None, 10, 20, 40],
                             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                             "max_features": ["sqrt", "log2"]},
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} RF Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def adaboost_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing AdaBoost Engine: {label}")
    search = RandomizedSearchCV(
        estimator=AdaBoostRegressor(random_state=42),
        param_distributions={
            "n_estimators": [50, 100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "loss": ["linear", "square", "exponential"]
        },
        n_iter=15, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} AdaBoost Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def gradient_boosting_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Gradient Boosting Engine: {label}")
    search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions={
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6, 10, 15, 20, 40],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 2, 4, 10, 50],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", None]
        },
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} Gradient Boosting Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def lightgbm_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing LightGBM Engine: {label}")
    search = RandomizedSearchCV(
        estimator=lgb.LGBMRegressor(random_state=42, device = "cpu", n_jobs=1, verbose=1, importance_type="gain"),
        param_distributions={
            "n_estimators": [100, 200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7, -1, 10, 20],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20, 30, 40],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0]
        },
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate Top Drivers
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} LightGBM Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def stochastic_gradient_boosting_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Stochastic Gradient Boosting: {label}")
    search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions={
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6, 7],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            "loss": ["squared_error", "absolute_error", "huber"]
        },
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} Stochastic GB Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def svc_classifier_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing SVC Classifier: {label}")
    search = RandomizedSearchCV(
        estimator=SVC(random_state=42),
        param_distributions={
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4]
        },
        n_iter=15, cv=5, scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{label} SVC Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
    
    return {'model': best_model, 'accuracy': accuracy, 'best_params': search.best_params_, 'runtime': elapsed}


def logistic_regression_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Logistic Regression: {label}")
    search = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_distributions={
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet", None],
            "solver": ["lbfgs", "saga", "liblinear"],
            "class_weight": [None, "balanced"]
        },
        n_iter=15, cv=5, scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{label} Logistic Regression Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
    
    return {'model': best_model, 'accuracy': accuracy, 'best_params': search.best_params_, 'runtime': elapsed}


def random_forest_classifier_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Random Forest Classifier: {label}")
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": [None, "balanced", "balanced_subsample"]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
    
    print(f"\n{label} RF Classifier Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
    
    return {'model': best_model, 'accuracy': accuracy, 'best_params': search.best_params_, 'runtime': elapsed, 'top_features': top_drivers}


def gradient_boosting_classifier_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"Initializing Gradient Boosting Classifier: {label}")
    search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", 0.5]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
    
    print(f"\n{label} GB Classifier Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
    
    return {'model': best_model, 'accuracy': accuracy, 'best_params': search.best_params_, 'runtime': elapsed, 'top_features': top_drivers}


def kernel_random_forest_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset", kernel_method='nystroem'):
    """
    Kernel Random Forest using kernel approximation followed by Random Forest.
    
    Parameters:
    - X_train_df, X_test_df: Training and test features
    - y_train, y_test: Training and test targets
    - label: Dataset label for logging
    - kernel_method: 'nystroem' or 'rbf' for kernel approximation method
    
    Returns:
    - ModelResult with performance metrics
    """
    print(f"Initializing Kernel Random Forest ({kernel_method}): {label}")
    
    param_distributions = {
        'kernel_approx__gamma': [0.001, 0.01, 0.1, 1.0, 10.0],
        'kernel_approx__n_components': [50, 100, 200, 300],
        'rf__n_estimators': [100, 200, 500],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__max_features': ['sqrt', 'log2', 0.5]
    }
    
    if kernel_method == 'nystroem':
        kernel_approx = Nystroem(kernel='rbf', random_state=42)
    elif kernel_method == 'rbf':
        kernel_approx = RBFSampler(random_state=42)
    else:
        raise ValueError(f"Unknown kernel_method: {kernel_method}. Use 'nystroem' or 'rbf'")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_approx', kernel_approx),
        ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    rf_model = best_model.named_steps['rf']
    kernel_transformed_feature_names = [f"kernel_feature_{i}" for i in range(best_model.named_steps['kernel_approx'].n_components)]
    feat_df = pd.DataFrame({
        'Feature': kernel_transformed_feature_names,
        'Importance': rf_model.feature_importances_
    })
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
    
    print(f"\n{label} Kernel RF Complete ({elapsed:.1f}s) | R2: {r2:.3f}")
    print(f"Kernel method: {kernel_method}, n_components: {best_model.named_steps['kernel_approx'].n_components}")
    
    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


def kernel_random_forest_classifier_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset", kernel_method='nystroem'):
    """
    Kernel Random Forest Classifier using kernel approximation followed by Random Forest Classifier.
    
    Parameters:
    - X_train_df, X_test_df: Training and test features
    - y_train, y_test: Training and test targets
    - label: Dataset label for logging
    - kernel_method: 'nystroem' or 'rbf' for kernel approximation method
    
    Returns:
    - Dict with model, accuracy, and other metrics
    """
    print(f"Initializing Kernel Random Forest Classifier ({kernel_method}): {label}")
    
    param_distributions = {
        'kernel_approx__gamma': [0.001, 0.01, 0.1, 1.0, 10.0],
        'kernel_approx__n_components': [50, 100, 200, 300],
        'rf__n_estimators': [100, 200, 500],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__max_features': ['sqrt', 'log2', 0.5],
        'rf__class_weight': [None, 'balanced']
    }
    
    if kernel_method == 'nystroem':
        kernel_approx = Nystroem(kernel='rbf', random_state=42)
    elif kernel_method == 'rbf':
        kernel_approx = RBFSampler(random_state=42)
    else:
        raise ValueError(f"Unknown kernel_method: {kernel_method}. Use 'nystroem' or 'rbf'")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_approx', kernel_approx),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train_df, y_train)
    elapsed = time.time() - start_time
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)
    
    rf_model = best_model.named_steps['rf']
    kernel_transformed_feature_names = [f"kernel_feature_{i}" for i in range(best_model.named_steps['kernel_approx'].n_components)]
    feat_df = pd.DataFrame({
        'Feature': kernel_transformed_feature_names,
        'Importance': rf_model.feature_importances_
    })
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)
    
    print(f"\n{label} Kernel RF Classifier Complete ({elapsed:.1f}s) | Accuracy: {accuracy:.3f}")
    print(f"Kernel method: {kernel_method}, n_components: {best_model.named_steps['kernel_approx'].n_components}")
    
    return {
        'model': best_model,
        'accuracy': accuracy,
        'best_params': search.best_params_,
        'runtime': elapsed,
        'top_features': top_drivers
    }


# =========================================================
# 4. REPEATED CV BATTLE (5x5 Arena)
# =========================================================
def final_battle(datasets_dict, y_train, n_splits=5, n_repeats=5, xgb_params=None, rf_params=None, 
                 gb_params=None, lgbm_params=None, ada_params=None):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3004)

    default_xgb = {
        'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 3,
        'subsample': 0.7, 'colsample_bytree': 0.2, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'n_jobs': -1, 'random_state': 3004
    }

    default_rf = {
        'n_estimators': 1000, 'max_depth': None,
        'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 3004
    }

    default_gb = {
        'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4,
        'subsample': 0.8, 'max_features': 'sqrt', 'random_state': 3004
    }

    default_lgbm = {
        'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5,
        'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'n_jobs': -1, 'random_state': 3004, 'verbose': -1
    }

    default_ada = {
        'n_estimators': 100, 'learning_rate': 0.1,
        'loss': 'square', 'random_state': 3004
    }

    current_xgb_params = xgb_params if xgb_params else default_xgb
    current_rf_params = rf_params if rf_params else default_rf
    current_gb_params = gb_params if gb_params else default_gb
    current_lgbm_params = lgbm_params if lgbm_params else default_lgbm
    current_ada_params = ada_params if ada_params else default_ada

    models_to_test = {
        "XGBoost": xgb.XGBRegressor(**current_xgb_params),
        "Random Forest": RandomForestRegressor(**current_rf_params),
        "Gradient Boosting": GradientBoostingRegressor(**current_gb_params),
        "LightGBM": lgb.LGBMRegressor(**current_lgbm_params),
        "AdaBoost": AdaBoostRegressor(**current_ada_params)
    }

    battle_results = []
    print(f"Starting Battle Arena ({n_splits}x{n_repeats} = {n_splits * n_repeats} runs)")
    # Expanded headers to include R2
    print(f"{'Dataset':<20} | {'Model':<15} | {'Avg RMSE':<12} | {'Avg R2':<10} | {'Std Dev':<10}")
    print("-" * 85)

    for data_name, X_data in datasets_dict.items():
        X_clean = X_data.copy()
        X_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_clean.columns]

        for model_name, model_obj in models_to_test.items():
            # Modified scoring to capture both RMSE and R2
            cv_metrics = cross_validate(
                model_obj, X_clean, y_train,
                cv=rkf,
                scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
                n_jobs=-1
            )

            mean_rmse = -np.mean(cv_metrics['test_rmse'])
            std_rmse = np.std(cv_metrics['test_rmse'])
            mean_r2 = np.mean(cv_metrics['test_r2'])

            # Print updated table row
            print(f"{data_name:<20} | {model_name:<15} | {mean_rmse:.2f} days   | {mean_r2:.3f}    | ±{std_rmse:.2f}")

            battle_results.append({
                'Dataset': data_name,
                'Model': model_name,
                'RMSE_Mean': mean_rmse,
                'RMSE_Std': std_rmse,
                'R2_Mean': mean_r2
            })

    return battle_results


# =========================================================
# 5. MODEL INTERPRETABILITY (LIME & SHAP)
# =========================================================
def explain_with_lime(model, X_train, X_test, feature_names, num_samples=5, num_features=10):
    """
    Generate LIME explanations for individual predictions.
    
    Parameters:
    - model: trained model with predict method
    - X_train: training data (pandas DataFrame or numpy array)
    - X_test: test data (pandas DataFrame or numpy array)
    - feature_names: list of feature names
    - num_samples: number of test samples to explain
    - num_features: number of top features to show in explanation
    
    Returns:
    - Dictionary with explanations for each sample
    """
    # Convert to numpy if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        mode='regression',
        random_state=42
    )
    
    explanations = {}
    print(f"Generating LIME explanations for {num_samples} samples...")
    
    for i in range(min(num_samples, len(X_test_np))):
        exp = explainer.explain_instance(
            X_test_np[i],
            model.predict,
            num_features=num_features
        )
        explanations[f'sample_{i}'] = {
            'prediction': model.predict([X_test_np[i]])[0],
            'explanation': exp.as_list(),
            'score': exp.score
        }
        print(f"  Sample {i}: Prediction = {explanations[f'sample_{i}']['prediction']:.2f}")
    
    return explanations


def explain_with_shap(model, X_train, X_test, feature_names, background_samples=100, num_samples=None):
    """
    Generate SHAP explanations for model predictions.
    
    Parameters:
    - model: trained model
    - X_train: training data (pandas DataFrame or numpy array)
    - X_test: test data (pandas DataFrame or numpy array) 
    - feature_names: list of feature names
    - background_samples: number of background samples for SHAP
    - num_samples: number of test samples to explain (None = all)
    
    Returns:
    - SHAP values and explainer object
    """
    # Convert to numpy if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Use a subset for background
    background = X_train_np[:min(background_samples, len(X_train_np))]
    
    print(f"Generating SHAP explanations...")
    
    # Try TreeExplainer first (faster for tree models)
    try:
        explainer = shap.TreeExplainer(model, background)
        print("  Using TreeExplainer (optimized for tree-based models)")
    except:
        # Fall back to KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict, background)
        print("  Using KernelExplainer (model-agnostic)")
    
    # Calculate SHAP values
    if num_samples is not None:
        X_explain = X_test_np[:num_samples]
    else:
        X_explain = X_test_np
    
    shap_values = explainer.shap_values(X_explain)
    
    print(f"  SHAP values computed for {len(X_explain)} samples")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': feature_names,
        'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
    }


# =========================================================
# 6. FEATURE CUTOFF CROSS-VALIDATION
# =========================================================
def get_taxonomic_level(feature_name):
    """
    Extract taxonomic level from MetaPhlAn feature name.
    
    Taxonomic levels:
    - k__ = Kingdom
    - p__ = Phylum
    - c__ = Class
    - o__ = Order
    - f__ = Family
    - g__ = Genus
    - s__ = Species
    - t__ = Terminal (SGB - Species-Level Genome Bin)
    
    Returns the deepest taxonomic level present in the feature name.
    """
    level_order = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__', 't__']
    level_names = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'SGB']
    
    deepest_level = None
    deepest_idx = -1
    
    for idx, level in enumerate(level_order):
        if level in str(feature_name):
            if idx > deepest_idx:
                deepest_level = level_names[idx]
                deepest_idx = idx
    
    return deepest_level if deepest_level else 'Unknown'


def filter_features_by_level(X, max_level='Genus'):
    """
    Filter features to include only those at or above the specified taxonomic level.
    
    Parameters:
    - X: DataFrame with MetaPhlAn feature names as columns
    - max_level: Maximum taxonomic depth to include
    
    Returns:
    - Filtered DataFrame
    """
    level_hierarchy = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'SGB']
    max_idx = level_hierarchy.index(max_level) if max_level in level_hierarchy else len(level_hierarchy) - 1
    
    selected_features = []
    for col in X.columns:
        level = get_taxonomic_level(col)
        if level != 'Unknown':
            level_idx = level_hierarchy.index(level)
            if level_idx <= max_idx:
                selected_features.append(col)
        else:
            # Keep non-taxonomic features (like metadata)
            selected_features.append(col)
    
    return X[selected_features]


def cross_validate_feature_cutoffs(X_train, y_train, feature_levels=None, model_configs=None, cv_folds=5):
    """
    Cross-validate model performance at different taxonomic feature levels with various models and hyperparameters.
    
    Parameters:
    - X_train: Training features DataFrame
    - y_train: Training target
    - feature_levels: List of taxonomic levels to test (default: all)
    - model_configs: List of dict with 'name', 'model', and 'params' (default: multiple configs)
    - cv_folds: Number of cross-validation folds
    
    Returns:
    - Dictionary with results for each level and model combination
    """
    if feature_levels is None:
        feature_levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    if model_configs is None:
        model_configs = [
            {'name': 'RF_100', 'model': RandomForestRegressor, 'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}},
            {'name': 'RF_200', 'model': RandomForestRegressor, 'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}},
            {'name': 'XGB_100', 'model': xgb.XGBRegressor, 'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}},
            {'name': 'XGB_lr01', 'model': xgb.XGBRegressor, 'params': {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 5, 'random_state': 42, 'n_jobs': -1}},
            {'name': 'GB_100', 'model': GradientBoostingRegressor, 'params': {'n_estimators': 100, 'random_state': 42}},
            {'name': 'LightGBM', 'model': lgb.LGBMRegressor, 'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}},
        ]
    
    all_results = {}
    print(f"Testing multiple models across taxonomic levels with {cv_folds}-fold CV...")
    print(f"{'Level':<12} | {'Model':<15} | {'# Features':<12} | {'Mean RMSE':<12} | {'Mean R2':<10} | {'Std RMSE':<10}")
    print("-" * 90)
    
    for level in feature_levels:
        X_filtered = filter_features_by_level(X_train, max_level=level)
        numeric_cols = X_filtered.select_dtypes(include=[np.number]).columns
        X_numeric = X_filtered[numeric_cols]
        
        if len(X_numeric.columns) == 0:
            print(f"{level:<12} | All models   | No features available")
            continue
        
        all_results[level] = {}
        
        for config in model_configs:
            model = config['model'](**config['params'])
            
            cv_metrics = cross_validate(
                model, X_numeric, y_train,
                cv=cv_folds,
                scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
                n_jobs=-1
            )
            
            rmse_scores = -cv_metrics['test_rmse']
            r2_scores = cv_metrics['test_r2']
            mean_rmse = np.mean(rmse_scores)
            std_rmse = np.std(rmse_scores)
            mean_r2 = np.mean(r2_scores)
            
            all_results[level][config['name']] = {
                'n_features': len(X_numeric.columns),
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'mean_r2': mean_r2,
                'cv_scores': rmse_scores,
                'r2_scores': r2_scores
            }
            
            print(f"{level:<12} | {config['name']:<15} | {len(X_numeric.columns):<12} | {mean_rmse:<12.3f} | {mean_r2:<10.3f} | ±{std_rmse:<10.3f}")
    
    return all_results


# =========================================================
# 7. VISUALIZATION FUNCTIONS FOR MODEL EXPLANATIONS
# =========================================================
def plot_feature_cutoff_comparison(cv_results, title="Model Performance vs Taxonomic Level"):
    """
    Visualize cross-validation results across different taxonomic levels.
    
    Parameters:
    - cv_results: Dictionary from cross_validate_feature_cutoffs
    - title: Plot title
    """
    levels = list(cv_results.keys())
    n_features = [cv_results[level]['n_features'] for level in levels]
    mean_rmse = [cv_results[level]['mean_rmse'] for level in levels]
    std_rmse = [cv_results[level]['std_rmse'] for level in levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE vs Taxonomic Level
    ax1.errorbar(levels, mean_rmse, yerr=std_rmse, marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Taxonomic Level', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title(f'{title}\nRMSE by Taxonomic Level', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Number of Features
    ax2.bar(levels, n_features, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Taxonomic Level', fontsize=12)
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Count by Taxonomic Level', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison_heatmap(results_dict, title="Model Performance Heatmap"):
    """
    Create a heatmap comparing different models and configurations.
    
    Parameters:
    - results_dict: Dictionary with model names as keys and ModelResult objects as values
    - title: Plot title
    """
    # Prepare data
    models = list(results_dict.keys())
    metrics = ['RMSE', 'R²']
    data = np.array([[results_dict[model].rmse, results_dict[model].r2] for model in models])
    
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.5 + 2))
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                xticklabels=metrics, yticklabels=models, 
                cbar_kws={'label': 'Score'}, ax=ax)
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(y_true, y_pred, prediction_std=None, sample_indices=None, 
                              title="Prediction Intervals"):
    """
    Plot predictions with confidence/prediction intervals.
    
    Parameters:
    - y_true: True values
    - y_pred: Predicted values
    - prediction_std: Standard deviation of predictions (optional, for intervals)
    - sample_indices: Specific sample indices to plot (default: all)
    - title: Plot title
    """
    if sample_indices is None:
        sample_indices = np.arange(len(y_true))
    
    y_true_subset = y_true.iloc[sample_indices] if hasattr(y_true, 'iloc') else y_true[sample_indices]
    y_pred_subset = y_pred[sample_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by true values for better visualization
    sort_idx = np.argsort(y_true_subset)
    x = np.arange(len(sort_idx))
    y_true_sorted = y_true_subset.iloc[sort_idx] if hasattr(y_true_subset, 'iloc') else y_true_subset[sort_idx]
    y_pred_sorted = y_pred_subset[sort_idx]
    
    # Plot true values and predictions
    ax.plot(x, y_true_sorted, 'o-', label='True Values', alpha=0.7, markersize=5)
    ax.plot(x, y_pred_sorted, 's-', label='Predictions', alpha=0.7, markersize=5)
    
    # Add prediction intervals if available
    if prediction_std is not None:
        pred_std_sorted = prediction_std[sample_indices][sort_idx]
        ax.fill_between(x, 
                        y_pred_sorted - 1.96 * pred_std_sorted,
                        y_pred_sorted + 1.96 * pred_std_sorted,
                        alpha=0.2, label='95% Prediction Interval')
    
    ax.set_xlabel('Sample (sorted by true value)', fontsize=12)
    ax.set_ylabel('Age Group', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals_analysis(y_true, y_pred, title="Residuals Analysis"):
    """
    Create comprehensive residuals analysis plots.
    
    Parameters:
    - y_true: True values
    - y_pred: Predicted values
    - title: Plot title
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of Residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuals', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scale-Location Plot
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
    axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=11)
    axes[1, 1].set_title('Scale-Location Plot', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Print residual statistics
    print(f"\n📊 Residual Statistics:")
    print(f"  Mean: {np.mean(residuals):.4f}")
    print(f"  Std Dev: {np.std(residuals):.4f}")
    print(f"  Min: {np.min(residuals):.4f}")
    print(f"  Max: {np.max(residuals):.4f}")
    print(f"  MAE: {np.mean(np.abs(residuals)):.4f}")


def plot_learning_curves(model, X_train, y_train, cv_folds=5, title="Learning Curves"):
    """
    Plot learning curves to diagnose bias/variance.
    
    Parameters:
    - model: Scikit-learn compatible model
    - X_train: Training features
    - y_train: Training target
    - cv_folds: Number of cross-validation folds
    - title: Plot title
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    # Convert to positive RMSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training RMSE', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    plt.plot(train_sizes_abs, val_mean, 'o-', label='Validation RMSE', linewidth=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================================================
# 8. PRE FILTERING OF FEATURES BASED ON ABUNDANCE, VARIATION, COLLINEARITY, PREVALENCE
# =========================================================

## Feature Selection Pipeline
def feature_selection_pipeline(X, prevalence_thresh=0.0633342156194934, abundance_thresh=0.00015905606434428443, variance_thresh=5.3173754852092485e-06,
                                   corr_thresh=0.9795715454616797):
    #defaults for genus level:
    #Best parameters: {'prevalence_thresh': 0.0633342156194934, 'abundance_thresh': 0.00015905606434428443, 'variance_thresh': 5.3173754852092485e-06, 'corr_thresh': 0.9795715454616797}
    #Best RMSE: 60.973, Best R2: 0.801

    #defaults for species level:
    #Best parameters: {'prevalence_thresh': 0.0632320223420868, 'abundance_thresh': 1.4626891739832243e-05, 'variance_thresh': 1.9229266155638468e-05, 'corr_thresh': 0.8932026069893284}
    #Best RMSE: 56.463, Best R2: 0.829


    """
    Feature filtering pipeline for microbiome data.

    Parameters:
        X: pd.DataFrame
            Feature matrix (samples x taxa)
        prevalence_thresh: float
            Minimum fraction of samples a feature must appear in
        abundance_thresh: float
            Minimum mean abundance for a feature to be kept
        variance_thresh: float
            Minimum variance threshold
        corr_thresh: float
            Maximum allowed correlation between features

    Returns:
        X_filtered: pd.DataFrame
            Filtered feature matrix
        removed_features: list
            Names of removed features
    """
    X_filtered = X.copy()
    removed_features = []

    # Prevalence filtering
    prevalence = (X_filtered > 0).sum(axis=0) / X_filtered.shape[0]
    low_prev = prevalence[prevalence < prevalence_thresh].index.tolist()
    X_filtered.drop(columns=low_prev, inplace=True)
    removed_features.extend(low_prev)
    print(f"Prevalence filtering: removed {len(low_prev)} features")

    # Mean abundance filtering
    mean_abundance = X_filtered.mean(axis=0)
    low_abundance = mean_abundance[mean_abundance < abundance_thresh].index.tolist()
    X_filtered.drop(columns=low_abundance, inplace=True)
    removed_features.extend(low_abundance)
    print(f"Abundance filtering: removed {len(low_abundance)} features")

    # Low variance filtering
    variance = X_filtered.var(axis=0)
    low_variance = variance[variance < variance_thresh].index.tolist()
    X_filtered.drop(columns=low_variance, inplace=True)
    removed_features.extend(low_variance)
    print(f"Variance filtering: removed {len(low_variance)} features")

    # Collinearity filtering (remove one of each pair of highly correlated features)
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    X_filtered.drop(columns=to_drop, inplace=True)
    removed_features.extend(to_drop)
    print(f"Collinearity filtering: removed {len(to_drop)} features")

    print(f"Remaining features: {X_filtered.shape[1]}")
    return X_filtered, removed_features