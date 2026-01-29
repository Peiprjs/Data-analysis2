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
#import tensorflow as tf
#from tensorflow.keras import layers, models, constraints, callbacks, initializers, regularizers
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate, learning_curve
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

import warnings

# =========================================================
# 1. GLOBAL CONFIG & SYSTEM-AGNOSTIC DEVICE SETUP
# =========================================================

# Suppress the specific Joblib/Parallel propagation warning
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")

# Alternatively, if you want to be broad for this specific module
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

MASTER_SEED = 3004
ModelResult = namedtuple('ModelResult', ['model', 'rmse', 'r2', 'best_params', 'runtime', 'top_features'])
NNResult = namedtuple('NNResult', ['X_train_elite', 'X_test_elite', 'feature_names', 'rmse', 'n_features'])

# CRITICAL FOR ROCm: These must be set BEFORE any TF operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# This helps with the '0 MB memory' error by using the system allocator
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Detect Device
'''gpus = tf.config.list_physical_devices('GPU')
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
'''

def set_global_seeds(seed=MASTER_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Global seeds set to {seed}")

'''
# =========================================================
# 2. NEURAL NETWORK SELECTOR (Robust Version)
# =========================================================
#import tensorflow as tf
#from tensorflow.keras import layers, regularizers, initializers, constraints, models, callbacks
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc


# 1. Helper Class for Results
class NNResult:
    def __init__(self, X_train, X_test, features, rmse, n_features):
        self.X_train = X_train
        self.X_test = X_test
        self.features = features
        self.rmse = rmse
        self.n_features = n_features


# 2. The New Elastic Gatekeeper Layer
class ElasticGatekeeper(layers.Layer):
    def __init__(self, num_features, l1=0.01, l2=0.01, **kwargs):
        super(ElasticGatekeeper, self).__init__(**kwargs)
        self.num_features = num_features
        self.l1 = float(l1)
        self.l2 = float(l2)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="gate_weights",
            shape=(self.num_features,),
            initializer=initializers.Ones(),
            trainable=True,
            constraint=constraints.NonNeg(),  # Keeps weights 0 to 1
            # Elastic Net: L1 for sparsity, L2 for stability
            regularizer=regularizers.L1L2(l1=self.l1, l2=self.l2)
        )

    def call(self, inputs):
        return inputs * self.w


# 3. Main Search Function
def nn_feature_search(
        X_train, X_test, Y_train,
        penalty_pairs=None,
        target_range=(250, 1250),
        consensus_threshold=0.7,
        # Hardware/Speed Parameters
        batch_size=64,
        jit_compile=False,
        mixed_precision=False,
        device="/CPU:0"
):
    # --- Setup ---
    if penalty_pairs is None:
        # Default list of penalties
        penalty_pairs = [
            (35.0, 35.0),
            (37.5, 37.5),
            (40.0, 40.0),
            (42.5, 42.5),
            (45.0, 45.0),
            (47.5, 47.5),
            (50.0, 50.0)
        ]

    base_lr = 0.01 if batch_size >= 4096 else (0.005 if batch_size > 512 else 0.001)
    repeats = 10
    epochs = 400
    patience = 50

    # Global Mixed Precision Setup
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('mixed_float16')
    else:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('float32')

    # Data Prep
    scaler_x = StandardScaler()
    X_train_tf = scaler_x.fit_transform(X_train).astype('float32')
    y_train_tf = Y_train.values.astype('float32')

    penalty_results = []

    # --- Reporting Header ---
    print(f"🧬 Search Started | Device: {device} | Batch: {batch_size}")
    print("-" * 85)
    print(f"{'Penalty (L1/L2)':<18} | {'Features':<10} | {'RMSE (Avg)':<12} | {'R2 (Avg)':<10} | {'Efficiency'}")
    print("-" * 85)

    # --- Search Loop ---
    for l1, l2 in tqdm(penalty_pairs, desc="Testing Elastic Pairs"):
        batch_weights, batch_rmse, batch_r2 = [], [], []
        tf.keras.backend.clear_session()
        gc.collect()

        for i in tqdm(range(repeats), desc=f"L1:{l1}/L2:{l2}", leave=False):
            with tf.device(device):
                inputs = layers.Input(shape=(X_train_tf.shape[1],))

                # Use ElasticGatekeeper
                gate = ElasticGatekeeper(X_train_tf.shape[1], l1=l1, l2=l2)(inputs)

                # Standard Architecture
                x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(gate)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
                outputs = layers.Dense(1, activation='linear', dtype="float32")(x)

                model = models.Model(inputs=inputs, outputs=outputs)
                steps_per_epoch = int(np.ceil(len(X_train_tf) / batch_size))

                # LR Schedule
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.0,
                    decay_steps=epochs * steps_per_epoch,
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

            # Record weights and metrics
            w = model.layers[1].get_weights()[0]
            batch_weights.append(w > 1e-2)

            y_pred = model(X_train_tf, training=False).numpy()
            batch_rmse.append(np.sqrt(mean_squared_error(y_train_tf, y_pred)))
            batch_r2.append(r2_score(y_train_tf, y_pred))

        # --- Aggregate Results ---
        avg_rmse = np.mean(batch_rmse)
        avg_r2 = np.mean(batch_r2)
        selection_frequency = np.mean(batch_weights, axis=0)
        consensus_feats = np.sum(selection_frequency >= consensus_threshold)

        # Efficiency Calculation (Kept for logging, ignored for selection)
        eff = avg_r2 / np.log1p(consensus_feats) if consensus_feats > 0 else 0

        # Print Row
        p_str = f"{l1}/{l2}"
        print(f"{p_str:<18} | {consensus_feats:<10} | {avg_rmse:<12.2f} | {avg_r2:<10.2f} | {eff:.4f}")

        penalty_results.append({
            'l1': l1, 'l2': l2,
            'n_features': consensus_feats,
            'rmse': avg_rmse,
            'rmse_list': batch_rmse,  # <--- CRITICAL: Store raw list for SE calc
            'r2': avg_r2,
            'efficiency': eff,
            'freq_mask': selection_frequency
        })

    # --- SCIENTIFIC CHAMPION SELECTION (1-SE Rule) ---

    # 1. Strict Range Filter
    valid_results = [r for r in penalty_results if target_range[0] < r['n_features'] < target_range[1]]

    if not valid_results:
        print("\nNo penalties fell within target feature range. Defaulting to Best RMSE overall.")
        champion = min(penalty_results, key=lambda x: x['rmse'])
    else:
        # 2. Find the "Gold Standard" (Model with absolute lowest RMSE)
        best_run = min(valid_results, key=lambda x: x['rmse'])
        best_rmse = best_run['rmse']

        # 3. Calculate Standard Error (SE)
        # Formula: SE = StdDev / sqrt(N_repeats)
        rmse_std = np.std(best_run['rmse_list'])
        n_repeats = len(best_run['rmse_list'])
        standard_error = rmse_std / np.sqrt(n_repeats)

        # Safety fallback if SE is 0 (shouldn't happen with 5 repeats)
        if standard_error == 0: standard_error = 0.1

        # 4. Define Threshold (Best RMSE + 1 Standard Error)
        rmse_threshold = best_rmse + standard_error

        # 5. Filter Candidates (Statistically Indistinguishable)
        candidates = [r for r in valid_results if r['rmse'] <= rmse_threshold]

        # 6. Select Winner (Parsimony Principle: The simplest model wins)
        champion = min(candidates, key=lambda x: x['n_features'])

        # --- Detailed Report ---
        print(f"\nScientific Selection (1-Standard-Error Rule):")
        print(f"Gold Standard RMSE: {best_rmse:.2f} ± {standard_error:.2f} (SE)")
        print(f"Selection Threshold: <= {rmse_threshold:.2f} (Best + 1 SE)")
        print(f"Candidates in Statistical Range: {len(candidates)}")

        if champion != best_run:
            diff = best_run['n_features'] - champion['n_features']
            print(f"DECISION: Selected simpler model. Saved {diff} features within error margin.")
        else:
            print(f"DECISION: The most accurate model was selected (no simpler model was within 1 SE).")

    elite_mask = champion['freq_mask'] >= consensus_threshold
    elite_names = X_train.columns[elite_mask].tolist()

    print(f"\n🏆 Champion Selected: L1={champion['l1']}, L2={champion['l2']}")
    print(f"   Features: {champion['n_features']}")
    print(f"   RMSE: {champion['rmse']:.2f}")

    return NNResult(X_train[elite_names], X_test[elite_names], elite_names, champion['rmse'], champion['n_features'])
'''
# =========================================================
# 3. CLASSICAL ML BENCHMARKS (XGB & RF)
# =========================================================

def run_model_benchmark(estimator, param_distributions, X_train_df, X_test_df, y_train, y_test, 
                        label="Dataset", n_iter=20, cv=5, task='regression', clean_columns=False):
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
        random_state=MASTER_SEED,
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
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=MASTER_SEED, n_jobs=-1),
        param_distributions={"n_estimators": [500, 1000], "learning_rate": [0.01, 0.05], "max_depth": [3, 4, 5],
                             "subsample": [0.7, 0.8], "colsample_bytree": [0.1, 0.2], "reg_alpha": [0.1, 0.5, 1.0],
                             "reg_lambda": [1.0, 5.0]},
        n_iter=13, cv=7, scoring="neg_root_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
    )

    start_time = time.time()
    search_xgb.fit(X_train_clean, y_train)
    elapsed = time.time() - start_time

    best_model = search_xgb.best_estimator_
    y_pred = best_model.predict(X_test_clean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feat_df = pd.DataFrame({'Bacteria': X_train_df.columns, 'Importance': best_model.feature_importances_})
    top_drivers = feat_df.sort_values('Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n{label} Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search_xgb.best_params_, elapsed, top_drivers)



def random_forest_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset", cv=7, n_iter=13):
    print(f"Initializing Random Forest Engine: {label}")
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=MASTER_SEED, n_jobs=-1),
        param_distributions={"n_estimators": [100, 300, 500, 800], "max_depth": [None, 10, 20, 40],
                             "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
                             "max_features": [1.0, "sqrt", "log2"]},
        n_iter=n_iter, cv=cv, scoring="neg_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=AdaBoostRegressor(random_state=MASTER_SEED),
        param_distributions={
            "n_estimators": [50, 100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "loss": ["linear", "square", "exponential"]
        },
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=GradientBoostingRegressor(random_state=MASTER_SEED),
        param_distributions={
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6, 10, 15, 20, 40],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 2, 4, 10, 50],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", None]
        },
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=lgb.LGBMRegressor(random_state=MASTER_SEED, device = "cpu", n_jobs=1, verbose=-1, importance_type="gain"),
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
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=GradientBoostingRegressor(random_state=MASTER_SEED),
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
        n_iter=20, cv=5, scoring="neg_mean_squared_error", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=SVC(random_state=MASTER_SEED),
        param_distributions={
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=LogisticRegression(random_state=MASTER_SEED, max_iter=1000),
        param_distributions={
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet", None],
            "solver": ["lbfgs", "saga", "liblinear"],
            "class_weight": [None, "balanced"]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=RandomForestClassifier(random_state=MASTER_SEED, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": [None, "balanced", "balanced_subsample"]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        estimator=GradientBoostingClassifier(random_state=MASTER_SEED),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2", 0.5]
        },
        n_iter=20, cv=5, scoring="accuracy", random_state=MASTER_SEED, n_jobs=-1, verbose=1
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
        kernel_approx = Nystroem(kernel='rbf', random_state=MASTER_SEED)
    elif kernel_method == 'rbf':
        kernel_approx = RBFSampler(random_state=MASTER_SEED)
    else:
        raise ValueError(f"Unknown kernel_method: {kernel_method}. Use 'nystroem' or 'rbf'")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_approx', kernel_approx),
        ('rf', RandomForestRegressor(random_state=MASTER_SEED, n_jobs=-1))
    ])
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=MASTER_SEED,
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
        kernel_approx = Nystroem(kernel='rbf', random_state=MASTER_SEED)
    elif kernel_method == 'rbf':
        kernel_approx = RBFSampler(random_state=MASTER_SEED)
    else:
        raise ValueError(f"Unknown kernel_method: {kernel_method}. Use 'nystroem' or 'rbf'")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kernel_approx', kernel_approx),
        ('rf', RandomForestClassifier(random_state=MASTER_SEED, n_jobs=-1))
    ])
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=MASTER_SEED,
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
def final_battle(datasets_dict, y_train, n_splits=5, n_repeats=5,
                 xgb_params=None, rf_params=None, gb_params=None,
                 lgbm_params=None, ada_params=None, sgb_params=None,
                 krf_params=None):
    from sklearn.model_selection import RepeatedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.kernel_approximation import Nystroem
    import re

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=MASTER_SEED)

    def prepare_params(defaults, best):
        if best is None: return defaults
        combined = {**defaults, **best}
        # Remove metadata keys that might be in the dict
        for key in ['mean_test_score', 'std_test_score', 'rank_test_score']:
            combined.pop(key, None)
        return combined

    # --- Configuration ---
    default_xgb = {'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 3, 'n_jobs': 4,
                   'random_state': MASTER_SEED}
    default_rf = {'n_estimators': 1000, 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': MASTER_SEED}
    default_gb = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4, 'random_state': MASTER_SEED}
    default_lgbm = {'n_estimators': 500, 'learning_rate': 0.05, 'n_jobs': 1, 'random_state': MASTER_SEED, 'verbose': -1}
    default_ada = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': MASTER_SEED}
    default_sgb = {**default_gb, 'subsample': 0.8}

    # FIX: Initialize Kernel RF Pipeline properly
    krf_pipe = Pipeline([
        ('kernel', Nystroem(random_state=MASTER_SEED)),
        ('rf', RandomForestRegressor(**default_rf))
    ])

    # Use set_params to handle the 'rf__' prefixes automatically
    if krf_params:
        valid_krf_params = {k: v for k, v in krf_params.items()
                            if any(k.startswith(p) for p in ['rf__', 'kernel__'])}
        krf_pipe.set_params(**valid_krf_params)

    models_to_test = {
        "XGBoost": xgb.XGBRegressor(**prepare_params(default_xgb, xgb_params)),
        "Random Forest": RandomForestRegressor(**prepare_params(default_rf, rf_params)),
        "Gradient Boosting": GradientBoostingRegressor(**prepare_params(default_gb, gb_params)),
        "LightGBM": lgb.LGBMRegressor(**prepare_params(default_lgbm, lgbm_params)),
        "AdaBoost": AdaBoostRegressor(**prepare_params(default_ada, ada_params)),
        "Stochastic GB": GradientBoostingRegressor(**prepare_params(default_sgb, sgb_params)),
        "Kernel RF": krf_pipe  # Use the configured pipeline here
    }

    battle_results = []
    print(f"Starting Battle Arena ({n_splits}x{n_repeats} = {n_splits * n_repeats} runs)")
    print(f"{'Dataset':<20} | {'Model':<18} | {'Avg RMSE':<12} | {'Avg R2':<10} | {'Std Dev':<10}")
    print("-" * 90)

    for data_name, X_data in datasets_dict.items():
        X_clean = X_data.copy()
        X_clean.columns = [re.sub('[^A-Za-z0-9_]+', '', str(col)) for col in X_clean.columns]

        for model_name, model_obj in models_to_test.items():
            cv_metrics = cross_validate(
                model_obj, X_clean, y_train,
                cv=rkf,
                scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
                n_jobs=-1
            )

            mean_rmse = -np.mean(cv_metrics['test_rmse'])
            std_rmse = np.std(cv_metrics['test_rmse'])
            mean_r2 = np.mean(cv_metrics['test_r2'])

            print(f"{data_name:<20} | {model_name:<18} | {mean_rmse:.2f} days   | {mean_r2:.3f}    | ±{std_rmse:.2f}")

            battle_results.append({
                'Dataset': data_name, 'Model': model_name,
                'RMSE_Mean': mean_rmse, 'RMSE_Std': std_rmse, 'R2_Mean': mean_r2
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
        random_state=MASTER_SEED
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
    - Tuple of (results_dict, results_df):
        * results_dict: Dictionary with results for each level and model combination
        * results_df: DataFrame with columns [Level, Model, Num_Features, Mean_RMSE, Std_RMSE, Mean_R2]
    """
    if feature_levels is None:
        feature_levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    if model_configs is None:
        model_configs = [
            {'name': 'RF_100', 'model': RandomForestRegressor, 'params': {'n_estimators': 100, 'random_state': MASTER_SEED, 'n_jobs': -1}},
            {'name': 'RF_200', 'model': RandomForestRegressor, 'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': MASTER_SEED, 'n_jobs': -1}},
            {'name': 'XGB_100', 'model': xgb.XGBRegressor, 'params': {'n_estimators': 100, 'random_state': MASTER_SEED, 'n_jobs': -1}},
            {'name': 'XGB_lr01', 'model': xgb.XGBRegressor, 'params': {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 5, 'random_state': MASTER_SEED, 'n_jobs': -1}},
            {'name': 'GB_100', 'model': GradientBoostingRegressor, 'params': {'n_estimators': 100, 'random_state': MASTER_SEED}},
            {'name': 'LightGBM', 'model': lgb.LGBMRegressor, 'params': {'n_estimators': 100, 'random_state': MASTER_SEED, 'n_jobs': 1, 'verbose': -1}},
        ]
    
    all_results = {}
    df_rows = []
    print(f"Testing multiple models across taxonomic levels with {cv_folds}-fold CV...")
    print(f"{'Level':<12} | {'Model':<15} | {'# Features':<12} | {'Mean RMSE':<12} | {'Mean R2':<10} | {'Std RMSE':<10}")
    print("-" * 90)
    
    total_combinations = len(feature_levels) * len(model_configs)
    pbar = tqdm(total=total_combinations, desc="Cross-validation progress")
    
    try:
        for level in feature_levels:
            X_filtered = filter_features_by_level(X_train, max_level=level)
            numeric_cols = X_filtered.select_dtypes(include=[np.number]).columns
            X_numeric = X_filtered[numeric_cols]

            if len(X_numeric.columns) == 0:
                print(f"{level:<12} | All models   | No features available")
                pbar.update(len(model_configs))
                continue

            all_results[level] = {}

            for config in model_configs:
                model = config['model'](**config['params'])

                pbar.set_description(f"CV: {level} - {config['name']}")

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
                std_r2 = np.std(r2_scores)

                all_results[level][config['name']] = {
                    'n_features': len(X_numeric.columns),
                    'mean_rmse': mean_rmse,
                    'std_rmse': std_rmse,
                    'mean_r2': mean_r2,
                    'cv_scores': rmse_scores,
                    'r2_scores': r2_scores
                }

                df_rows.append({
                    'Level': level,
                    'Model': config['name'],
                    'Num_Features': len(X_numeric.columns),
                    'Mean_RMSE': mean_rmse,
                    'Std_RMSE': std_rmse,
                    'Mean_R2': mean_r2,
                    'Std_R2': std_r2
                })

                print(f"{level:<12} | {config['name']:<15} | {len(X_numeric.columns):<12} | {mean_rmse:<12.3f} | {mean_r2:<10.3f} | ±{std_rmse:<10.3f}")
                pbar.update(1)
    finally:
        pbar.close()

    results_df = pd.DataFrame(df_rows)
    return all_results, results_df


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


def plot_model_comparison_heatmap(results_list, title="Final Model Performance Comparison"):
    """
    Creates a heatmap from a list of dictionaries.
    Handles 'model' and 'label' to create unique row identifiers.
    """
    # 1. Convert the list of dicts directly to a DataFrame
    df = pd.DataFrame(results_list)

    # 2. Combine model and label for the Y-axis (e.g., "XGBoost (Species Level)")
    # This prevents rows from overlapping if you use the same model on different sets
    df['Model_Setup'] = df['model'] + " - " + df['label']

    # 3. Prepare data for the heatmap
    # We set our new 'Model_Setup' as the index and select only metrics
    plot_df = df.set_index('Model_Setup')[['rmse', 'r2']]

    # 4. Create the plot
    # Height scales with the number of models to keep it readable
    plt.figure(figsize=(10, len(results_list) * 0.7 + 2))

    # Use 'coolwarm' or 'YlGnBu' for clear differentiation
    sns.heatmap(plot_df,
                annot=True,
                fmt='.4f',
                cmap='YlGnBu',
                linewidths=1,
                linecolor='white')

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Model & Configuration", fontsize=12)
    plt.xlabel("Evaluation Metrics", fontsize=12)

    # Move X-axis labels to the top for a professional "Table" feel
    plt.tick_params(axis='both', which='major', labelsize=10,
                    labelbottom=False, bottom=False, top=False, labeltop=True)

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
    
    print(f"\nResidual Statistics:")
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
        random_state=MASTER_SEED
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
def tune_filtering_parameters(X_train, Y_train, n_iterations=20):
    """
    Faster Optimization: Uses 5-Fold CV and Random Forest without repetitions.
    """
    current_params = {
        "prevalence_thresh": 0.05, "abundance_thresh": 1e-4,
        "variance_thresh": 1e-5, "corr_thresh": 0.9
    }
    param_ranges = {
        "prevalence_thresh": 0.01, "abundance_thresh": 5e-5,
        "variance_thresh": 5e-6, "corr_thresh": 0.05
    }

    best_avg_rmse = np.inf

    kf = KFold(n_splits=5, shuffle=True, random_state=MASTER_SEED)

    for i in range(n_iterations):
        trial_params = {
            k: max(1e-9, current_params[k] + np.random.uniform(-param_ranges[k], param_ranges[k]))
            for k in current_params
        }
        if trial_params["corr_thresh"] >= 1.0: trial_params["corr_thresh"] = 0.99

        fold_rmses = []

        for train_idx, val_idx in kf.split(X_train):
            X_itrain, X_ival = X_train.iloc[train_idx], X_train.iloc[val_idx]
            Y_itrain, Y_ival = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

            # Internal filter
            X_trial_train, _ = feature_selection_pipeline(
                X_itrain, Y_itrain, optimize=False, verbose=False, **trial_params
            )

            # Efficiency check: if no features left, this trial is a fail
            if X_trial_train.shape[1] == 0:
                fold_rmses.append(np.inf)
                break

            # Evaluate with the ACTUAL model you use later
            rf = RandomForestRegressor(n_estimators=50, random_state=MASTER_SEED, n_jobs=-1)
            rf.fit(X_trial_train, Y_itrain)

            X_trial_val = X_ival[X_trial_train.columns]
            rmse = np.sqrt(mean_squared_error(Y_ival, rf.predict(X_trial_val)))
            fold_rmses.append(rmse)

        avg_rmse = np.mean(fold_rmses)
        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            current_params = trial_params.copy()

    return current_params

def feature_selection_pipeline(X_train, Y_train, optimize=True, n_iterations=30, verbose=True, **kwargs):
    """
    Main Pipeline: Returns (Filtered_DF, Removed_Features_List)
    """
    if optimize:
        if verbose: print(f"--- Running K-Fold Optimization ({n_iterations} iterations) ---")
        best_params = tune_filtering_parameters(X_train, Y_train, n_iterations)
    else:
        best_params = {
            "prevalence_thresh": kwargs.get("prevalence_thresh", 0.05),
            "abundance_thresh": kwargs.get("abundance_thresh", 1e-4),
            "variance_thresh": kwargs.get("variance_thresh", 1e-5),
            "corr_thresh": kwargs.get("corr_thresh", 0.95)
        }

    # FILTERING LOGIC
    X_filtered = X_train.copy()
    initial_cols = set(X_filtered.columns)

    # 1. Prevalence
    prev = (X_filtered > 0).sum(axis=0) / len(X_filtered)
    X_filtered.drop(columns=prev[prev < best_params["prevalence_thresh"]].index, inplace=True)

    # 2. Abundance
    abun = X_filtered.mean(axis=0)
    X_filtered.drop(columns=abun[abun < best_params["abundance_thresh"]].index, inplace=True)

    # 3. Variance
    var = X_filtered.var(axis=0)
    X_filtered.drop(columns=var[var < best_params["variance_thresh"]].index, inplace=True)

    # 4. Correlation
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > best_params["corr_thresh"])]
    X_filtered.drop(columns=to_drop, inplace=True)

    removed_features = list(initial_cols - set(X_filtered.columns))

    if verbose:
        print(f"Done. Kept: {X_filtered.shape[1]} | Removed: {len(removed_features)}")
        print(f"Best Filtering Parameters Found: {best_params}")

    return X_filtered, removed_features


def pca_feature_selection(X, n_components=0.95, scale=True):
    """
    Apply PCA for dimensionality reduction.
    
    Parameters:
    - X: Feature matrix (DataFrame or array)
    - n_components: Number of components or variance ratio to retain
    - scale: Whether to scale features before PCA
    
    Returns:
    - X_pca: Transformed features
    - pca: Fitted PCA object
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    if scale:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)
    
    pca = PCA(n_components=n_components, random_state=MASTER_SEED)
    X_pca = pca.fit_transform(X_arr)
    
    print(f"PCA: reduced to {X_pca.shape[1]} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_pca, pca


def feature_importance_selection(X_train, y_train, X_test=None, n_features=100, model_type='RandomForest'):
    """
    Select features based on model feature importance.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features (optional)
    - n_features: Number of top features to select
    - model_type: Model to use for importance ('RandomForest', 'XGBoost', 'GradientBoosting')
    
    Returns:
    - X_train_selected: Selected training features
    - X_test_selected: Selected test features (if X_test provided)
    - selected_features: List of selected feature names
    """
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=MASTER_SEED, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, random_state=MASTER_SEED, n_jobs=-1)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=MASTER_SEED)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f"feature_{i}" for i in range(X_train_arr.shape[1])]
    
    model.fit(X_train_arr, y_train)
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1][:n_features]
    selected_features = [feature_names[i] for i in indices]
    
    if isinstance(X_train, pd.DataFrame):
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features] if X_test is not None else None
    else:
        X_train_selected = X_train_arr[:, indices]
        X_test_selected = X_test[:, indices] if X_test is not None else None
    
    print(f"Feature importance selection: selected {len(selected_features)} features using {model_type}")
    
    if X_test is not None:
        return X_train_selected, X_test_selected, selected_features
    return X_train_selected, selected_features


def anova_feature_selection(X_train, y_train, X_test=None, n_features=100, mode='regression'):
    """
    Select features based on ANOVA F-statistic.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features (optional)
    - n_features: Number of top features to select
    - mode: 'regression' or 'classification'
    
    Returns:
    - X_train_selected: Selected training features
    - X_test_selected: Selected test features (if X_test provided)
    - selected_features: List of selected feature names
    """
    score_func = f_regression if mode == 'regression' else f_classif
    selector = SelectKBest(score_func=score_func, k=min(n_features, X_train.shape[1]))
    
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f"feature_{i}" for i in range(X_train_arr.shape[1])]
    
    X_train_selected = selector.fit_transform(X_train_arr, y_train)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    if X_test is not None:
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        X_test_selected = selector.transform(X_test_arr)
    else:
        X_test_selected = None
    
    print(f"ANOVA feature selection: selected {len(selected_features)} features for {mode}")
    
    if isinstance(X_train, pd.DataFrame):
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        if X_test_selected is not None:
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    if X_test is not None:
        return X_train_selected, X_test_selected, selected_features
    return X_train_selected, selected_features


def compare_feature_selection_methods(X_train, y_train, X_test, y_test, methods=['importance', 'anova', 'pca'],
                                       n_features=100, model_for_eval='RandomForest'):
    """
    Compare different feature selection methods.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - methods: List of methods to compare ('importance', 'anova', 'pca', 'baseline')
    - n_features: Number of features to select
    - model_for_eval: Model to use for evaluation

    Returns:
    - results: Dictionary with performance metrics for each method
    """
    results = {}
    n_list = n_features if isinstance(n_features, list) else [n_features]

    if model_for_eval == 'RandomForest':
        eval_model = RandomForestRegressor(n_estimators=100, random_state=MASTER_SEED, n_jobs=-1)
    elif model_for_eval == 'XGBoost':
        eval_model = xgb.XGBRegressor(n_estimators=100, random_state=MASTER_SEED, n_jobs=-1)
    elif model_for_eval == 'GradientBoosting':
        eval_model = GradientBoostingRegressor(n_estimators=100, random_state=MASTER_SEED)
    else:
        eval_model = RandomForestRegressor(n_estimators=100, random_state=MASTER_SEED, n_jobs=-1)

    print(f"Comparing feature selection methods using {model_for_eval}...")
    print(f"{'Method':<20} | {'# Features':<12} | {'RMSE':<12} | {'R2':<10}")
    print("-" * 60)

    pbar = tqdm(total=len(methods), desc="Feature selection methods")
    for n in n_list:
        if 'baseline' in methods and 'baseline' not in results:
            pbar.set_description("Baseline (no selection)")
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)
            results['baseline'] = {
                'n_features': X_train.shape[1],
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            print(f"{'Baseline':<20} | {X_train.shape[1]:<12} | {results['baseline']['rmse']:<12.3f} | {results['baseline']['r2']:<10.3f}")
            pbar.update(1)

        if 'importance' in methods:
            if 'importance' not in results: results['importance'] = {}
            pbar.set_description(f"Importance (n={n})")
            X_train_sel, X_test_sel, _ = feature_importance_selection(X_train, y_train, X_test, n, 'RandomForest')
            eval_model.fit(X_train_sel, y_train)
            y_pred = eval_model.predict(X_test_sel)
            results['importance'][n] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            print(f"{f'Importance (n={n})':<20} | {n:<12} | {results['importance'][n]['rmse']:<12.3f} | {results['importance'][n]['r2']:<10.3f}")
            pbar.update(1)

        if 'anova' in methods:
            if 'anova' not in results: results['anova'] = {}
            pbar.set_description(f"ANOVA F-value (n={n})")
            X_train_sel, X_test_sel, _ = anova_feature_selection(X_train, y_train, X_test, n, 'regression')
            eval_model.fit(X_train_sel, y_train)
            y_pred = eval_model.predict(X_test_sel)
            results['anova'][n] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            print(f"{f'ANOVA (n={n})':<20} | {n:<12} | {results['anova'][n]['rmse']:<12.3f} | {results['anova'][n]['r2']:<10.3f}")
            pbar.update(1)

        if 'pca' in methods:
            if 'pca' not in results: results['pca'] = {}
            pbar.set_description(f"PCA (n={n})")
            X_train_pca, pca_obj = pca_feature_selection(X_train, n_components=n if n < X_train.shape[1] else 0.95)
            X_test_pca = pca_obj.transform(X_test.values)
            eval_model.fit(X_train_pca, y_train)
            y_pred = eval_model.predict(X_test_pca)
            results['pca'][n] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            print(f"{f'PCA (n={n})':<20} | {X_train_pca.shape[1]:<12} | {results['pca'][n]['rmse']:<12.3f} | {results['pca'][n]['r2']:<10.3f}")
            pbar.update(1)

    pbar.close()
    return results


def find_best_evaluation_metric(y_true, y_pred, task='regression'):
    """
    Calculate and compare different evaluation metrics.
    
    Parameters:
    - y_true: True values
    - y_pred: Predicted values
    - task: 'regression' or 'classification'
    
    Returns:
    - metrics: Dictionary of metric values
    """
    metrics = {}
    
    if task == 'regression':
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print("Regression Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    elif task == 'classification':
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted')
        try:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
        except (ValueError, TypeError):
            metrics['ROC_AUC'] = None
        
        print("Classification Metrics:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
    
    return metrics

# =========================================================
# 9. CLR Transformation
# =========================================================

"""The CLR function based on: https://medium.com/@nextgendatascientist/a-guide-for-data-scientists-log-ratio-transformations-in-machine-learning-a2db44e2a455"""

def clr_transform(X, epsilon=1e-6):
    """
    Compute CLR with a tunable zero-replacement value (epsilon).
    """

    #To capture metadata from the original dataframe
    if isinstance(X, pd.DataFrame):
        index = X.index
        columns = X.columns
        X_arr = X.values.astype(float)
    else:
        X_arr = np.array(X).astype(float)

    # 1. Replace zeros with epsilon (tunable parameter)
    X_replaced = np.where(X_arr == 0, epsilon, X_arr)

    # 2. Compute Geometric Mean
    # exp(mean(log)) is safer and standard for this
    gm = np.exp(np.log(X_replaced).mean(axis=1, keepdims=True))

    # 3. CLR transformation
    X_clr = np.log(X_replaced / gm)


    #Rebulding back a NumPy array to a dataframe
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(X_clr, index=index, columns=columns)

    return X_clr

