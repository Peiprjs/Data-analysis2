import pandas as pd
import numpy as np
import time
import re
import random
import os
import gc
from collections import namedtuple

# Machine Learning & Stats
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, constraints, callbacks, initializers, regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
        print(f"🚀 GPU Active: {gpus[0].name}")
    except Exception as e:
        print(f"⚠️ GPU Initialization Failed: {e}. Switching to CPU for stability.")
        DEVICE = '/CPU:0'
else:
    DEVICE = '/CPU:0'
    print("💻 No GPU found. Running on CPU mode.")


def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✅ Global seeds set to {seed}")


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


def nn_feature_search(X_train, X_test, Y_train, target_range=(50, 1250), consensus_threshold=0.7):
    scaler_x = StandardScaler()
    X_train_tf = scaler_x.fit_transform(X_train).astype('float32')
    y_train_tf = Y_train.values.astype('float32')

    # Your requested penalty range
    penalties = [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    repeats = 10

    # Storage for stability selection
    penalty_results = []

    print(f"🔬 Starting Scientific Stability Search on {X_train.shape[1]} features...")
    print(f"{'Penalty':<8} | {'Avg Feats':<10} | {'Avg RMSE':<10} | {'Avg R2':<8} | {'Stability'}")
    print("-" * 75)

    for p in penalties:
        batch_weights = []
        batch_rmse = []
        batch_r2 = []

        for i in range(repeats):
            tf.keras.backend.clear_session()
            gc.collect()

            with tf.device(DEVICE):
                inputs = layers.Input(shape=(X_train_tf.shape[1],))
                gate = GatekeeperLayer(X_train_tf.shape[1], l1_penalty=p)(inputs)
                x = layers.Dense(128, activation='relu')(gate)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(64, activation='relu')(x)
                outputs = layers.Dense(1, activation='linear')(x)

                model = models.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

                model.fit(
                    X_train_tf, y_train_tf,
                    epochs=400, batch_size=64, validation_split=0.2,
                    verbose=0, callbacks=[callbacks.EarlyStopping(patience=40, restore_best_weights=True)]
                )

            # Record weights and metrics
            w = model.layers[1].get_weights()[0]
            batch_weights.append(w > 1e-2)  # Binary mask of selected features

            y_pred = model.predict(X_train_tf, verbose=0)
            batch_rmse.append(np.sqrt(mean_squared_error(y_train_tf, y_pred)))
            batch_r2.append(r2_score(y_train_tf, y_pred))

        # Calculate Consensus for this penalty
        # How many times was each feature selected?
        selection_frequency = np.mean(batch_weights, axis=0)
        consensus_feats = np.sum(selection_frequency >= consensus_threshold)

        avg_rmse = np.mean(batch_rmse)
        avg_r2 = np.mean(batch_r2)

        print(
            f"{p:<8} | {consensus_feats:<10} | {avg_rmse:<10.2f} | {avg_r2:<8.2f} | {consensus_threshold * 100:.0f}% Match")

        penalty_results.append({
            'penalty': p,
            'n_features': consensus_feats,
            'rmse': avg_rmse,
            'r2': avg_r2,
            'freq_mask': selection_frequency
        })

    # Find the "Scientific Champion"
    # Criteria: Within target range, then highest R2

    for res in penalty_results:
        # We want a balance: High R2, Low RMSE, and manageable feature count
        # This prevents the model from just picking the highest penalty
        res['efficiency'] = res['r2'] / np.log1p(res['n_features'])

        # Find the "Scientific Sweet Spot"
        # We look for the penalty that maximizes Efficiency within the target range
    valid_results = [r for r in penalty_results if target_range[0] < r['n_features'] < target_range[1]]

    if not valid_results:
        print("🛑 No penalty level met the consensus target range.")
        return None

    # CHOOSE THE SWEET SPOT (Not just the highest penalty)
    champion = max(valid_results, key=lambda x: x['efficiency'])

    print(f"\n🎯 Sweet Spot Found at Penalty {champion['penalty']}")
    print(f"Features: {champion['n_features']} | R2: {champion['r2']:.3f} | RMSE: {champion['rmse']:.2f}")

    elite_mask = champion['freq_mask'] >= consensus_threshold
    elite_names = X_train.columns[elite_mask].tolist()

    return NNResult(X_train[elite_names], X_test[elite_names], elite_names, champion['rmse'], champion['n_features'])

# =========================================================
# 3. CLASSICAL ML BENCHMARKS (XGB & RF)
# =========================================================
def xgboost_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"🚀 Initializing XGBoost Engine: {label}")
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

    print(f"\n✅ {label} Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search_xgb.best_params_, elapsed, top_drivers)


def random_forest_benchmark(X_train_df, X_test_df, y_train, y_test, label="Dataset"):
    print(f"🌲 Initializing Random Forest Engine: {label}")
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

    print(f"\n✅ {label} RF Complete ({elapsed:.1f}s) | R2: {r2:.3f}")

    return ModelResult(best_model, rmse, r2, search.best_params_, elapsed, top_drivers)


# =========================================================
# 4. REPEATED CV BATTLE (5x5 Arena)
# =========================================================
def final_battle(datasets_dict, y_train, n_splits=5, n_repeats=5, xgb_params=None, rf_params=None):
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

    current_xgb_params = xgb_params if xgb_params else default_xgb
    current_rf_params = rf_params if rf_params else default_rf

    models_to_test = {
        "XGBoost": xgb.XGBRegressor(**current_xgb_params),
        "Random Forest": RandomForestRegressor(**current_rf_params)
    }

    battle_results = []
    print(f"⚔️ Starting Battle Arena ({n_splits}x{n_repeats} = {n_splits * n_repeats} runs)")
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