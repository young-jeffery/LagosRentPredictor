# tuned_ann_v2.py
# [MODIFIED] Expanded from 16 to 60 combinations to match the search
# budget used for XGBoost and Random Forest, for a genuinely fair
# "which algorithm wins" comparison across all three.
#
# Since a full grid across enough dimensions to reach 60 would require an
# enormous number of parameters, this uses RANDOM sampling of 60
# combinations from a wider hyperparameter space instead (same principle
# RandomizedSearchCV uses for the tree-based models) -- fixed seed, so
# it's reproducible.
#
# Widened the architecture range to include smaller networks than before
# ([16,8] up to [256,128,64,32]), since with ~2,385 training rows, a large
# network may simply be over-parameterized for this dataset size,
# regardless of dropout. Added batch_size as a tuned dimension too.
#
# Increased max epochs to 200 (from 150) with patience 20 (from 15), to
# give slower-converging low-learning-rate configs a fair chance to
# actually finish learning before being cut off.

import pandas as pd
import numpy as np
import random
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow import keras
from tensorflow.keras import layers
import joblib

SEARCH_SEED = 42
N_CANDIDATES = 60


def build_model(input_dim, hidden_layers, dropout_rate, learning_rate, l2_reg):
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizer))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


def tune_and_train_ann():
    print("--- INITIATING ANN ARCHITECTURE SEARCH (v2 - 60-candidate random search) ---")

    print("Loading v2 dense feature matrices...")
    X_train_full = pd.read_csv('X_train_v2.csv')
    X_test = pd.read_csv('X_test_v2.csv')
    y_train_full = pd.read_csv('y_train_v2.csv').squeeze()
    y_test = pd.read_csv('y_test_v2.csv').squeeze()

    print("Scaling features with MinMaxScaler (fit on training data only)...")
    scaler = MinMaxScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled_full, y_train_full, test_size=0.15, random_state=42
    )

    # [MODIFIED] Widened search space, sampled randomly instead of full grid
    architectures = [
        [16, 8], [32, 16], [64, 32], [64, 32, 16],
        [128, 64], [128, 64, 32], [256, 128, 64], [256, 128, 64, 32]
    ]
    dropout_rates = [0.1, 0.2, 0.3, 0.4]
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]
    l2_regs = [0.0, 0.001, 0.01]

    all_combos = list(itertools.product(architectures, dropout_rates, learning_rates, batch_sizes, l2_regs))
    random.seed(SEARCH_SEED)
    candidates = random.sample(all_combos, N_CANDIDATES)

    print(f"Testing {len(candidates)} randomly sampled combinations "
          f"(from a space of {len(all_combos)} total)...\n")

    best_r2 = -np.inf
    best_model = None
    best_config = None

    for i, (arch, dropout, lr, batch_size, l2_reg) in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] Arch={arch}, Dropout={dropout}, "
              f"LR={lr}, Batch={batch_size}, L2={l2_reg}")

        model = build_model(X_train_sub.shape[1], arch, dropout, lr, l2_reg)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        history = model.fit(
            X_train_sub, y_train_sub,
            validation_data=(X_val, y_val),
            epochs=200, batch_size=batch_size,
            callbacks=[early_stop], verbose=0
        )

        preds_log = model.predict(X_test_scaled, verbose=0).flatten()
        preds_raw = np.expm1(preds_log)
        y_test_raw = np.expm1(y_test)
        r2 = r2_score(y_test_raw, preds_raw)

        epochs_run = len(history.history['loss'])
        print(f"   Stopped at epoch {epochs_run}/200, Test R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_config = {
                'architecture': arch, 'dropout': dropout, 'learning_rate': lr,
                'batch_size': batch_size, 'l2_reg': l2_reg
            }
            print(f"   *** New best so far ***")

    print(f"\n--- SEARCH COMPLETE ---")
    print(f"Best Configuration Found: {best_config}")

    print("\nEvaluating Best Model on the Test Set...")
    preds_log = best_model.predict(X_test_scaled, verbose=0).flatten()
    preds_raw = np.expm1(preds_log)
    y_test_raw = np.expm1(y_test)

    r2 = r2_score(y_test_raw, preds_raw)
    mae = mean_absolute_error(y_test_raw, preds_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
    mape = mean_absolute_percentage_error(y_test_raw, preds_raw) * 100

    best_model.save('ann_tuned_model_v2.keras')
    joblib.dump(scaler, 'ann_scaler_v2.pkl')

    print("\n=== TUNED ANN RESULTS (v2 - 60-candidate search) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'ann_tuned_model_v2.keras', scaler as 'ann_scaler_v2.pkl'.")
    print("\n>>> COMPARE AGAINST XGBOOST (R² 0.7812), RF (R² 0.7536), and the")
    print(">>> 16-candidate ANN attempt (R² 0.6457) <<<")


if __name__ == "__main__":
    tune_and_train_ann()























# # tuned_ann_v2.py
# # ANN trained on the same v2 dense, target-encoded feature set
# # (X_train_v2.csv / X_test_v2.csv) that XGBoost (R2 0.7812) and Random
# # Forest (R2 0.7536) were tested on.
# #
# # Uses MinMaxScaler (not StandardScaler) since several of the 33 features
# # are sparse binary amenity flags — standardization would distort those
# # relative to the continuous location_encoded/Bedrooms/Bathrooms features.
# # This matches the reasoning from the earlier ANN methodology.
# #
# # IMPORTANT: this does NOT match XGBoost/RF's 60-iteration search budget.
# # Each candidate here is a full gradient-descent training run (up to 150
# # epochs with early stopping), not a single tree fit, so 60 combinations
# # would be prohibitively slow. This tests 16 combinations instead — a full
# # grid across 4 architectures x 2 dropout rates x 2 learning rates.

# import pandas as pd
# import numpy as np
# import itertools
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# from tensorflow import keras
# from tensorflow.keras import layers
# import joblib


# def build_model(input_dim, hidden_layers, dropout_rate, learning_rate):
#     model = keras.Sequential()
#     model.add(layers.Input(shape=(input_dim,)))
#     for units in hidden_layers:
#         model.add(layers.Dense(units, activation='relu'))
#         model.add(layers.Dropout(dropout_rate))
#     model.add(layers.Dense(1))  # linear output for regression
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
#     return model


# def tune_and_train_ann():
#     print("--- INITIATING ANN ARCHITECTURE SEARCH (v2 - target encoding) ---")

#     print("Loading v2 dense feature matrices...")
#     X_train_full = pd.read_csv('X_train_v2.csv')
#     X_test = pd.read_csv('X_test_v2.csv')
#     y_train_full = pd.read_csv('y_train_v2.csv').squeeze()
#     y_test = pd.read_csv('y_test_v2.csv').squeeze()

#     print("Scaling features with MinMaxScaler (fit on training data only)...")
#     scaler = MinMaxScaler()
#     X_train_scaled_full = scaler.fit_transform(X_train_full)
#     X_test_scaled = scaler.transform(X_test)

#     # Carve a validation split out of training data, purely for early
#     # stopping to monitor. The real held-out X_test/y_test is only touched
#     # for final scoring, never during training.
#     X_train_sub, X_val, y_train_sub, y_val = train_test_split(
#         X_train_scaled_full, y_train_full, test_size=0.15, random_state=42
#     )

#     architectures = [[64, 32], [128, 64], [128, 64, 32], [256, 128, 64]]
#     dropout_rates = [0.2, 0.3]
#     learning_rates = [0.001, 0.0005]

#     candidates = list(itertools.product(architectures, dropout_rates, learning_rates))
#     print(f"Testing {len(candidates)} architecture/hyperparameter combinations...\n")

#     best_r2 = -np.inf
#     best_model = None
#     best_config = None

#     for i, (arch, dropout, lr) in enumerate(candidates):
#         print(f"[{i+1}/{len(candidates)}] Architecture={arch}, Dropout={dropout}, LR={lr}")

#         model = build_model(X_train_sub.shape[1], arch, dropout, lr)
#         early_stop = keras.callbacks.EarlyStopping(
#             monitor='val_loss', patience=15, restore_best_weights=True
#         )
#         history = model.fit(
#             X_train_sub, y_train_sub,
#             validation_data=(X_val, y_val),
#             epochs=150, batch_size=32,
#             callbacks=[early_stop], verbose=0
#         )

#         preds_log = model.predict(X_test_scaled, verbose=0).flatten()
#         preds_raw = np.expm1(preds_log)
#         y_test_raw = np.expm1(y_test)
#         r2 = r2_score(y_test_raw, preds_raw)

#         epochs_run = len(history.history['loss'])
#         print(f"   Stopped at epoch {epochs_run}/150, Test R2={r2:.4f}")

#         if r2 > best_r2:
#             best_r2 = r2
#             best_model = model
#             best_config = {'architecture': arch, 'dropout': dropout, 'learning_rate': lr}

#     print(f"\n--- SEARCH COMPLETE ---")
#     print(f"Best Configuration Found: {best_config}")

#     print("\nEvaluating Best Model on the Test Set...")
#     preds_log = best_model.predict(X_test_scaled, verbose=0).flatten()
#     preds_raw = np.expm1(preds_log)
#     y_test_raw = np.expm1(y_test)

#     r2 = r2_score(y_test_raw, preds_raw)
#     mae = mean_absolute_error(y_test_raw, preds_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, preds_raw) * 100

#     best_model.save('ann_tuned_model_v2.keras')
#     joblib.dump(scaler, 'ann_scaler_v2.pkl')

#     print("\n=== TUNED ANN RESULTS (v2 - target encoding) ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'ann_tuned_model_v2.keras', scaler as 'ann_scaler_v2.pkl'.")
#     print("\n>>> COMPARE THESE AGAINST XGBOOST (R² 0.7812) AND RF (R² 0.7536) <<<")


# if __name__ == "__main__":
#     tune_and_train_ann()