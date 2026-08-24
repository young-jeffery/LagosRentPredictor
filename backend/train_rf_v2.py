# train_rf_v2.py
# Identical to the active version of train_rf.py (Optuna-tuned Random
# Forest, 30 trials). Only the input/output filenames change, so it
# trains on the v2 (reordered) feature matrices and saves to a separate
# model file, leaving rf_tuned_model_complete.pkl (your live model) untouched.
#
# TO REVERT / COMPARE: New file, does not touch train_rf.py.

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import optuna
import joblib

def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds_log = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_log))
    return rmse

def train_and_evaluate_rf():
    print("--- INITIATING OPTUNA TUNED RANDOM FOREST (v2) ---")

    print("Loading v2 feature matrices...")
    X_train = scipy.sparse.load_npz('X_train_v2.npz')
    X_test = scipy.sparse.load_npz('X_test_v2.npz')

    y_train = pd.read_csv('y_train_v2.csv').squeeze()
    y_test = pd.read_csv('y_test_v2.csv').squeeze()

    print("Starting Optuna Study (30 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)

    print("\n--- OPTUNA TUNING COMPLETE ---")
    print(f"Best Parameters Found: {study.best_params}")

    print("\nTraining Final Champion Random Forest (v2)...")
    best_rf_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    best_rf_model.fit(X_train, y_train)

    print("Evaluating on the Test Set...")
    predictions_log = best_rf_model.predict(X_test)
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)

    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100

    # [MODIFIED] Saved as rf_tuned_model_v2.pkl — your live model file is untouched.
    joblib.dump(best_rf_model, 'rf_tuned_model_v2.pkl')

    print("\n=== TUNED RANDOM FOREST RESULTS (v2 - reordered pipeline) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'rf_tuned_model_v2.pkl'.")
    print("\n>>> COMPARE THESE NUMBERS AGAINST YOUR CURRENT BASELINE: R² 0.7166 <<<")

if __name__ == "__main__":
    train_and_evaluate_rf()