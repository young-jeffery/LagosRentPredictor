# cv_comparison_xgb_vs_rf.py
#
# Reviewer wants uncertainty quantification on the XGBoost-vs-RF production
# decision, not just a single train/test split point estimate. This loads
# the ALREADY-TUNED hyperparameters from both saved models (no new search --
# that would be a different, much longer job) and refits each 5 times on
# 5-fold CV splits of the combined train+test data, so we get a real
# fold-wise R²/MAPE spread for both.

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

N_FOLDS = 5
SEED = 42

def main():
    # Combine train+test back into one pool -- CV needs to draw its own
    # folds, not reuse the single held-out split these models were
    # evaluated on originally.
    X_train = pd.read_csv('X_train_v3.csv')
    X_test = pd.read_csv('X_test_v3.csv')
    y_train = pd.read_csv('y_train_v3.csv').squeeze()
    y_test = pd.read_csv('y_test_v3.csv').squeeze()

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    print(f"Combined pool: {len(X)} rows, {X.shape[1]} features")

    # Pull the already-tuned hyperparameters straight from the saved
    # production models -- no re-tuning here, just reusing what
    # tuned_xgboost_v3.py / tuned_rf_v3.py already found.
    xgb_saved = joblib.load('xgb_tuned_model_v3.pkl')
    rf_saved = joblib.load('rf_tuned_model_v3_monotonic.pkl')

    xgb_params = xgb_saved.get_params()
    rf_params = rf_saved.get_params()
    print("\nUsing XGBoost's tuned params:", {k: v for k, v in xgb_params.items() if k in
          ['learning_rate', 'n_estimators', 'max_depth', 'subsample', 'colsample_bytree', 'min_child_weight']})
    print("Using RF's tuned params:", {k: v for k, v in rf_params.items() if k in
          ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']})

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    xgb_r2_scores, xgb_mape_scores = [], []
    rf_r2_scores, rf_mape_scores = [], []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fresh model each fold, same tuned hyperparameters throughout.
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_fold_train, y_fold_train)
        xgb_pred_raw = np.expm1(xgb_model.predict(X_fold_val))
        y_val_raw = np.expm1(y_fold_val)
        xgb_r2 = r2_score(y_val_raw, xgb_pred_raw)
        xgb_mape = mean_absolute_percentage_error(y_val_raw, xgb_pred_raw) * 100
        xgb_r2_scores.append(xgb_r2)
        xgb_mape_scores.append(xgb_mape)

        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_fold_train, y_fold_train)
        rf_pred_raw = np.expm1(rf_model.predict(X_fold_val))
        rf_r2 = r2_score(y_val_raw, rf_pred_raw)
        rf_mape = mean_absolute_percentage_error(y_val_raw, rf_pred_raw) * 100
        rf_r2_scores.append(rf_r2)
        rf_mape_scores.append(rf_mape)

        print(f"\nFold {fold_num}/{N_FOLDS}:")
        print(f"  XGBoost -- R²: {xgb_r2:.4f}  MAPE: {xgb_mape:.2f}%")
        print(f"  RF      -- R²: {rf_r2:.4f}  MAPE: {rf_mape:.2f}%")

    xgb_r2_arr, rf_r2_arr = np.array(xgb_r2_scores), np.array(rf_r2_scores)
    xgb_mape_arr, rf_mape_arr = np.array(xgb_mape_scores), np.array(rf_mape_scores)

    print("\n" + "=" * 70)
    print("5-FOLD CV SUMMARY")
    print("=" * 70)
    print(f"XGBoost R²:   {xgb_r2_arr.mean():.4f} ± {xgb_r2_arr.std():.4f}  (range: {xgb_r2_arr.min():.4f}-{xgb_r2_arr.max():.4f})")
    print(f"RF R²:        {rf_r2_arr.mean():.4f} ± {rf_r2_arr.std():.4f}  (range: {rf_r2_arr.min():.4f}-{rf_r2_arr.max():.4f})")
    print(f"XGBoost MAPE: {xgb_mape_arr.mean():.2f}% ± {xgb_mape_arr.std():.2f}%")
    print(f"RF MAPE:      {rf_mape_arr.mean():.2f}% ± {rf_mape_arr.std():.2f}%")

    non_overlapping = xgb_r2_arr.min() > rf_r2_arr.max()
    print(f"\nDo XGBoost's and RF's R² ranges overlap across folds? {'NO -- XGBoost strictly ahead every fold' if non_overlapping else 'YES -- ranges overlap'}")
    print("\nUse this in your paper sentence:")
    if non_overlapping:
        print(f'  "Across 5-fold cross-validation, XGBoost (R² = {xgb_r2_arr.mean():.4f} ± {xgb_r2_arr.std():.4f})')
        print(f'  outperformed Random Forest (R² = {rf_r2_arr.mean():.4f} ± {rf_r2_arr.std():.4f}) in every fold,')
        print(f'  indicating the gap exceeds ordinary fold-to-fold variance."')
    else:
        print(f'  "Across 5-fold cross-validation, XGBoost (R² = {xgb_r2_arr.mean():.4f} ± {xgb_r2_arr.std():.4f})')
        print(f'  and Random Forest (R² = {rf_r2_arr.mean():.4f} ± {rf_r2_arr.std():.4f}) showed overlapping')
        print(f'  per-fold ranges, indicating the single-split gap should not be read as decisive on its own."')


if __name__ == '__main__':
    main()