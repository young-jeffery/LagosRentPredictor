# tuned_xgboost_v3.py -- Monotonic constraint + Property_Type features
#
# [MODIFIED] Now trains against the v3 feature set (34 features: v2's 33
# + property_type_encoded) instead of the original v2 set (33 features).
# Monotonic constraints on Bedrooms/Bathrooms unchanged -- built dynamically
# by column name, so it carries over automatically regardless of feature
# count or position.
#
# [HISTORY] This file was previously trained on X_train_v2.csv (33
# features, monotonic constraint only -- R2 0.7534, MAE ~2.28M,
# MAPE 45.46%). This run supersedes that result; xgb_tuned_model_v3.pkl
# is overwritten. If you need the old 33-feature constrained model back,
# it must be retrained from X_train_v2.csv -- it was not kept as a
# separate file.
#
# TO REVERT to unconstrained: use tuned_xgboost_v2.py (still points at
# X_train_v2.csv, still produces xgb_tuned_model_v2.pkl, R2 0.7812 --
# untouched and safe).

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib

MONOTONE_UP_FEATURES = ['Bedrooms', 'Bathrooms']

def tune_and_train_xgboost():
    print("--- INITIATING XGBOOST HYPERPARAMETER TUNING (v3 - monotonic constraints + Property_Type) ---")

    print("Loading v3 dense feature matrices...")
    X_train = pd.read_csv('X_train_v3.csv')
    X_test = pd.read_csv('X_test_v3.csv')

    y_train = pd.read_csv('y_train_v3.csv').squeeze()
    y_test = pd.read_csv('y_test_v3.csv').squeeze()

    missing = [f for f in MONOTONE_UP_FEATURES if f not in X_train.columns]
    if missing:
        raise ValueError(f"Expected monotone features not found in X_train: {missing}")

    monotone_constraints = tuple(1 if col in MONOTONE_UP_FEATURES else 0 for col in X_train.columns)
    constrained_positions = {col: i for i, col in enumerate(X_train.columns) if col in MONOTONE_UP_FEATURES}
    print(f"Loaded {X_train.shape[1]} features (v2 had 33, +1 for property_type_encoded)")
    print(f"Monotone constraints applied at positions: {constrained_positions}")
    print(f"Constraint tuple ({len(monotone_constraints)} features): {monotone_constraints}")

    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }

    base_xgb = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        monotone_constraints=monotone_constraints
    )

    print("Starting Randomized Search (Testing 60 combinations, 5-fold CV. This may take a while)...")
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=60,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_xgb_model = random_search.best_estimator_

    print("\n--- TUNING COMPLETE ---")
    print(f"Best Parameters Found: {random_search.best_params_}")

    print("\nEvaluating Best Model on the Test Set...")
    predictions_log = best_xgb_model.predict(X_test)

    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)

    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100

    joblib.dump(best_xgb_model, 'xgb_tuned_model_v3.pkl')

    print("\n=== TUNED XGBOOST RESULTS (v3 - monotonic constraints + Property_Type) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'xgb_tuned_model_v3.pkl' (overwrites prior v3 constrained-only model).")
    print("\n>>> COMPARE AGAINST: v2 unconstrained R² 0.7812 | v3 constrained-only R² 0.7534 <<<")
    print(">>> If Property_Type earns back some/all of the 0.0278 R² lost to the constraint, <<<")
    print(">>> this becomes the new production candidate. <<<")

if __name__ == "__main__":
    tune_and_train_xgboost()


























# # tuned_xgboost_v3.py -- Monotonic constraint variant
# #
# # [MODIFIED FROM v2] Adds XGBoost monotone_constraints=1 on Bedrooms and
# # Bathrooms to fix the non-monotonic (price decreasing with more bedrooms)
# # prediction bug in sparse high-bedroom regions. Constraint tuple is built
# # dynamically from X_train.columns (not hardcoded indices) so it can't
# # silently break if column order ever shifts upstream.
# #
# # Everything else (grid, n_iter=60, cv=5, target-encoded dense CSVs) is
# # identical to tuned_xgboost_v2.py.
# #
# # TO REVERT: use tuned_xgboost_v2.py — untouched, still your production
# # model (xgb_tuned_model_v2.pkl unaffected by this script).

# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import joblib

# MONOTONE_UP_FEATURES = ['Bedrooms', 'Bathrooms']

# def tune_and_train_xgboost():
#     print("--- INITIATING XGBOOST HYPERPARAMETER TUNING (v3 - monotonic constraints) ---")

#     print("Loading v2 dense feature matrices...")
#     X_train = pd.read_csv('X_train_v2.csv')
#     X_test = pd.read_csv('X_test_v2.csv')

#     y_train = pd.read_csv('y_train_v2.csv').squeeze()
#     y_test = pd.read_csv('y_test_v2.csv').squeeze()

#     # [NEW] Build constraint tuple from actual column order, not hardcoded
#     # positions -- safe even if upstream pipeline reorders columns later.
#     missing = [f for f in MONOTONE_UP_FEATURES if f not in X_train.columns]
#     if missing:
#         raise ValueError(f"Expected monotone features not found in X_train: {missing}")

#     monotone_constraints = tuple(1 if col in MONOTONE_UP_FEATURES else 0 for col in X_train.columns)
#     constrained_positions = {col: i for i, col in enumerate(X_train.columns) if col in MONOTONE_UP_FEATURES}
#     print(f"Monotone constraints applied at positions: {constrained_positions}")
#     print(f"Constraint tuple ({len(monotone_constraints)} features): {monotone_constraints}")

#     param_grid = {
#         'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
#         'n_estimators': [100, 200, 300, 500, 800],
#         'max_depth': [2, 3, 4, 5, 6, 7],
#         'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#         'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
#         'min_child_weight': [1, 3, 5, 7]
#     }

#     # [NEW] monotone_constraints fixed here, not searched -- it's a
#     # correctness constraint, not a tunable hyperparameter.
#     base_xgb = XGBRegressor(
#         random_state=42,
#         n_jobs=-1,
#         monotone_constraints=monotone_constraints
#     )

#     print("Starting Randomized Search (Testing 60 combinations, 5-fold CV. This may take a while)...")
#     random_search = RandomizedSearchCV(
#         estimator=base_xgb,
#         param_distributions=param_grid,
#         n_iter=60,
#         scoring='neg_mean_squared_error',
#         cv=5,
#         verbose=1,
#         random_state=42,
#         n_jobs=-1
#     )

#     random_search.fit(X_train, y_train)
#     best_xgb_model = random_search.best_estimator_

#     print("\n--- TUNING COMPLETE ---")
#     print(f"Best Parameters Found: {random_search.best_params_}")

#     print("\nEvaluating Best Model on the Test Set...")
#     predictions_log = best_xgb_model.predict(X_test)

#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)

#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100

#     joblib.dump(best_xgb_model, 'xgb_tuned_model_v3.pkl')

#     print("\n=== TUNED XGBOOST RESULTS (v3 - monotonic constraints) ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'xgb_tuned_model_v3.pkl'.")
#     print("\n>>> COMPARE THESE NUMBERS AGAINST YOUR CURRENT BEST: R² 0.7812 (v2, unconstrained) <<<")
#     print(">>> Expect R² to hold roughly steady or dip slightly. The real check is monotonicity, <<<")
#     print(">>> not global R² -- run a manual sweep (fix all features, vary Bedrooms 1-10) after training. <<<")

# if __name__ == "__main__":
#     tune_and_train_xgboost()