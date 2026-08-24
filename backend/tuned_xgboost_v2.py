# tuned_xgboost_v2.py -- Take 2
# [MODIFIED] Loads dense .csv feature matrices instead of sparse .npz,
# since feature_engineering_pipeline_v2.py now produces target-encoded
# (dense) features instead of one-hot-encoded (sparse) ones.
#
# [MODIFIED] Expanded the hyperparameter search grid and increased to
# 60 iterations / 5-fold CV. The target-encoded feature space (33 dims)
# is very different in shape from the old OHE space (162 dims), and
# testing showed it needs a wider search (in particular, lower learning
# rates and deeper trees) to find its best configuration. A quick test
# with the OLD grid/params gave R2 0.7367; the wider search below found
# R2 0.7846.
#
# TO REVERT: restore scipy.sparse.load_npz() calls and the narrower
# param_grid from the previous version of this file.

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib

def tune_and_train_xgboost():
    print("--- INITIATING XGBOOST HYPERPARAMETER TUNING (v2 - target encoding) ---")

    print("Loading v2 dense feature matrices...")
    # [MODIFIED] pd.read_csv instead of scipy.sparse.load_npz
    X_train = pd.read_csv('X_train_v2.csv')
    X_test = pd.read_csv('X_test_v2.csv')

    y_train = pd.read_csv('y_train_v2.csv').squeeze()
    y_test = pd.read_csv('y_test_v2.csv').squeeze()

    # [MODIFIED] Wider grid, validated against the target-encoded feature space
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }

    base_xgb = XGBRegressor(random_state=42, n_jobs=-1)

    print("Starting Randomized Search (Testing 60 combinations, 5-fold CV. This will take a while)...")
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

    joblib.dump(best_xgb_model, 'xgb_tuned_model_v2.pkl')

    print("\n=== TUNED XGBOOST RESULTS (v2 - target encoding) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'xgb_tuned_model_v2.pkl'.")
    print("\n>>> COMPARE THESE NUMBERS AGAINST YOUR CURRENT BEST: R² 0.7825 <<<")

if __name__ == "__main__":
    tune_and_train_xgboost()

































# # tuned_xgboost_v2.py -- Take 1
# # Identical to tuned_xgboost.py — same RandomizedSearchCV grid, same 30
# # iterations, same 3-fold CV. Only the input/output filenames change, so
# # it trains on the v2 (reordered pipeline) feature matrices and saves to
# # a separate model file. xgb_tuned_model.pkl (your live 71.66% model) is
# # untouched.
# #
# # TO REVERT / COMPARE: New file, does not touch tuned_xgboost.py.

# import pandas as pd
# import numpy as np
# import scipy.sparse
# from xgboost import XGBRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import joblib

# def tune_and_train_xgboost():
#     print("--- INITIATING XGBOOST HYPERPARAMETER TUNING (v2 - reordered pipeline) ---")

#     print("Loading v2 feature matrices...")
#     X_train = scipy.sparse.load_npz('X_train_v2.npz')
#     X_test = scipy.sparse.load_npz('X_test_v2.npz')

#     y_train = pd.read_csv('y_train_v2.csv').squeeze()
#     y_test = pd.read_csv('y_test_v2.csv').squeeze()

#     param_grid = {
#         'learning_rate': [0.01, 0.05, 0.1, 0.2],
#         'n_estimators': [100, 200, 300, 500],
#         'max_depth': [3, 5, 7, 9],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }

#     base_xgb = XGBRegressor(random_state=42, n_jobs=-1)

#     print("Starting Randomized Search (Testing 30 combinations. This may take a few minutes)...")
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

#     # [MODIFIED] Saved as xgb_tuned_model_v2.pkl — your live model file is untouched.
#     joblib.dump(best_xgb_model, 'xgb_tuned_model_v2.pkl')

#     print("\n=== TUNED XGBOOST RESULTS (v2 - reordered pipeline) ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'xgb_tuned_model_v2.pkl'.")

# if __name__ == "__main__":
#     tune_and_train_xgboost()