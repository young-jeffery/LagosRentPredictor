# tuned_rf_v3.py -- Random Forest on v3 feature set (+ Property_Type),
# NOW WITH monotonic constraint on Bedrooms/Bathrooms.
#
# [UPDATED] scikit-learn >= 1.4 added native monotonic constraint support
# to RandomForestRegressor via the `monotonic_cst` parameter -- an array
# of {-1, 0, 1} per feature, same semantics as XGBoost's monotone_constraints
# (1 = prediction must be non-decreasing as the feature increases). This
# closes the gap that previously existed: RF had no equivalent mechanism
# to XGBoost's monotone_constraints when the original v3 comparison was
# run, which was the deciding factor XGBoost was selected for production
# on (see Chapter 5, Section 5.3). Requires scikit-learn>=1.4 -- check
# your installed version with `pip show scikit-learn` before running.
#
# Same search budget (60 iter, 5-fold CV) as tuned_xgboost_v3.py, for a
# fair, apples-to-apples comparison now that both algorithms are
# constrained the same way on the same features.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib
import sklearn

MONOTONE_UP_FEATURES = ['Bedrooms', 'Bathrooms']

def tune_and_train_rf():
    print("--- INITIATING RANDOM FOREST HYPERPARAMETER TUNING (v3 - + Property_Type + monotonic constraint) ---")

    sklearn_version = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
    if sklearn_version < (1, 4):
        raise RuntimeError(
            f"monotonic_cst requires scikit-learn>=1.4, found {sklearn.__version__}. "
            f"Run 'pip install --upgrade scikit-learn' before continuing."
        )
    print(f"scikit-learn version: {sklearn.__version__} (monotonic_cst supported)")

    print("Loading v3 dense feature matrices...")
    X_train = pd.read_csv('X_train_v3.csv')
    X_test = pd.read_csv('X_test_v3.csv')

    y_train = pd.read_csv('y_train_v3.csv').squeeze()
    y_test = pd.read_csv('y_test_v3.csv').squeeze()

    # Build the monotonic_cst array from actual column order, exactly the
    # same pattern tuned_xgboost_v3.py uses for its constraint tuple, so
    # it can't silently break if column order shifts upstream. sklearn
    # wants {-1, 0, 1} per feature: 1 = non-decreasing, 0 = unconstrained.
    missing = [f for f in MONOTONE_UP_FEATURES if f not in X_train.columns]
    if missing:
        raise ValueError(f"Expected monotone features not found in X_train: {missing}")

    monotonic_cst = np.array([1 if col in MONOTONE_UP_FEATURES else 0 for col in X_train.columns])
    constrained_positions = {col: i for i, col in enumerate(X_train.columns) if col in MONOTONE_UP_FEATURES}
    print(f"Loaded {X_train.shape[1]} features")
    print(f"Monotone constraints applied at positions: {constrained_positions}")
    print(f"monotonic_cst array ({len(monotonic_cst)} features): {monotonic_cst.tolist()}")

    param_grid = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    # NOTE: monotonic_cst is fixed here, not searched -- it's a correctness
    # constraint, not a tunable hyperparameter (same treatment as XGBoost's
    # monotone_constraints in tuned_xgboost_v3.py).
    #
    # NOTE: sklearn's monotonic trees currently require the 'squared_error'
    # criterion and do not support parallel tree building the same way as
    # unconstrained RF in every version -- if you hit a fit-time slowdown
    # relative to the old tuned_rf_v3.py run, that's expected, not a bug.
    base_rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        monotonic_cst=monotonic_cst
    )

    print("Starting Randomized Search (Testing 60 combinations, 5-fold CV. This will take a while)...")
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_grid,
        n_iter=60,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_rf_model = random_search.best_estimator_

    print("\n--- TUNING COMPLETE ---")
    print(f"Best Parameters Found: {random_search.best_params_}")

    print("\nEvaluating Best Model on the Test Set...")
    predictions_log = best_rf_model.predict(X_test)

    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)

    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100

    joblib.dump(best_rf_model, 'rf_tuned_model_v3_monotonic.pkl')

    print("\n=== TUNED RANDOM FOREST RESULTS (v3 - + Property_Type + monotonic constraint) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'rf_tuned_model_v3_monotonic.pkl' (old unconstrained")
    print("rf_tuned_model_v3.pkl left untouched for comparison).")
    print("\n>>> COMPARE AGAINST: unconstrained RF v3 (R² 0.7535, no monotonic mechanism),")
    print(">>> XGBoost v3 constrained+PropType (R² 0.7735, MAPE 43.82%) <<<")
    print(">>> Run check_monotonicity.py against this model before trusting it in production --")
    print(">>> monotonic_cst constrains the SPLIT search, it does not retroactively guarantee")
    print(">>> zero violations the way a hard post-hoc calibration would. Verify, don't assume. <<<")

if __name__ == "__main__":
    tune_and_train_rf()





















# # tuned_rf_v3.py -- Random Forest on v3 feature set (+ Property_Type)
# # Same search budget (60 iter, 5-fold CV) as tuned_xgboost_v3.py, for a
# # fair comparison. Note: RF has no monotone_constraints enabled here,
# # so this is RF on the v3 features WITHOUT any bedroom/bathroom
# # constraint -- comparing "best RF can do on this feature set" against
# # XGBoost's constrained result. Not apples-to-apples on the constraint,
# # but still tells you whether RF's algorithm-level ceiling on this richer
# # feature set beats XGBoost's constrained ceiling.

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import joblib

# def tune_and_train_rf():
#     print("--- INITIATING RANDOM FOREST HYPERPARAMETER TUNING (v3 - + Property_Type) ---")

#     print("Loading v3 dense feature matrices...")
#     X_train = pd.read_csv('X_train_v3.csv')
#     X_test = pd.read_csv('X_test_v3.csv')

#     y_train = pd.read_csv('y_train_v3.csv').squeeze()
#     y_test = pd.read_csv('y_test_v3.csv').squeeze()

#     param_grid = {
#         'n_estimators': [100, 200, 300, 500, 800],
#         'max_depth': [None, 5, 10, 15, 20, 30],
#         'min_samples_split': [2, 5, 10, 15],
#         'min_samples_leaf': [1, 2, 4, 8],
#         'max_features': ['sqrt', 'log2', None]
#     }

#     base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

#     print("Starting Randomized Search (Testing 60 combinations, 5-fold CV. This will take a while)...")
#     random_search = RandomizedSearchCV(
#         estimator=base_rf,
#         param_distributions=param_grid,
#         n_iter=60,
#         scoring='neg_mean_squared_error',
#         cv=5,
#         verbose=1,
#         random_state=42,
#         n_jobs=-1
#     )

#     random_search.fit(X_train, y_train)
#     best_rf_model = random_search.best_estimator_

#     print("\n--- TUNING COMPLETE ---")
#     print(f"Best Parameters Found: {random_search.best_params_}")

#     print("\nEvaluating Best Model on the Test Set...")
#     predictions_log = best_rf_model.predict(X_test)

#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)

#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100

#     joblib.dump(best_rf_model, 'rf_tuned_model_v3.pkl')

#     print("\n=== TUNED RANDOM FOREST RESULTS (v3 - + Property_Type) ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'rf_tuned_model_v3.pkl'.")
#     print("\n>>> COMPARE AGAINST: XGBoost v3 constrained+PropType (R² 0.7735),")
#     print(">>> RF v2 (R² 0.7536), XGBoost v2 unconstrained (R² 0.7812) <<<")

# if __name__ == "__main__":
#     tune_and_train_rf()