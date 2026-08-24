# tuned_rf_v3.py -- Random Forest on v3 feature set (+ Property_Type)
# Same search budget (60 iter, 5-fold CV) as tuned_xgboost_v3.py, for a
# fair comparison. Note: RF has no monotone_constraints equivalent in
# sklearn, so this is RF on the v3 features WITHOUT any bedroom/bathroom
# constraint -- comparing "best RF can do on this feature set" against
# XGBoost's constrained result. Not apples-to-apples on the constraint,
# but still tells you whether RF's algorithm-level ceiling on this richer
# feature set beats XGBoost's constrained ceiling.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib

def tune_and_train_rf():
    print("--- INITIATING RANDOM FOREST HYPERPARAMETER TUNING (v3 - + Property_Type) ---")

    print("Loading v3 dense feature matrices...")
    X_train = pd.read_csv('X_train_v3.csv')
    X_test = pd.read_csv('X_test_v3.csv')

    y_train = pd.read_csv('y_train_v3.csv').squeeze()
    y_test = pd.read_csv('y_test_v3.csv').squeeze()

    param_grid = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

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

    joblib.dump(best_rf_model, 'rf_tuned_model_v3.pkl')

    print("\n=== TUNED RANDOM FOREST RESULTS (v3 - + Property_Type) ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'rf_tuned_model_v3.pkl'.")
    print("\n>>> COMPARE AGAINST: XGBoost v3 constrained+PropType (R² 0.7735),")
    print(">>> RF v2 (R² 0.7536), XGBoost v2 unconstrained (R² 0.7812) <<<")

if __name__ == "__main__":
    tune_and_train_rf()