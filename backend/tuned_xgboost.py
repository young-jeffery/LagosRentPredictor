# First implementation of the XGBoost tuning script. This script will load the processed training and testing data, perform a randomized search to find the best hyperparameters for the XGBoost model, and then evaluate the tuned model on the test set. The results will be printed in a clear format, and the best model will be saved for future use.
import pandas as pd
import numpy as np
import scipy.sparse
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib

def tune_and_train_xgboost():
    print("--- INITIATING XGBOOST HYPERPARAMETER TUNING ---")
    
    # 1. Loading the matrices and targets
    print("Loading feature matrices...")
    X_train = scipy.sparse.load_npz('X_train_processed.npz')
    X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
    y_train = pd.read_csv('y_train.csv').squeeze() 
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    # 2. Defining the parameter grid for RandomizedSearchCV
    # This grid controls how aggressive and how deep the XGBoost algorithm can go
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0] # How many features (columns) it looks at per tree
    }
    
    # 3. Initializing the base model
    base_xgb = XGBRegressor(random_state=42, n_jobs=-1)
    
    # 4. Setting up the Randomized Search
    # We are testing 30 random combinations across 3 folds of cross-validation (90 total fits)
    print("Starting Randomized Search (Testing 30 combinations. This will take a few minutes)...")
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=30,
        scoring='neg_mean_squared_error', # We want it to minimize the MSE in log-space
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Execute the search
    random_search.fit(X_train, y_train)
    
    # Extract the absolute best model from the search
    best_xgb_model = random_search.best_estimator_
    
    print("\n--- TUNING COMPLETE ---")
    print(f"Best Parameters Found: {random_search.best_params_}")
    
    # 6. Evaluate the winning model
    print("\nEvaluating Best Model on the Test Set...")
    predictions_log = best_xgb_model.predict(X_test)
    
    # 7. Reverse the Log Transformation for accurate metrics
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)
    
    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
    # 8. Save the tuned model
    joblib.dump(best_xgb_model, 'xgb_tuned_model.pkl')
    
    print("\n=== TUNED XGBOOST RESULTS ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")

if __name__ == "__main__":
    tune_and_train_xgboost()