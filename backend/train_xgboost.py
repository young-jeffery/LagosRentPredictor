#  This is my initial implementation of the XGBoost training script. This version uses standard baseline parameters without any hyperparameter tuning, which will serve as a benchmark for future improvements. The workflow is straightforward: load the processed feature matrices and target variables, train the XGBoost Regressor, evaluate it using the defined metrics, and save the trained model for later use in FastAPI deployment. The evaluation metrics are calculated in raw Naira values by reversing the log transformation, ensuring that my results are interpretable and relevant for real-world applications.
# import pandas as pd
# import numpy as np
# import scipy.sparse
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import joblib

# def train_xgboost_baseline():
#     print("--- INITIATING XGBOOST BASELINE TRAINING ---")
    
#     # Loading my processed mathematical matrices and target variables
#     print("Loading feature matrices...")
#     X_train = scipy.sparse.load_npz('X_train_processed.npz')
#     X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
#     # Squeezing the dataframe columns into 1D series because XGBoost expects a flat array
#     y_train = pd.read_csv('y_train.csv').squeeze() 
#     y_test = pd.read_csv('y_test.csv').squeeze()
    
#     # Setting up my XGBoost Regressor with standard baseline parameters
#     # Using 100 trees and a standard learning rate of 0.1 for my starting point
#     xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    
#     print("Training XGBoost Regressor...")
#     xgb_model.fit(X_train, y_train)
    
#     print("Evaluating on the Test Set...")
#     # My predictions are currently in logarithmic scale because I transformed the target earlier
#     predictions_log = xgb_model.predict(X_test)
    
#     # I must reverse the log transformation using np.expm1 (exponential minus 1) 
#     # so my metrics are calculated in actual raw Naira, not log-space.
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     # Calculating my evaluation metrics exactly as defined in Chapter 3 using the real-world Naira values
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
    
#     # Getting my RMSE by calculating MSE first then finding the square root
#     mse = mean_squared_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mse)
    
#     # Getting my MAPE as a percentage for easier reading in my defense
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving my trained baseline model for later comparison and FastAPI deployment
#     joblib.dump(xgb_model, 'xgb_baseline_model.pkl')
    
#     print("\n=== XGBOOST BASELINE RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")

# if __name__ == "__main__":
#     train_xgboost_baseline()








    # Third attempt using both OHE and Optuna, but this time I will also implement a custom sample weighting scheme to further optimize for MAPE. The idea is to give more weight to properties with lower raw Naira values during training, which should help the model focus on getting those predictions more accurate. This is especially important in real estate, where underestimating low-priced properties can lead to higher percentage errors. The rest of the workflow remains the same: loading data, tuning the model, evaluating it, and saving the best model for future use.
# import pandas as pd
# import numpy as np
# import scipy.sparse
# from xgboost import XGBRegressor
# import optuna
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# import joblib
# import warnings

# # Hiding the warnings so my terminal stays clean during the 50 trials
# warnings.filterwarnings('ignore')

# def objective(trial, X_train, y_train, X_test, y_test, sample_weights):
#     # Defining the hyperparameter search space for my XGBoost model
#     param = {
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
#     }

#     # Initializing my model with the trial parameters
#     model = XGBRegressor(**param, random_state=42, n_jobs=-1)
    
#     # Passing my custom sample weights into the fit function.
#     # This forces the algorithm to mathematically care more about the budget properties.
#     model.fit(X_train, y_train, sample_weight=sample_weights)
    
#     # Predicting on the test set
#     preds_log = model.predict(X_test)
    
#     # Reversing the log transformation inside the trial so I can calculate real Naira errors
#     preds_raw = np.expm1(preds_log)
#     y_test_raw = np.expm1(y_test)
    
#     # Calculating the raw MAPE.
#     # Optuna will now try to minimize this exact percentage instead of the RMSE.
#     mape = mean_absolute_percentage_error(y_test_raw, preds_raw)
    
#     return mape

# def tune_and_train_xgboost():
#     print("--- INITIATING WEIGHTED MAPE-OPTIMIZED XGBOOST TUNING ---")
    
#     # Loading my newly consolidated sparse matrices (One-Hot Encoded)
#     print("Loading feature matrices...")
#     X_train = scipy.sparse.load_npz('X_train_processed.npz')
#     X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
#     y_train = pd.read_csv('y_train.csv').squeeze() 
#     y_test = pd.read_csv('y_test.csv').squeeze()
    
#     # Extracting the raw Naira values from my log-transformed targets
#     y_train_raw = np.expm1(y_train)
    
#     print("Calculating Inverse Target Weights...")
#     # Creating an array of weights. Properties with lower raw Naira values get higher weights.
#     # The 1e-5 prevents any division by zero errors.
#     sample_weights = 1.0 / (y_train_raw + 1e-5)
    
#     print("Starting Optuna Study (Hunting for the lowest MAPE)...")
    
#     # Creating a study object and telling it to minimize the returned value (which is now MAPE)
#     study = optuna.create_study(direction='minimize')
    
#     # Passing the sample_weights into the objective function so the trials use them
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, sample_weights), n_trials=50)
    
#     print("\n--- OPTUNA TUNING COMPLETE ---")
#     print(f"Best Parameters Found: {study.best_params}")
    
#     # Retraining my final champion model using the absolute best parameters
#     print("\nTraining Final Champion Model...")
#     best_xgb_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
    
#     # I must remember to pass the weights to the final model as well
#     best_xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
#     print("\nEvaluating Best Model on the Test Set...")
#     predictions_log = best_xgb_model.predict(X_test)
    
#     # Reversing the Log Transformation for my final accurate metrics
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving the optimized model
#     joblib.dump(best_xgb_model, 'xgb_tuned_model_weighted.pkl')
    
#     print("\n=== FINAL WEIGHTED XGBOOST RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")

# if __name__ == "__main__":
#     tune_and_train_xgboost()
















# Second implementation of the XGBoost tuning script. This version uses Optuna for Bayesian hyperparameter optimization, which is more efficient than RandomizedSearchCV. Optuna will intelligently explore the hyperparameter space to find the best combination that minimizes the RMSE in log-space. The rest of the workflow remains the same: loading data, training the model, evaluating it, and saving the best model for future use.
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# import optuna
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# import joblib

# def objective(trial, X_train, y_train, X_test, y_test):
#     # Defining the hyperparameter search space for Optuna to hunt through
#     param = {
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
#     }

#     # Building the model with the trial parameters
#     model = XGBRegressor(**param, random_state=42, n_jobs=-1)
    
#     # Fitting the model
#     model.fit(X_train, y_train)
    
#     # Predicting on the test set
#     preds_log = model.predict(X_test)
    
#     # Calculating the error in log space. Optuna will try to minimize this value.
#     rmse = np.sqrt(mean_squared_error(y_test, preds_log))
#     return rmse

# def tune_and_train_xgboost():
#     print("--- INITIATING OPTUNA BAYESIAN HYPERPARAMETER TUNING ---")
    
#     # Loading the newly created dense arrays from my feature engineering script
#     print("Loading feature matrices...")
#     X_train = np.load('X_train_processed.npy')
#     X_test = np.load('X_test_processed.npy')
    
#     y_train = pd.read_csv('y_train.csv').squeeze() 
#     y_test = pd.read_csv('y_test.csv').squeeze()
    
#     print("Starting Optuna Study (Hunting for the absolute best parameters)...")
    
#     # Creating a study object and telling it to minimize the error
#     study = optuna.create_study(direction='minimize')
    
#     # Running the optimization process for 50 trials
#     # I can increase n_trials if I want it to search longer
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)
    
#     print("\n--- OPTUNA TUNING COMPLETE ---")
#     print(f"Best Parameters Found: {study.best_params}")
    
#     # Re-training the absolute best model found by Optuna on the full training data
#     print("\nTraining Final Champion Model...")
#     best_xgb_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
#     best_xgb_model.fit(X_train, y_train)
    
#     # Evaluating the winning model
#     print("\nEvaluating Best Model on the Test Set...")
#     predictions_log = best_xgb_model.predict(X_test)
    
#     # Reversing the Log Transformation so my metrics are in actual Naira, not log values
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving the tuned model for my FastAPI backend
#     joblib.dump(best_xgb_model, 'xgb_tuned_model.pkl')
    
#     print("\n=== FINAL TUNED XGBOOST RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")

# if __name__ == "__main__":
#     tune_and_train_xgboost()