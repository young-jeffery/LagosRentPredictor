import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import optuna
import joblib

def objective(trial, X_train, y_train, X_test, y_test):
    # I am defining the hyperparameter search space for my Random Forest
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    # Initializing the model with the trial parameters
    model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
    
    # Fitting the model to the training data
    model.fit(X_train, y_train)
    
    # Predicting on the unseen test set
    preds_log = model.predict(X_test)
    
    # Calculating the RMSE in log-space. Optuna will use this to figure out if it is improving.
    rmse = np.sqrt(mean_squared_error(y_test, preds_log))
    return rmse

def train_and_evaluate_rf():
    print("--- INITIATING OPTUNA TUNED RANDOM FOREST ---")
    
    # I am loading my sparse feature matrices created by the One-Hot Encoder
    print("Loading feature matrices...")
    X_train = scipy.sparse.load_npz('X_train_processed.npz')
    X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
    # Squeezing converts the dataframe column into the 1D series that sklearn prefers
    y_train = pd.read_csv('y_train.csv').squeeze() 
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    print("Starting Optuna Study (Hunting for the absolute best parameters)...")
    
    # Creating a study object and telling it to minimize the error
    study = optuna.create_study(direction='minimize')
    
    # Running the optimization process for 30 trials to find the peak configuration
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    
    print("\n--- OPTUNA TUNING COMPLETE ---")
    print(f"Best Parameters Found: {study.best_params}")
    
    # Re-training my final champion Random Forest using the absolute best parameters
    print("\nTraining Final Champion Random Forest...")
    best_rf_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    best_rf_model.fit(X_train, y_train)
    
    print("Evaluating on the Test Set...")
    # My predictions are currently in logarithmic scale because of my target transformation
    predictions_log = best_rf_model.predict(X_test)
    
    # I must reverse the log transformation using np.expm1 so my metrics are in actual raw Naira
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)
    
    # Calculating the four official metrics for my defense
    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
    # Saving my optimized baseline model so it can be loaded later by my FastAPI server
    joblib.dump(best_rf_model, 'rf_tuned_model_complete.pkl')
    
    print("\n=== TUNED RANDOM FOREST RESULTS ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Model saved as 'rf_tuned_model_complete.pkl'.")

if __name__ == "__main__":
    train_and_evaluate_rf()

















# Second implementation of the Random Forest training script. This version incorporates Optuna for hyperparameter tuning, which will help us find the best configuration for our Random Forest model. The workflow remains the same: loading the processed data, defining an objective function for Optuna, running the optimization process, and then evaluating the best model on the test set before saving it for future use.
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import optuna
# import joblib

# def objective(trial, X_train, y_train, X_test, y_test):
#     # I am defining the hyperparameter search space for my Random Forest
#     param = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#         'max_depth': trial.suggest_int('max_depth', 5, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
#     }

#     # Initializing the model with the trial parameters
#     model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
    
#     # Fitting the model to the training data
#     model.fit(X_train, y_train)
    
#     # Predicting on the unseen test set
#     preds_log = model.predict(X_test)
    
#     # Calculating the RMSE in log-space. Optuna will use this to figure out if it is improving.
#     rmse = np.sqrt(mean_squared_error(y_test, preds_log))
#     return rmse

# def train_and_evaluate_rf():
#     print("--- INITIATING OPTUNA TUNED RANDOM FOREST ---")
    
#     # I am loading my dense feature matrices created by the Target Encoder
#     print("Loading feature matrices...")
#     X_train = np.load('X_train_processed.npy')
#     X_test = np.load('X_test_processed.npy')
    
#     # Squeezing converts the dataframe column into the 1D series that sklearn prefers
#     y_train = pd.read_csv('y_train.csv').squeeze() 
#     y_test = pd.read_csv('y_test.csv').squeeze()
    
#     print("Starting Optuna Study (Hunting for the absolute best parameters)...")
    
#     # Creating a study object and telling it to minimize the error
#     study = optuna.create_study(direction='minimize')
    
#     # Running the optimization process for 30 trials to find the peak configuration
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    
#     print("\n--- OPTUNA TUNING COMPLETE ---")
#     print(f"Best Parameters Found: {study.best_params}")
    
#     # Re-training my final champion Random Forest using the absolute best parameters
#     print("\nTraining Final Champion Random Forest...")
#     best_rf_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
#     best_rf_model.fit(X_train, y_train)
    
#     print("Evaluating on the Test Set...")
#     # My predictions are currently in logarithmic scale because of my target transformation
#     predictions_log = best_rf_model.predict(X_test)
    
#     # I must reverse the log transformation using np.expm1 so my metrics are in actual raw Naira
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     # Calculating the four official metrics for my defense
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving my optimized baseline model so it can be loaded later by my FastAPI server
#     joblib.dump(best_rf_model, 'rf_tuned_model_complete.pkl')
    
#     print("\n=== TUNED RANDOM FOREST RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'rf_tuned_model_complete.pkl'.")

# if __name__ == "__main__":
#     train_and_evaluate_rf()














# import pandas as pd
# import numpy as np
# import scipy.sparse
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import joblib

# def train_and_evaluate():
#     print("--- INITIATING BASELINE MODEL TRAINING ---")
    
#     # Loading my preprocessed mathematical matrices and target variables
#     print("Loading feature matrices...")
#     X_train = scipy.sparse.load_npz('X_train_processed.npz')
#     X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
#     # Squeeze converts the dataframe column into a 1D series, which sklearn prefers
#     y_train = pd.read_csv('y_train.csv').squeeze() 
#     y_test = pd.read_csv('y_test.csv').squeeze()
    
#     # Initializing the Random Forest Regressor
#     # Using 100 trees as a solid baseline and utilizing all CPU cores for maximum speed
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
#     print("Training Random Forest Regressor (This may take a few moments)...")
#     rf_model.fit(X_train, y_train)
    
#     print("Evaluating on the Test Set...")
#     # These predictions are currently in logarithmic scale because I transformed the target earlier
#     predictions_log = rf_model.predict(X_test)
    
#     # I must reverse the log transformation using np.expm1 (exponential minus 1) 
#     # so my metrics are calculated in actual raw Naira, not log-space.
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     # Calculating the four official metrics for my defense using the raw, real-world values
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
    
#     # sklearn doesn't have a direct RMSE function, so I calculate MSE and take the square root
#     mse = mean_squared_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mse)
    
#     # Calculating MAPE and multiplying by 100 to get a clean percentage format
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving the baseline model so it can be loaded later by my FastAPI server
#     joblib.dump(rf_model, 'rf_baseline_model_complete.pkl')
    
#     # Outputting the comprehensive evaluation results
#     print("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Model saved as 'rf_baseline_model_complete.pkl'.")

# if __name__ == "__main__":
#     train_and_evaluate()








# # import pandas as pd
# # import numpy as np
# # import scipy.sparse
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# # import joblib

# # def train_and_evaluate():
# #     print("--- INITIATING BASELINE MODEL TRAINING ---")
    
# #     # Load the processed mathematical matrices and target variables
# #     print("Loading feature matrices...")
# #     X_train = scipy.sparse.load_npz('X_train_processed.npz')
# #     X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
# #     # Squeeze converts the dataframe column into a 1D series, which sklearn prefers
# #     y_train = pd.read_csv('y_train.csv').squeeze() 
# #     y_test = pd.read_csv('y_test.csv').squeeze()
    
# #     # Initialize the Random Forest Regressor
# #     # Using 100 trees as a solid baseline and utilizing all CPU cores for speed
# #     rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
# #     print("Training Random Forest Regressor (This may take a few moments)...")
# #     rf_model.fit(X_train, y_train)
    
# #     print("Evaluating on the Test Set...")
# #     predictions = rf_model.predict(X_test)
    
# #     # Calculate the four official metrics defined in Section 3.5.3
# #     r2 = r2_score(y_test, predictions)
# #     mae = mean_absolute_error(y_test, predictions)
    
# #     # sklearn doesn't have a direct RMSE function, so we calculate MSE and take the square root
# #     mse = mean_squared_error(y_test, predictions)
# #     rmse = np.sqrt(mse)
    
# #     # Calculate MAPE and multiply by 100 to get a clean percentage format
# #     mape = mean_absolute_percentage_error(y_test, predictions) * 100
    
# #     # Save the baseline model so it can be loaded later by the FastAPI server
# #     joblib.dump(rf_model, 'rf_baseline_model_complete.pkl')
    
# #     # Output the comprehensive evaluation results
# #     print("\n=== COMPREHENSIVE EVALUATION RESULTS ===")
# #     print(f"1. R-Squared (R²): {r2:.4f}")
# #     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
# #     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
# #     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# #     print("========================================")
# #     print("Model saved as 'rf_baseline_model_complete.pkl'.")

# # if __name__ == "__main__":
# #     train_and_evaluate()