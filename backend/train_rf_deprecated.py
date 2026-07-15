import pandas as pd
import scipy.sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_and_evaluate():
    print("--- INITIATING BASELINE MODEL TRAINING ---")
    
    # 1. Load the processed mathematical matrices and target variables
    print("Loading feature matrices...")
    X_train = scipy.sparse.load_npz('X_train_processed.npz')
    X_test = scipy.sparse.load_npz('X_test_processed.npz')
    
    y_train = pd.read_csv('y_train.csv').squeeze() # .squeeze() converts DataFrame column to Series
    y_test = pd.read_csv('y_test.csv').squeeze()
    
    # 2. Initialize the Random Forest Regressor
    # n_estimators=100 is the standard baseline. n_jobs=-1 uses all CPU cores for speed.
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 3. Train the Model
    print("Training Random Forest Regressor (This may take a few moments depending on your CPU)...")
    rf_model.fit(X_train, y_train)
    
    # 4. Make Predictions on the blind Test Set
    print("Evaluating on the Test Set...")
    predictions = rf_model.predict(X_test)
    
    # 5. Calculate Metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 6. Save the baseline model for the portal
    joblib.dump(rf_model, 'rf_baseline_model.pkl')
    
    # 7. Output Results
    print("\n=== BASELINE EVALUATION RESULTS ===")
    print(f"R-Squared (Accuracy Score): {r2:.4f}")
    print(f"Mean Absolute Error (MAE):  ₦{mae:,.2f}")
    print("===================================")
    print("Model saved as 'rf_baseline_model.pkl'.")

if __name__ == "__main__":
    train_and_evaluate()