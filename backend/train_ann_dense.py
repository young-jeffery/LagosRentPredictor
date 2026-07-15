import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import optuna

def objective(trial, X_train_scaled, y_train, X_test_scaled, y_test):
    # I am letting Optuna dynamically build my deep learning architecture to find the absolute best combination of neurons
    input_dim = X_train_scaled.shape[1]
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    units_1 = trial.suggest_int('units_1', 64, 512)
    model.add(Dense(units_1, activation='relu'))
    
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.4)
    model.add(Dropout(dropout_1))
    
    units_2 = trial.suggest_int('units_2', 32, 256)
    model.add(Dense(units_2, activation='relu'))
    
    dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
    model.add(Dropout(dropout_2))
    
    units_3 = trial.suggest_int('units_3', 16, 128)
    model.add(Dense(units_3, activation='relu'))
    
    model.add(Dense(1, activation='linear'))
    
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # I am using early stopping so my trials don't waste time on failing architectures
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    preds_log = model.predict(X_test_scaled, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds_log))
    return rmse

def train_and_evaluate_dense_ann():
    print("--- INITIATING DENSE OPTUNA TUNED NEURAL NETWORK ---")
    
    # I am loading my newly created dense numpy arrays directly
    print("Loading dense feature matrices...")
    X_train_dense = np.load('X_train_dense.npy')
    X_test_dense = np.load('X_test_dense.npy')
    
    y_train = pd.read_csv('y_train_dense.csv').squeeze().values
    y_test = pd.read_csv('y_test_dense.csv').squeeze().values
    
    # I am using MinMaxScaler to keep my feature values between 0 and 1 so the network weights calculate smoothly
    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    
    print("Starting Optuna Study (Hunting for the best Deep Learning architecture)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=20)
    
    print("\n--- OPTUNA TUNING COMPLETE ---")
    print(f"Best Architecture Found: {study.best_params}")
    
    # I am manually reconstructing the exact best architecture returned by Optuna for my final training run
    best_params = study.best_params
    best_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(best_params['units_1'], activation='relu'),
        Dropout(best_params['dropout_1']),
        Dense(best_params['units_2'], activation='relu'),
        Dropout(best_params['dropout_2']),
        Dense(best_params['units_3'], activation='relu'),
        Dense(1, activation='linear')
    ])
    
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), 
        loss='mse'
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print("\nTraining Final Champion Dense Neural Network...")
    best_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\nEvaluating on the Test Set...")
    predictions_log = best_model.predict(X_test_scaled).flatten()
    
    # I must reverse the np.log1p transformation so my metrics reflect actual Naira values
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)
    
    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
    print("\n=== DENSE ARTIFICIAL NEURAL NETWORK RESULTS ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")

if __name__ == "__main__":
    train_and_evaluate_dense_ann()