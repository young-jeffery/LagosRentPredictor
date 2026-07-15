import os
import random
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib

# 1. Lock the Python random number generator
random.seed(42)

# 2. Lock the Numpy random number generator
np.random.seed(42)

# 3. Lock the TensorFlow random number generator
tf.random.set_seed(42)

# 4. Optional: Force TensorFlow to use deterministic operations (Can slow down training slightly)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def objective(trial, X_train_scaled, y_train, X_test_scaled, y_test):
    # I am dynamically building my neural network architecture for Optuna to test
    input_dim = X_train_scaled.shape[1]
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    # Optuna will test between 64 and 512 neurons for my first hidden layer
    units_1 = trial.suggest_int('units_1', 64, 512)
    model.add(Dense(units_1, activation='relu'))
    
    # Optuna will test dropout rates to see how much regularization prevents overfitting
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.4)
    model.add(Dropout(dropout_1))
    
    # Testing the second hidden layer
    units_2 = trial.suggest_int('units_2', 32, 256)
    model.add(Dense(units_2, activation='relu'))
    dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
    model.add(Dropout(dropout_2))
    
    # Testing a third hidden layer
    units_3 = trial.suggest_int('units_3', 16, 128)
    model.add(Dense(units_3, activation='relu'))
    
    # Final output layer for my continuous regression price
    model.add(Dense(1, activation='linear'))
    
    # Optuna will hunt for the absolute best learning rate for the Adam optimizer
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Implementing early stopping so bad trials end quickly to save me time
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    
    # Fitting the model silently so my terminal doesn't get flooded with Keras progress bars
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Predicting and calculating the log-space error for Optuna to minimize
    preds_log = model.predict(X_test_scaled, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds_log))
    return rmse

def train_and_evaluate_ann():
    print("--- INITIATING OPTUNA TUNED ARTIFICIAL NEURAL NETWORK ---")
    
    # I am loading my newly sparse numpy arrays from the pipeline
    print("Loading feature matrices...")
    X_train_sparse = scipy.sparse.load_npz('X_train_processed.npz')
    X_test_sparse = scipy.sparse.load_npz('X_test_processed.npz')
    
    y_train = pd.read_csv('y_train.csv').squeeze().values
    y_test = pd.read_csv('y_test.csv').squeeze().values
    
    # I must convert the sparse matrices to dense arrays because Keras layers require them
    X_train_dense = X_train_sparse.toarray()
    X_test_dense = X_test_sparse.toarray()
    
    # I still need MinMaxScaler to protect my neural weights from large numerical disparities
    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    
    print("Starting Optuna Study (Hunting for the best Deep Learning architecture)...")
    study = optuna.create_study(direction='minimize')
    
    # Running 20 trials for the ANN since deep learning takes a bit longer to train
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=20)
    
    print("\n--- OPTUNA TUNING COMPLETE ---")
    print(f"Best Architecture Found: {study.best_params}")
    
    # I am manually reconstructing the absolute best architecture found by Optuna
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
    
    print("\nTraining Final Champion Neural Network...")
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
    
    # Reversing the Log Transformation safely back to Naira
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)
    
    r2 = r2_score(y_test_raw, predictions_raw)
    mae = mean_absolute_error(y_test_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
    mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
    # Saving my trained model and the scaler for deployment
    best_model.save('ann_tuned_model.keras')
    joblib.dump(scaler, 'ann_feature_scaler.pkl') 
    
    print("\n=== TUNED ARTIFICIAL NEURAL NETWORK RESULTS ===")
    print(f"1. R-Squared (R²): {r2:.4f}")
    print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
    print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
    print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("========================================")
    print("Saved 'ann_tuned_model.keras' and 'ann_feature_scaler.pkl'.")

if __name__ == "__main__":
    train_and_evaluate_ann()
























# Second implementation of the ANN training script. This version incorporates Optuna for hyperparameter tuning, which will allow us to find the best architecture and learning rate for our neural network. The workflow remains the same: loading the processed data, defining an objective function for Optuna, running the optimization process, and then evaluating the best model on the test set before saving it for future use. 
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping
# import optuna
# import joblib

# def objective(trial, X_train_scaled, y_train, X_test_scaled, y_test):
#     # I am dynamically building my neural network architecture for Optuna to test
#     input_dim = X_train_scaled.shape[1]
#     model = Sequential()
#     model.add(Input(shape=(input_dim,)))
    
#     # Optuna will test between 64 and 512 neurons for my first hidden layer
#     units_1 = trial.suggest_int('units_1', 64, 512)
#     model.add(Dense(units_1, activation='relu'))
    
#     # Optuna will test dropout rates to see how much regularization prevents overfitting
#     dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.4)
#     model.add(Dropout(dropout_1))
    
#     # Testing the second hidden layer
#     units_2 = trial.suggest_int('units_2', 32, 256)
#     model.add(Dense(units_2, activation='relu'))
#     dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
#     model.add(Dropout(dropout_2))
    
#     # Testing a third hidden layer
#     units_3 = trial.suggest_int('units_3', 16, 128)
#     model.add(Dense(units_3, activation='relu'))
    
#     # Final output layer for my continuous regression price
#     model.add(Dense(1, activation='linear'))
    
#     # Optuna will hunt for the absolute best learning rate for the Adam optimizer
#     lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
#     # Implementing early stopping so bad trials end quickly to save me time
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    
#     # Fitting the model silently so my terminal doesn't get flooded with Keras progress bars
#     model.fit(
#         X_train_scaled, y_train,
#         validation_split=0.2,
#         epochs=100,
#         batch_size=32,
#         callbacks=[early_stop],
#         verbose=0
#     )
    
#     # Predicting and calculating the log-space error for Optuna to minimize
#     preds_log = model.predict(X_test_scaled, verbose=0).flatten()
#     rmse = np.sqrt(mean_squared_error(y_test, preds_log))
#     return rmse

# def train_and_evaluate_ann():
#     print("--- INITIATING OPTUNA TUNED ARTIFICIAL NEURAL NETWORK ---")
    
#     # I am loading my newly dense numpy arrays
#     print("Loading feature matrices...")
#     X_train_dense = np.load('X_train_processed.npy')
#     X_test_dense = np.load('X_test_processed.npy')
    
#     y_train = pd.read_csv('y_train.csv').squeeze().values
#     y_test = pd.read_csv('y_test.csv').squeeze().values
    
#     # I still need MinMaxScaler to protect my neural weights from large numerical disparities
#     print("Scaling features using MinMaxScaler...")
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train_dense)
#     X_test_scaled = scaler.transform(X_test_dense)
    
#     print("Starting Optuna Study (Hunting for the best Deep Learning architecture)...")
#     study = optuna.create_study(direction='minimize')
    
#     # Running 20 trials for the ANN since deep learning takes a bit longer to train
#     study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=20)
    
#     print("\n--- OPTUNA TUNING COMPLETE ---")
#     print(f"Best Architecture Found: {study.best_params}")
    
#     # I am manually reconstructing the absolute best architecture found by Optuna
#     best_params = study.best_params
#     best_model = Sequential([
#         Input(shape=(X_train_scaled.shape[1],)),
#         Dense(best_params['units_1'], activation='relu'),
#         Dropout(best_params['dropout_1']),
#         Dense(best_params['units_2'], activation='relu'),
#         Dropout(best_params['dropout_2']),
#         Dense(best_params['units_3'], activation='relu'),
#         Dense(1, activation='linear')
#     ])
    
#     best_model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), 
#         loss='mse'
#     )
    
#     early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
#     print("\nTraining Final Champion Neural Network...")
#     best_model.fit(
#         X_train_scaled, y_train,
#         validation_split=0.2,
#         epochs=150,
#         batch_size=32,
#         callbacks=[early_stop],
#         verbose=1
#     )
    
#     print("\nEvaluating on the Test Set...")
#     predictions_log = best_model.predict(X_test_scaled).flatten()
    
#     # Reversing the Log Transformation safely back to Naira
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # Saving my trained model and the scaler for deployment
#     best_model.save('ann_tuned_model.keras')
#     joblib.dump(scaler, 'ann_feature_scaler.pkl') 
    
#     print("\n=== TUNED ARTIFICIAL NEURAL NETWORK RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Saved 'ann_tuned_model.keras' and 'ann_feature_scaler.pkl'.")

# if __name__ == "__main__":
#     train_and_evaluate_ann()






















# import pandas as pd
# import numpy as np
# import scipy.sparse
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping
# import joblib

# def train_and_evaluate_ann():
#     print("--- INITIATING ARTIFICIAL NEURAL NETWORK TRAINING ---")
    
#     # 1. Load the matrices and targets
#     print("Loading feature matrices...")
#     X_train_sparse = scipy.sparse.load_npz('X_train_processed.npz')
#     X_test_sparse = scipy.sparse.load_npz('X_test_processed.npz')
    
#     y_train = pd.read_csv('y_train.csv').squeeze().values
#     y_test = pd.read_csv('y_test.csv').squeeze().values
    
#     # 2. Convert to Dense Arrays
#     X_train_dense = X_train_sparse.toarray()
#     X_test_dense = X_test_sparse.toarray()
    
#     # 3. FIXED: Using MinMaxScaler instead of StandardScaler to protect the sparse 0/1 data
#     print("Scaling features using MinMaxScaler...")
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train_dense)
#     X_test_scaled = scaler.transform(X_test_dense)
    
#     # 4. Build the Neural Network Architecture
#     print("Building Deep Learning Architecture...")
#     input_dim = X_train_scaled.shape[1]
    
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(256, activation='relu'),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dropout(0.2),
#         Dense(64, activation='relu'),
#         Dense(1, activation='linear')
#     ])
    
#     # 5. Compile the Model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
#     # 6. Setup Early Stopping
#     early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
#     # 7. Train the Model
#     print("Training Neural Network... (This may take a few moments)")
#     model.fit(
#         X_train_scaled, y_train,
#         validation_split=0.2,
#         epochs=150,
#         batch_size=32,
#         callbacks=[early_stop],
#         verbose=1
#     )
    
#     # 8. Evaluate on the Test Set
#     print("\nEvaluating on the Test Set...")
#     predictions_log = model.predict(X_test_scaled).flatten()
    
#     # 9. Reverse the Log Transformation safely
#     predictions_raw = np.expm1(predictions_log)
#     y_test_raw = np.expm1(y_test)
    
#     # Calculate final metrics
#     r2 = r2_score(y_test_raw, predictions_raw)
#     mae = mean_absolute_error(y_test_raw, predictions_raw)
#     rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
#     mape = mean_absolute_percentage_error(y_test_raw, predictions_raw) * 100
    
#     # 10. Save the models
#     model.save('ann_baseline_model.keras')
#     joblib.dump(scaler, 'ann_feature_scaler.pkl') 
    
#     print("\n=== ARTIFICIAL NEURAL NETWORK RESULTS ===")
#     print(f"1. R-Squared (R²): {r2:.4f}")
#     print(f"2. Mean Absolute Error (MAE): ₦{mae:,.2f}")
#     print(f"3. Root Mean Squared Error (RMSE): ₦{rmse:,.2f}")
#     print(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
#     print("========================================")
#     print("Saved 'ann_baseline_model.keras' and 'ann_feature_scaler.pkl'.")

# if __name__ == "__main__":
#     train_and_evaluate_ann()