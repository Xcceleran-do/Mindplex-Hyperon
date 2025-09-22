# model.py
# TensorFlow neural network training and inference for CCRS
# (FR-3.1, FR-3.2)

import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Configure logging (FR-5.2)
logging.basicConfig(
    filename='ccrs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

MODEL_PATH = os.path.join('models', 'nn_model.h5')

def _normalize(df, mean=None, std=None):
    """
    Normalize numeric columns using mean/std (manual, no scikit-learn).
    Returns normalized df, mean, std.
    """
    numeric = ['length']
    if mean is None or std is None:
        mean = df[numeric].mean()
        std = df[numeric].std().replace(0, 1)
    df[numeric] = (df[numeric] - mean) / std
    return df, mean, std

def train_model(data):
    """
    Train a feedforward NN to predict engagement (FR-3.1).
    Args:
        data: list of dicts from db.fetch_data()
    Returns:
        model, normalization params (mean, std), feature columns
    """
    logging.info("Starting model training.")
    df = pd.DataFrame(data)
    if len(df) < 50:
        logging.warning("Training on small dataset (<50 posts). Model quality may be poor.")
    # One-hot encode categorical features
    X = df[['length', 'topic', 'style', 'structure']]
    X = pd.get_dummies(X, columns=['topic', 'style', 'structure'])
    y = df['engagement_score'].values
    # Save feature columns for inference
    feature_cols = X.columns.tolist()
    # Normalize
    X, mean, std = _normalize(X)
    X = X.values.astype(np.float32)
    # Train/test split (manual, 80/20). For small datasets ensure there is at least one test sample when possible.
    if len(X) >= 5:
        idx = int(0.8 * len(X))
        if idx == len(X):
            idx = len(X) - 1
        X_train, X_test = X[:idx], X[idx:]
        y_train, y_test = y[:idx], y[idx:]
    else:
        # Too small for a reliable test split — use all data for training
        X_train, X_test = X, np.empty((0, X.shape[1]))
        y_train, y_test = y, np.empty((0,))
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.01, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])
    final_loss = history.history['loss'][-1]
    logging.info(f"Model trained. Final loss: {final_loss:.2f}")
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)  # <-- Only save weights/architecture
    logging.info(f"Model saved to {MODEL_PATH}")
    return model, mean, std, feature_cols

def predict_engagement(input_df, mean, std, feature_cols):
    """
    Predict engagement for new inputs (FR-3.2).
    Args:
        input_df: pandas DataFrame with columns matching training features
        mean, std: normalization params from training
        feature_cols: list of columns from training
    Returns:
        np.array of predictions
    """
    logging.info("Starting inference.")
    # One-hot encode to match training
    X = pd.get_dummies(input_df, columns=['topic', 'style', 'structure'])
    # Add missing columns (from training) as zeros
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    # Normalize
    X, _, _ = _normalize(X, mean, std)
    X = X.values.astype(np.float32)
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # <-- Don't compile on load
    preds = model.predict(X).flatten()
    logging.info(f"Inference complete. Predictions: {preds}")
    return preds

# Example usage for manual testing
if __name__ == "__main__":
    from db import fetch_data
    data = fetch_data()
    if not data:
        print("No data available for training.")
    else:
        if len(data) < 50:
            print("Warning: fewer than 50 posts available — training will proceed but results may be less reliable.")
        model, mean, std, feature_cols = train_model(data)
        print("Model trained and saved.")
        # Test prediction
        test_df = pd.DataFrame([{
            'length': 1000, 'topic': 'AI', 'style': 'Casual', 'structure': 'List'
        }])
        pred = predict_engagement(test_df, mean, std, feature_cols)
        print(f"Predicted engagement: {pred[0]:.2f}")

