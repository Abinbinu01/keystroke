# train_model.py - SIMPLIFIED VERSION (works with CSV only)
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATASET_FILE = 'keystroke_emotion_dataset.csv'
MODEL_FILE = 'keystroke_lstm_model.h5'

EMOTIONS = ['Happy', 'Sad', 'Calm', 'Stressed']
NUM_FEATURES = 5  # avg_key_hold, avg_flight_time, total_pauses, avg_pause_duration, wpm

def load_csv_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"No dataset found at {DATASET_FILE}")
        return None, None
    
    X, y = [], []
    with open(DATASET_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [
                    float(row['avg_key_hold']),
                    float(row['avg_flight_time']),
                    float(row['total_pauses']),
                    float(row['avg_pause_duration']),
                    float(row['wpm'])
                ]
                emotion = row['emotion_label']
                if emotion in EMOTIONS:
                    X.append(features)
                    y.append(EMOTIONS.index(emotion))
            except (ValueError, KeyError):
                continue
    
    return np.array(X), np.array(y)

def build_simple_model(input_features=5, num_classes=4):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_features,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    X, y = load_csv_dataset()
    
    if X is None or len(X) < 8:
        print("Need at least 8 samples (2 per emotion) to train.")
        print("Current dataset rows:", 0 if X is None else len(X))
        return
    
    print(f"Loaded {len(X)} samples")
    
    # Normalize features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X = (X - X_mean) / X_std
    
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_simple_model()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=8,
        verbose=1
    )
    
    # Save model + normalization params
    model.save(MODEL_FILE)
    np.savez('normalization_params.npz', mean=X_mean, std=X_std)
    
    print(f"✅ Model trained and saved: {MODEL_FILE}")
    print("🔄 Restart 'python app.py' to load the new model")

if __name__ == '__main__':
    main()
