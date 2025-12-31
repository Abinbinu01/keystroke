# app.py - COMPLETE CORRECTED VERSION
# Flask backend for privacy-centric keystroke emotion detection.
# Works with simplified Dense model trained from CSV features.

import os
import csv
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

import tensorflow as tf
from tensorflow.keras.models import load_model

DATASET_FILE = 'keystroke_emotion_dataset.csv'
MODEL_FILE = 'keystroke_lstm_model.h5'

app = Flask(__name__, static_folder='.', static_url_path='')

# Global model and normalization params
model = None
normalization_mean = np.array([0,0,0,0,0])
normalization_std = np.array([1,1,1,1,1])

EMOTIONS = ['Happy', 'Sad', 'Calm', 'Stressed']

def load_trained_model():
    global model, normalization_mean, normalization_std
    
    # Load model if exists
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        print(f"✅ Model loaded: {MODEL_FILE}")
    else:
        model = None
        print("⚠️ No model found. Collect data and run train_model.py")
    
    # Load normalization params if exists
    if os.path.exists('normalization_params.npz'):
        params = np.load('normalization_params.npz')
        normalization_mean = params['mean']
        normalization_std = params['std']
        print("✅ Normalization params loaded")

def ensure_dataset_file():
    """Create CSV with headers if not exists"""
    header = ['avg_key_hold', 'avg_flight_time', 'total_pauses',
              'avg_pause_duration', 'wpm', 'emotion_label']
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"✅ Dataset initialized: {DATASET_FILE}")

def extract_features_from_timing(timing_data, total_duration_ms, wpm):
    """Extract keystroke features from timing data"""
    hold_times = []
    flight_times = []
    pause_durations = []

    for ev in timing_data:
        down = ev.get('downTime', 0)
        up = ev.get('upTime')
        flight = ev.get('flightTime', 0)

        # Key hold duration (dwell time)
        if down is not None and up is not None:
            hold_times.append(up - down)

        # Flight time between keys
        if flight is not None and flight > 0:
            flight_times.append(flight)
            if flight > 1000.0:  # Pause > 1 second
                pause_durations.append(flight)

    # Compute averages
    avg_hold = np.mean(hold_times) if hold_times else 150.0  # Default
    avg_flight = np.mean(flight_times) if flight_times else 200.0
    total_pauses = len(pause_durations)
    avg_pause = np.mean(pause_durations) if pause_durations else 0.0

    return {
        'avg_key_hold': float(avg_hold),
        'avg_flight_time': float(avg_flight),
        'total_pauses': int(total_pauses),
        'avg_pause_duration': float(avg_pause),
        'wpm': float(wpm)
    }

def predict_from_features(features):
    """Predict emotion using trained model"""
    global model, normalization_mean, normalization_std
    
    if model is None:
        return None
    
    try:
        # Prepare feature array [1 sample x 5 features]
        feat_array = np.array([[
            features['avg_key_hold'],
            features['avg_flight_time'],
            features['total_pauses'],
            features['avg_pause_duration'],
            features['wpm']
        ]], dtype='float32')
        
        # Normalize using training params
        feat_array = (feat_array - normalization_mean) / normalization_std
        
        # Predict
        proba = model.predict(feat_array, verbose=0)
        idx = int(np.argmax(proba[0]))
        
        confidence = float(np.max(proba[0]) * 100)
        predicted = EMOTIONS[idx] if 0 <= idx < len(EMOTIONS) else None
        
        print(f"Prediction: {predicted} ({confidence:.1f}% confidence)")
        return predicted
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def style_css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script_js():
    return send_from_directory('.', 'script.js')

@app.route('/api/submit_keystrokes', methods=['POST'])
def submit_keystrokes():
    """Main API: Receive timing data, extract features, save to CSV, predict"""
    
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON'}), 400

        timing_data = data.get('timing_data', [])
        total_duration_ms = data.get('total_duration_ms', 0.0)
        wpm = data.get('wpm', 0.0)
        emotion_label = data.get('emotion_label')

        if not emotion_label or emotion_label not in EMOTIONS:
            return jsonify({'error': f'Invalid emotion. Must be: {EMOTIONS}'}), 400

        # Extract features
        features = extract_features_from_timing(timing_data, total_duration_ms, wpm)
        
        # Save to CSV dataset
        ensure_dataset_file()
        row = [
            features['avg_key_hold'],
            features['avg_flight_time'],
            features['total_pauses'],
            features['avg_pause_duration'],
            features['wpm'],
            emotion_label
        ]
        
        with open(DATASET_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"✅ Saved sample #{os.path.getsize(DATASET_FILE)//1000}: {features}")

        # Predict emotion
        predicted_emotion = predict_from_features(features)

        return jsonify({
            'status': 'success',
            'features': features,
            'dataset_rows': os.path.getsize(DATASET_FILE)//1000,
            'predicted_emotion': predicted_emotion,
            'message': 'Data saved successfully!'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset_info')
def dataset_info():
    """Check dataset status"""
    if os.path.exists(DATASET_FILE):
        size = os.path.getsize(DATASET_FILE)
        return jsonify({
            'dataset_exists': True,
            'file_size_bytes': size,
            'approx_rows': size // 1000,
            'model_exists': os.path.exists(MODEL_FILE)
        })
    return jsonify({'dataset_exists': False})

if __name__ == '__main__':
    # Initialize on startup
    ensure_dataset_file()
    load_trained_model()
    print("🚀 Server ready at http://127.0.0.1:5000")
    print("📊 Dataset:", DATASET_FILE)
    print("🧠 Model:", MODEL_FILE if model else "Not trained yet")
    app.run(debug=True, host='0.0.0.0', port=5000)
