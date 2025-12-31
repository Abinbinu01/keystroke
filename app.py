import os
import csv
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# Try TensorFlow first (LSTM), fallback if fails
TF_AVAILABLE = False
model = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow + LSTM loaded")
except ImportError:
    print("⚠️ TensorFlow not available - using rule-based fallback")

DATASET_FILE = 'keystroke_emotion_dataset.csv'
MODEL_FILE = 'keystroke_lstm_model.h5'
NORMALIZATION_FILE = 'normalization_params.npz'

app = Flask(__name__, static_folder='.', static_url_path='')
EMOTIONS = ['Happy', 'Sad', 'Calm', 'Stressed']

normalization_mean = np.array([0,0,0,0,0])
normalization_std = np.array([1,1,1,1,1])

def load_model_safely():
    global model, normalization_mean, normalization_std
    if TF_AVAILABLE and os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE)
            print(f"✅ LSTM model loaded: {MODEL_FILE}")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            model = None
    
    if os.path.exists(NORMALIZATION_FILE):
        try:
            params = np.load(NORMALIZATION_FILE)
            normalization_mean = params['mean']
            normalization_std = params['std']
            print("✅ Normalization loaded")
        except:
            pass

def ensure_dataset_file():
    header = ['avg_key_hold', 'avg_flight_time', 'total_pauses', 'avg_pause_duration', 'wpm', 'emotion_label']
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(header)

def extract_features_from_timing(timing_data, total_duration_ms, wpm):
    hold_times = []
    flight_times = []
    pause_durations = []
    for ev in timing_data:
        down = ev.get('downTime', 0)
        up = ev.get('upTime')
        flight = ev.get('flightTime', 0)
        if down is not None and up is not None:
            hold_times.append(up - down)
        if flight > 0:
            flight_times.append(flight)
            if flight > 1000:
                pause_durations.append(flight)
    avg_hold = np.mean(hold_times) if hold_times else 150.0
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

def lstm_prediction(features):
    """LSTM prediction if available"""
    global model, normalization_mean, normalization_std
    if model is None:
        return None
    
    try:
        feat_array = np.array([[features['avg_key_hold'], features['avg_flight_time'],
                              features['total_pauses'], features['avg_pause_duration'], features['wpm']]], dtype='float32')
        feat_array = (feat_array - normalization_mean) / normalization_std
        proba = model.predict(feat_array, verbose=0)
        idx = int(np.argmax(proba[0]))
        return EMOTIONS[idx]
    except:
        return None

def rule_based_prediction(features):
    """Fallback rules (all 4 emotions)"""
    wpm = features['wpm']
    hold = features['avg_key_hold']
    flight = features['avg_flight_time']
    pauses = features['total_pauses']
    
    if wpm > 55 and hold < 130 and pauses <= 1:
        return 'Happy'
    elif wpm < 35 and pauses >= 3:
        return 'Sad'
    elif pauses >= 5 or flight > 300:
        return 'Stressed'
    else:
        return 'Calm'

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
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    timing_data = data.get('timing_data', [])
    total_duration_ms = data.get('total_duration_ms', 0.0)
    wpm = data.get('wpm', 0.0)
    emotion_label = data.get('emotion_label')

    if not emotion_label or emotion_label not in EMOTIONS:
        return jsonify({'error': f'Invalid emotion'}), 400

    features = extract_features_from_timing(timing_data, total_duration_ms, wpm)
    
    # Save to CSV
    ensure_dataset_file()
    row = [features[k] for k in ['avg_key_hold', 'avg_flight_time', 'total_pauses', 'avg_pause_duration', 'wpm']] + [emotion_label]
    with open(DATASET_FILE, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

    # Predict: LSTM first, rules fallback
    predicted_emotion = lstm_prediction(features)
    if not predicted_emotion:
        predicted_emotion = rule_based_prediction(features)
        print("🔄 Used rule-based prediction")
    else:
        print("🧠 Used LSTM prediction")

    return jsonify({
        'status': 'success',
        'features': features,
        'predicted_emotion': predicted_emotion,
        'method': 'LSTM' if TF_AVAILABLE and model else 'Rules'
    })

if __name__ == '__main__':
    ensure_dataset_file()
    load_model_safely()
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 LSTM Emotion Detector ready on port {port}")
    print(f"   Model: {'✅ Loaded' if model else 'Rules fallback'}")
    app.run(host='0.0.0.0', port=port, debug=False)
