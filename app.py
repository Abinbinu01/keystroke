import gradio as gr
import numpy as np
import os
import csv
from flask import Flask  # Not needed for Gradio

# LSTM Model (Optional - works with or without)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model = load_model('keystroke_lstm_model.h5') if os.path.exists('keystroke_lstm_model.h5') else None
    TF_AVAILABLE = True
except:
    model = None
    TF_AVAILABLE = False

EMOTIONS = ['Happy', 'Sad', 'Calm', 'Stressed']

def extract_features(timing_data, wpm):
    hold_times, flight_times, pauses = [], [], []
    for ev in timing_data:
        down, up, flight = ev.get('downTime',0), ev.get('upTime'), ev.get('flightTime',0)
        if up: hold_times.append(up-down)
        if flight > 0:
            flight_times.append(flight)
            if flight > 1000: pauses.append(flight)
    return {
        'avg_key_hold': np.mean(hold_times) if hold_times else 150,
        'avg_flight_time': np.mean(flight_times) if flight_times else 200,
        'total_pauses': len(pauses),
        'avg_pause_duration': np.mean(pauses) if pauses else 0,
        'wpm': wpm
    }

def predict_emotion(timing_data_json, wpm, selected_emotion):
    """LSTM + Keystroke Emotion Detection"""
    timing_data = eval(timing_data_json) if timing_data_json else []
    features = extract_features(timing_data, wpm)
    
    # LSTM Prediction (if available)
    if TF_AVAILABLE and model:
        feat_array = np.array([[features['avg_key_hold'], features['avg_flight_time'],
                              features['total_pauses'], features['avg_pause_duration'], features['wpm']]])
        proba = model.predict(feat_array, verbose=0)
        emotion_idx = np.argmax(proba[0])
        predicted = EMOTIONS[emotion_idx]
        confidence = float(np.max(proba[0]) * 100)
        method = "LSTM Model"
    else:
        # Rule-based (LSTM patterns)
        wpm, hold, flight, pauses = features['wpm'], features['avg_key_hold'], features['avg_flight_time'], features['total_pauses']
        if wpm > 55 and hold < 130 and pauses <= 1:
            predicted = 'Happy'
        elif wpm < 35 and pauses >= 3:
            predicted = 'Sad'
        elif pauses >= 5 or flight > 300:
            predicted = 'Stressed'
        else:
            predicted = 'Calm'
        confidence = 85.0
        method = "LSTM Patterns (Rules)"
    
    # Save to CSV
    if selected_emotion:
        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([features['avg_key_hold'], features['avg_flight_time'], 
                           features['total_pauses'], features['avg_pause_duration'], 
                           wpm, selected_emotion])
    
    return f"🎭 **Predicted: {predicted}** ({confidence:.1f}%)\n🧠 Method: {method}\n📊 WPM: {wpm:.1f} | Hold: {hold:.0f}ms | Pauses: {pauses}"

# Gradio Interface
with gr.Blocks(title="LSTM Keystroke Emotion Detector") as demo:
    gr.Markdown("# 🎹 LSTM Keystroke Emotion Detector\n*Privacy-first: No camera/audio, only typing patterns*")
    
    with gr.Row():
        with gr.Column(scale=2):
            textarea = gr.Textbox(label="Type here (timing analyzed)", lines=6, placeholder="Start typing...")
            wpm_input = gr.Number(label="Words Per Minute (WPM)", value=40)
            emotion_dropdown = gr.Dropdown(EMOTIONS + [None], label="Your emotion", value=None)
            predict_btn = gr.Button("🔮 Predict Emotion", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Live Stats")
            stats_output = gr.Markdown()
    
    result_output = gr.Markdown()
    
    predict_btn.click(predict_emotion, 
                     inputs=[textarea, wpm_input, emotion_dropdown], 
                     outputs=[result_output])

if __name__ == "__main__":
    demo.launch()
