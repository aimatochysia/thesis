
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, jsonify, request

try:
    import onnxruntime as ort
    USE_ONNX = True
except ImportError:
    print("Error: ONNXRuntime not found.")
    print("Install ONNXRuntime for ONNX model inference: pip install onnxruntime")
    sys.exit(1)

MODEL_CONFIGS = {
    'v2': {
        'name': 'CNN (v2)',
        'onnx_file': 'ecg_cnn_v2_pytorch_final.onnx',
        'scaler_file': 'scaler_v2_pytorch.pkl',
        'input_shape': (1, 1, 188),
        'beat_length': 188,
        'context_aware': False,
    },
    'v3': {
        'name': 'LSTM (v3)',
        'onnx_file': 'ecg_lstm_v3_pytorch_final.onnx',
        'scaler_file': 'scaler_v3_pytorch.pkl',
        'input_shape': (1, 188, 1),
        'beat_length': 188,
        'context_aware': False,
    },
    'v5': {
        'name': 'Transformer (v5)',
        'onnx_file': 'ecg_transformer_v5_pytorch_final.onnx',
        'scaler_file': 'scaler_v5_pytorch.pkl',
        'input_shape': (1, 188, 1),
        'beat_length': 188,
        'context_aware': False,
    },
    'v6': {
        'name': 'Context-Aware CNN1D (v6)',
        'onnx_file': 'context_ecg_model.onnx',
        'scaler_file': 'context_ecg_scaler.pkl',
        'input_shape': (1, 7, 200),
        'beat_length': 200,
        'context_aware': True,
        'context_window_size': 7,
        'pre_r_samples': 90,
        'post_r_samples': 110,
    },
}

BEAT_LENGTH = 188
BEAT_LENGTH_V6 = 200
PRE_SAMPLES = 70
POST_SAMPLES = 118
PRE_SAMPLES_V6 = 90
POST_SAMPLES_V6 = 110
CONTEXT_WINDOW_SIZE = 7
SAMPLING_RATE = 360
NORMAL_BEAT_TYPE = 'N'

app = Flask(__name__)
ecg_data = None
annotations = None
model = None
scaler = None
model_config = None
current_sample = 0
classification_results = []
is_running = False
speed_multiplier = 10

beat_buffer = []


def load_data(model_version='v3', use_training_data=False, use_record_119=True):
    global ecg_data, annotations, model, scaler, model_config, beat_buffer
    
    beat_buffer = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    print(f"{MODEL_CONFIGS[model_version]['name']}: Using record 119 (excluded from training) for validation")
    
    if use_record_119:
        signal_path = os.path.join(sample_dir, '119.csv')
        annotation_path = os.path.join(sample_dir, '119annotations.txt')
        print("Using MIT-BIH record 119 (excluded from training - true test data)")
    elif use_training_data:
        signal_path = os.path.join(sample_dir, 'demo_training_signal.csv')
        annotation_path = os.path.join(sample_dir, 'demo_training_annotations.txt')
        if not os.path.exists(signal_path):
            print("Warning: Training demo data not found, falling back to record 119")
            signal_path = os.path.join(sample_dir, '119.csv')
            annotation_path = os.path.join(sample_dir, '119annotations.txt')
    else:
        signal_path = os.path.join(sample_dir, '119.csv')
        annotation_path = os.path.join(sample_dir, '119annotations.txt')
        print("Using MIT-BIH record 119 (excluded from training - true test data)")
    
    df = pd.read_csv(signal_path)
    df.columns = df.columns.str.strip().str.strip("'")
    ecg_data = df['MLII'].values.astype(np.float32)
    
    annotations_list = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                sample_idx = int(parts[1])
                beat_type = parts[2]
                time_str = parts[0]
                annotations_list.append({
                    'sample_index': sample_idx,
                    'beat_type': beat_type,
                    'time': time_str
                })
            except (ValueError, IndexError):
                continue
    annotations = pd.DataFrame(annotations_list)
    
    if model_version not in MODEL_CONFIGS:
        print(f"Unknown model version '{model_version}'. Using v3 (LSTM) as default.")
        model_version = 'v3'
    
    model_config = MODEL_CONFIGS[model_version]
    print(f"\nLoading {model_config['name']} model...")
    
    onnx_model_path = os.path.join(sample_dir, model_config['onnx_file'])
    if os.path.exists(onnx_model_path):
        print(f"Loading ONNX model from: {onnx_model_path}")
        model = ort.InferenceSession(onnx_model_path)
        print(f"✓ {model_config['name']} ONNX model loaded successfully")
    else:
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    scaler_path = os.path.join(sample_dir, model_config['scaler_file'])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    print(f"\nLoaded {len(ecg_data)} ECG samples")
    print(f"Loaded {len(annotations)} annotations")


def extract_beat_v6(signal, r_peak_idx):
    start_idx = r_peak_idx - PRE_SAMPLES_V6
    end_idx = r_peak_idx + POST_SAMPLES_V6
    
    if start_idx < 0:
        pad_before = -start_idx
        beat = np.zeros(BEAT_LENGTH_V6, dtype=np.float32)
        available = signal[:end_idx]
        beat[pad_before:pad_before + len(available)] = available
    elif end_idx > len(signal):
        beat = np.zeros(BEAT_LENGTH_V6, dtype=np.float32)
        available = signal[start_idx:]
        beat[:len(available)] = available
    else:
        beat = signal[start_idx:end_idx].astype(np.float32)
    
    return beat


def extract_and_classify_beat(signal, r_peak_idx, beat_type):
    global beat_buffer
    
    is_context_aware = model_config.get('context_aware', False)
    
    if is_context_aware:
        beat = extract_beat_v6(signal, r_peak_idx)
        raw_beat = beat.copy()
        
        beat_buffer.append((beat, beat_type))
        
        if len(beat_buffer) > CONTEXT_WINDOW_SIZE:
            beat_buffer = beat_buffer[-CONTEXT_WINDOW_SIZE:]
        
        if len(beat_buffer) < CONTEXT_WINDOW_SIZE:
            return {
                'r_peak': r_peak_idx,
                'beat_type': beat_type,
                'ground_truth': "NORMAL" if beat_type == NORMAL_BEAT_TYPE else "ABNORMAL",
                'predicted': "WAITING",
                'probability': 0.0,
                'correct': None,
                'beat_waveform': raw_beat.tolist(),
                'buffer_size': len(beat_buffer),
                'context_aware': True
            }
        
        context_beats = np.stack([b for b, _ in beat_buffer], axis=0)
        
        flat_size = CONTEXT_WINDOW_SIZE * BEAT_LENGTH_V6
        context_flat = context_beats.reshape(1, flat_size)
        
        normalized = scaler.transform(context_flat).astype(np.float32)
        
        context_input = normalized.reshape(1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH_V6)
        
        center_beat_type = beat_buffer[3][1]
        
    else:
        start_idx = r_peak_idx - PRE_SAMPLES
        end_idx = r_peak_idx + POST_SAMPLES
        
        if start_idx < 0:
            pad_before = -start_idx
            beat = np.zeros(BEAT_LENGTH, dtype=np.float32)
            available = signal[:end_idx]
            beat[pad_before:pad_before + len(available)] = available
        elif end_idx > len(signal):
            beat = np.zeros(BEAT_LENGTH, dtype=np.float32)
            available = signal[start_idx:]
            beat[:len(available)] = available
        else:
            beat = signal[start_idx:end_idx].astype(np.float32)
        
        raw_beat = beat.copy()
        
        beat_2d = beat.reshape(1, -1)
        normalized = scaler.transform(beat_2d).flatten().astype(np.float32)
        
        input_shape = model_config['input_shape']
        context_input = normalized.reshape(input_shape)
        center_beat_type = beat_type
    
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: context_input})[0]
    
    if output.shape[1] == 2:
        needs_softmax = (np.min(output) < 0 or np.max(output) > 1 or 
                         abs(np.sum(output[0]) - 1.0) > 0.01)
        if needs_softmax:
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            proba = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        else:
            proba = output
        prob_abnormal = float(proba[0, 1])
    else:
        prob_abnormal = float(output[0, 0])
    
    prob_abnormal = max(0.0, min(1.0, prob_abnormal))
    
    predicted_class = 1 if prob_abnormal >= 0.5 else 0
    predicted_label = "ABNORMAL" if predicted_class == 1 else "NORMAL"
    
    if center_beat_type == NORMAL_BEAT_TYPE:
        ground_truth = "NORMAL"
    else:
        ground_truth = "ABNORMAL"
    
    if is_context_aware:
        r_peak_pos_in_beat = PRE_SAMPLES_V6
    else:
        r_peak_pos_in_beat = PRE_SAMPLES
    
    result = {
        'r_peak': r_peak_idx,
        'beat_type': center_beat_type,
        'ground_truth': ground_truth,
        'predicted': predicted_label,
        'probability': round(prob_abnormal, 4),
        'correct': ground_truth == predicted_label,
        'beat_waveform': raw_beat.tolist(),
        'r_peak_pos_in_beat': r_peak_pos_in_beat,
        'beat_length': BEAT_LENGTH_V6 if is_context_aware else BEAT_LENGTH
    }
    
    if is_context_aware:
        result['context_aware'] = True
        result['buffer_size'] = len(beat_buffer)
    
    return result


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Real-Time Classification</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
        }
        #startBtn {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #1a1a2e;
        }
        #stopBtn {
            background: linear-gradient(45deg, #ff4757, #ff3838);
            color: white;
        }
        #resetBtn {
            background: linear-gradient(45deg, #5352ed, #3742fa);
            color: white;
        }
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        }
        .stats-bar {
            display: flex;
            justify-content: space-around;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00ff88;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .ecg-container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        #ecgCanvas {
            width: 100%;
            min-height: 300px;
            height: 300px;
            background: #0a0a1a;
            border-radius: 10px;
            transition: height 0.3s ease;
        }
        .time-display {
            text-align: center;
            font-size: 24px;
            color: #00ff88;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
        }
        .results-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .panel h3 {
            margin-bottom: 15px;
            color: #00ff88;
            border-bottom: 1px solid rgba(0, 255, 136, 0.3);
            padding-bottom: 10px;
        }
        .classification-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .classification-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .classification-item.normal {
            background: rgba(0, 255, 136, 0.2);
            border-left: 4px solid #00ff88;
        }
        .classification-item.abnormal {
            background: rgba(255, 71, 87, 0.2);
            border-left: 4px solid #ff4757;
        }
        .beat-info {
            font-size: 14px;
        }
        .beat-time {
            color: #888;
            font-size: 12px;
        }
        .prediction-badge {
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .prediction-badge.normal {
            background: #00ff88;
            color: #1a1a2e;
        }
        .prediction-badge.abnormal {
            background: #ff4757;
            color: white;
        }
        .current-beat {
            text-align: center;
            padding: 30px;
        }
        .current-beat .label {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
        }
        .current-beat .value {
            font-size: 48px;
            font-weight: bold;
        }
        .current-beat .value.normal {
            color: #00ff88;
        }
        .current-beat .value.abnormal {
            color: #ff4757;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .probability-bar {
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin-top: 15px;
            overflow: hidden;
        }
        .probability-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .speed-control {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #888;
            background: rgba(0,0,0,0.3);
            padding: 8px 12px;
            border-radius: 20px;
        }
        .speed-btn {
            padding: 5px 10px;
            font-size: 12px;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            cursor: pointer;
        }
        .speed-btn.active {
            background: rgba(0,255,136,0.3);
            border-color: #00ff88;
        }
        .speed-btn:hover {
            background: rgba(0,255,136,0.2);
        }
        .model-badge {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #1a1a2e;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ECG Real-Time Classification Monitor</h1>
        <p style="text-align: center; color: #888; margin-bottom: 15px;">
            Using PyTorch ONNX Model: <span id="modelName" class="model-badge">Loading...</span>
        </p>
        
        <div class="controls">
            <button id="startBtn" onclick="startSimulation()">▶ Start</button>
            <button id="stopBtn" onclick="stopSimulation()">⏹ Stop</button>
            <button id="resetBtn" onclick="resetSimulation()">🔄 Reset</button>
            <div class="speed-control">
                <span>Speed:</span>
                <button class="speed-btn" onclick="setSpeed(0.1)">0.1x</button>
                <button class="speed-btn" onclick="setSpeed(0.5)">0.5x</button>
                <button class="speed-btn active" onclick="setSpeed(1)">1x</button>
                <button class="speed-btn" onclick="setSpeed(5)">5x</button>
                <button class="speed-btn" onclick="setSpeed(10)">10x</button>
                <span id="speedValue">1x</span>
            </div>
        </div>
        
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value" id="totalBeats">0</div>
                <div class="stat-label">Total Beats</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="normalBeats" style="color: #00ff88;">0</div>
                <div class="stat-label">Normal</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="abnormalBeats" style="color: #ff4757;">0</div>
                <div class="stat-label">Abnormal</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="accuracy">--</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="heartRate">--</div>
                <div class="stat-label">BPM</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="falseCount" style="color: #ffd700;">0</div>
                <div class="stat-label">False Predictions</div>
            </div>
        </div>
        
        <div class="ecg-container">
            <canvas id="ecgCanvas"></canvas>
            <div class="time-display">
                Time: <span id="currentTime">0:00.000</span>
                <span id="historyIndicator" style="display: none; margin-left: 15px; background: rgba(255,215,0,0.2); color: #ffd700; padding: 3px 10px; border-radius: 10px; font-size: 12px;">📜 Viewing History</span>
            </div>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <button onclick="scrollHistory(-5)" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">⏪ -5s</button>
                <button onclick="scrollHistory(-1)" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">◀ -1s</button>
                <button id="liveBtn" onclick="goToLive()" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: linear-gradient(45deg, #ffd700, #ffb700); border: none; color: #1a1a2e; font-weight: bold; cursor: pointer;">🔴 Live</button>
                <button id="fwdBtn" onclick="scrollHistory(1)" disabled style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">▶ +1s</button>
                <button id="fwd5Btn" onclick="scrollHistory(5)" disabled style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">⏩ +5s</button>
                <span style="margin: 0 10px; color: #444;">|</span>
                <button onclick="downloadAllBatches()" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: #00ff88; cursor: pointer;" title="Download all batches as ZIP">📦 Download Batches (ZIP)</button>
            </div>
            <div id="batchStatus" style="text-align: center; margin-top: 8px; font-size: 12px;">
                <span style="color: #888;">📦 Auto-saves every 2 min | Click button to download as ZIP</span>
            </div>
            <p style="text-align: center; color: #666; font-size: 11px; margin-top: 5px;">💡 Drag the graph to scroll through history | Batches auto-save, download ZIP when ready</p>
        </div>
        
        <!-- Beat Snapshot Panel - Shows the current beat segment sent to ONNX model -->
        <div class="beat-snapshot-container" style="background: rgba(0, 0, 0, 0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(0, 255, 136, 0.3);">
            <h3 style="color: #00ff88; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 255, 136, 0.3); padding-bottom: 10px;">Current Beat Snapshot (Input to ONNX Model)</h3>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="flex: 1;">
                    <canvas id="beatCanvas" style="width: 100%; height: 150px; background: #0a0a1a; border-radius: 10px;"></canvas>
                </div>
                <div style="min-width: 200px; text-align: center;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 5px;">Beat Type (Annotation)</div>
                    <div id="beatTypeDisplay" style="font-size: 24px; font-weight: bold; color: #00ff88;">--</div>
                    <div style="color: #888; font-size: 12px; margin-top: 10px;">Ground Truth</div>
                    <div id="groundTruthDisplay" style="font-size: 18px; font-weight: bold; color: #00ff88;">--</div>
                    <div style="color: #888; font-size: 12px; margin-top: 10px;">Model Prediction</div>
                    <div id="predictionDisplay" style="font-size: 18px; font-weight: bold; color: #00ff88;">--</div>
                </div>
            </div>
            <div style="text-align: center; color: #666; font-size: 11px; margin-top: 10px;">
                188 samples extracted around R-peak → Normalized with scaler → Fed to ONNX model → Output: 0=Normal, 1=Abnormal
            </div>
        </div>
        
        <div class="results-container">
            <div class="panel">
                <h3>Current Classification</h3>
                <div class="current-beat">
                    <div class="label">Latest Heartbeat Status</div>
                    <div class="value" id="currentStatus">Waiting...</div>
                    <div class="probability-bar">
                        <div class="probability-fill" id="probBar" style="width: 0%; background: #00ff88; display: none;"></div>
                    </div>
                    <div id="probText" style="margin-top: 10px; display: none; color: #888;">Abnormal Probability: --</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Classification History</h3>
                <div class="classification-list" id="classificationList">
                    <p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>
                </div>
            </div>
            
            <div class="panel" style="border-left: 4px solid #ffd700;">
                <h3 style="color: #ffd700 !important;">⚠️ False Detections</h3>
                <div class="classification-list" id="falseDetectionList">
                    <p style="color: #888; text-align: center;">No false detections yet.</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('ecgCanvas');
        const ctx = canvas.getContext('2d');
        
        // Beat snapshot canvas
        const beatCanvas = document.getElementById('beatCanvas');
        const beatCtx = beatCanvas.getContext('2d');
        
        let ecgData = [];
        let annotations = [];
        let currentIndex = 0;
        let isRunning = false;
        let animationId = null;
        let displayBuffer = [];
        let classifications = [];
        let falseDetections = [];
        let beatTimes = [];  // Store recent beat times for BPM calculation
        let speedMultiplier = 1;  // 1x = real-time
        let currentBeatWaveform = null;
        let currentRPeakPos = 70;  // R-peak position in beat waveform
        let currentBeatLength = 188;  // Beat length
        
        // Graph height tracking - expand but never shrink for better readability
        let maxGraphHeight = 300;  // Track maximum height achieved
        const MIN_GRAPH_HEIGHT = 300;  // Minimum height
        const MAX_GRAPH_HEIGHT = 800;  // Maximum allowed height
        
        // Y-axis scale tracking - expand to fit largest signal seen, but never shrink
        // This ensures consistent vertical scale across the entire recording
        let globalMinVal = Infinity;   // Track minimum value seen across all data
        let globalMaxVal = -Infinity;  // Track maximum value seen across all data
        
        // History navigation
        let viewOffset = 0;  // 0 = live view, negative = viewing history
        let isLive = true;
        
        // High-speed stability: track pending classification requests
        let isClassifying = false;
        let classificationQueue = [];  // Queue for pending beats to classify
        let processedBeats = new Set();  // Track already processed beats to avoid duplicates
        const MAX_CLASSIFICATIONS = 1000;  // Limit stored classifications to prevent memory issues
        const MAX_FALSE_DETECTIONS = 100;  // Limit stored false detections
        
        const SAMPLING_RATE = 360;
        const DISPLAY_SECONDS = 5;
        const DISPLAY_SAMPLES = SAMPLING_RATE * DISPLAY_SECONDS;
        
        // ============================================================
        // AUTO-BATCH EXPORT SYSTEM
        // ============================================================
        // Automatically saves batches during recording for faster export
        const AUTO_BATCH_INTERVAL_SECONDS = 120;  // Auto-save every 2 minutes
        const AUTO_BATCH_INTERVAL_SAMPLES = AUTO_BATCH_INTERVAL_SECONDS * SAMPLING_RATE;
        const MIN_BATCH_SECONDS = 5;  // Minimum seconds for a batch
        const MIN_BATCH_SAMPLES = MIN_BATCH_SECONDS * SAMPLING_RATE;
        const BATCH_CHECK_INTERVAL_MS = 5000;  // Check for auto-batch every 5 seconds
        const BATCH_GRID_SPACING = 30;  // Grid spacing in batch canvas
        let savedBatches = [];  // Array of {startSample, endSample, dataURL, timestamp}
        let lastBatchEndSample = 0;  // Track where last batch ended
        let autoBatchEnabled = true;  // Toggle for auto-batch feature
        
        // Speed control
        function setSpeed(speed) {
            speedMultiplier = speed;
            document.getElementById('speedValue').textContent = speed + 'x';
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent === speed + 'x') btn.classList.add('active');
            });
        }
        
        // History navigation functions
        function scrollHistory(seconds) {
            if (currentIndex < DISPLAY_SAMPLES) return;
            
            viewOffset += seconds;
            const maxHistory = -currentIndex / SAMPLING_RATE;
            viewOffset = Math.max(maxHistory, Math.min(0, viewOffset));
            
            isLive = viewOffset >= -0.1;
            updateHistoryUI();
            drawECG();
            updateTime();
        }
        
        function goToLive() {
            viewOffset = 0;
            isLive = true;
            updateHistoryUI();
            drawECG();
            updateTime();
        }
        
        function navigateToTime(sampleIndex) {
            const targetOffset = (sampleIndex - currentIndex + DISPLAY_SAMPLES/2) / SAMPLING_RATE;
            if (targetOffset >= 0) {
                goToLive();
                return;
            }
            viewOffset = targetOffset;
            isLive = false;
            updateHistoryUI();
            drawECG();
            updateTime();
        }
        
        function updateHistoryUI() {
            const indicator = document.getElementById('historyIndicator');
            const fwdBtn = document.getElementById('fwdBtn');
            const fwd5Btn = document.getElementById('fwd5Btn');
            
            if (isLive) {
                indicator.style.display = 'none';
                fwdBtn.disabled = true;
                fwd5Btn.disabled = true;
            } else {
                indicator.style.display = 'inline';
                fwdBtn.disabled = false;
                fwd5Btn.disabled = false;
            }
        }
        
        // Update graph height dynamically - can expand but never shrinks
        function updateGraphHeight(requestedHeight) {
            const newHeight = Math.max(MIN_GRAPH_HEIGHT, Math.min(MAX_GRAPH_HEIGHT, requestedHeight));
            if (newHeight > maxGraphHeight) {
                maxGraphHeight = newHeight;
                canvas.style.height = maxGraphHeight + 'px';
                resizeCanvas();
            }
        }
        
        // Resize canvas to be pixel-perfect
        function resizeCanvas() {
            // ALWAYS ensure CSS height is at least maxGraphHeight (never shrink)
            // Set it every time to guarantee stability
            canvas.style.height = maxGraphHeight + 'px';
            
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            // Use maxGraphHeight directly instead of getBoundingClientRect for height
            // This ensures the canvas never shrinks even during CSS transitions
            const heightToUse = Math.max(rect.height, maxGraphHeight);
            canvas.height = heightToUse * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            // Also resize beat canvas
            const beatRect = beatCanvas.getBoundingClientRect();
            beatCanvas.width = beatRect.width * window.devicePixelRatio;
            beatCanvas.height = beatRect.height * window.devicePixelRatio;
            beatCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            // Redraw beat if available
            if (currentBeatWaveform) {
                drawBeatWaveform(currentBeatWaveform);
            }
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // ============================================================
        // DRAG INTERACTION FOR SCROLLABLE HISTORY
        // ============================================================
        let isDragging = false;
        let lastDragX = 0;
        
        canvas.style.cursor = 'grab';
        
        function startDrag(x) {
            isDragging = true;
            lastDragX = x;
            canvas.style.cursor = 'grabbing';
        }
        
        function drag(x) {
            if (!isDragging) return;
            
            const deltaX = x - lastDragX;
            lastDragX = x;
            
            // Convert pixel delta to time delta (negative = go back in time)
            const canvasWidth = canvas.getBoundingClientRect().width;
            const secondsPerPixel = DISPLAY_SECONDS / canvasWidth;
            const deltaSeconds = -deltaX * secondsPerPixel;
            
            if (Math.abs(deltaSeconds) > 0.01) {
                scrollHistory(deltaSeconds);
            }
        }
        
        function endDrag() {
            isDragging = false;
            canvas.style.cursor = 'grab';
        }
        
        // Mouse events
        canvas.addEventListener('mousedown', (e) => startDrag(e.clientX));
        canvas.addEventListener('mousemove', (e) => drag(e.clientX));
        canvas.addEventListener('mouseup', () => endDrag());
        canvas.addEventListener('mouseleave', () => endDrag());
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startDrag(e.touches[0].clientX);
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            drag(e.touches[0].clientX);
        });
        canvas.addEventListener('touchend', () => endDrag());
        
        // ============================================================
        // EXPORT TO MEDICAL IMAGE
        // ============================================================
        function exportECG(format = 'png') {
            // Export from 0 second to current realtime position (not just visible window)
            // Multi-row and multi-part support for long recordings
            const startSample = 0;
            const endSample = currentIndex > 0 ? currentIndex : Math.min(DISPLAY_SAMPLES, ecgData.length);
            
            // Export dimension constants
            const EXPORT_MAX_WIDTH = 10000;        // Maximum width per image
            const EXPORT_MAX_HEIGHT = 10000;       // Maximum height before creating new part
            const ROW_HEIGHT = 250;                // Height per ECG row (including labels)
            const HEADER_HEIGHT = 90;              // Space for header
            const FOOTER_HEIGHT = 80;              // Space for legend
            const SECONDS_PER_ROW = 30;            // Seconds of data per row at max width
            const PIXELS_PER_SECOND = EXPORT_MAX_WIDTH / SECONDS_PER_ROW;  // ~333 pixels per second
            
            const totalSeconds = endSample / SAMPLING_RATE;
            const totalSamples = endSample - startSample;
            const samplesPerRow = Math.round(SECONDS_PER_ROW * SAMPLING_RATE);
            const numRows = Math.ceil(totalSamples / samplesPerRow);
            
            // Calculate rows per part (image)
            const maxRowsPerPart = Math.floor((EXPORT_MAX_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT) / ROW_HEIGHT);
            const numParts = Math.ceil(numRows / maxRowsPerPart);
            
            // Get COMPLETE data from 0 to current position
            let fullBuffer = [];
            for (let i = startSample; i < endSample && i < ecgData.length; i++) {
                fullBuffer.push(ecgData[i]);
            }
            
            // Find global min/max for consistent scaling across all rows/parts
            const globalMinVal = Math.min(...fullBuffer);
            const globalMaxVal = Math.max(...fullBuffer);
            const globalRange = globalMaxVal - globalMinVal || 1;
            
            const modelName = document.getElementById('modelName').textContent;
            const timestamp = new Date().toISOString();
            
            console.log(`[ECG] Exporting ${totalSeconds.toFixed(2)}s recording: ${numRows} rows across ${numParts} part(s)`);
            
            // Generate each part
            for (let partIdx = 0; partIdx < numParts; partIdx++) {
                const rowsInThisPart = Math.min(maxRowsPerPart, numRows - partIdx * maxRowsPerPart);
                const exportWidth = EXPORT_MAX_WIDTH;
                const exportHeight = HEADER_HEIGHT + rowsInThisPart * ROW_HEIGHT + FOOTER_HEIGHT;
                
                // Create canvas for this part
                const exportCanvas = document.createElement('canvas');
                exportCanvas.width = exportWidth;
                exportCanvas.height = exportHeight;
                const exportCtx = exportCanvas.getContext('2d');
                
                // White background for medical printing
                exportCtx.fillStyle = '#ffffff';
                exportCtx.fillRect(0, 0, exportWidth, exportHeight);
                
                // Header section
                exportCtx.fillStyle = '#333333';
                exportCtx.font = 'bold 18px Arial';
                const partLabel = numParts > 1 ? ` (Part ${partIdx + 1} of ${numParts})` : '';
                exportCtx.fillText('ECG Analysis Report - Complete Recording' + partLabel, 20, 30);
                
                exportCtx.font = '12px Arial';
                exportCtx.fillStyle = '#666666';
                exportCtx.fillText('Model: ' + modelName, 20, 50);
                exportCtx.fillText('Timestamp: ' + timestamp, 20, 68);
                
                // Calculate time range for this part
                const partStartRow = partIdx * maxRowsPerPart;
                const partEndRow = partStartRow + rowsInThisPart;
                const partStartSample = partStartRow * samplesPerRow;
                const partEndSample = Math.min(partEndRow * samplesPerRow, totalSamples);
                const partStartTime = (partStartSample / SAMPLING_RATE).toFixed(2);
                const partEndTime = (partEndSample / SAMPLING_RATE).toFixed(2);
                
                exportCtx.fillText(`Time Range: ${partStartTime}s - ${partEndTime}s | Total: ${totalSeconds.toFixed(2)}s`, 300, 50);
                exportCtx.fillText(`Rows ${partStartRow + 1}-${partEndRow} of ${numRows} | ${SECONDS_PER_ROW}s per row`, 300, 68);
                
                // Draw each row in this part
                for (let rowInPart = 0; rowInPart < rowsInThisPart; rowInPart++) {
                    const globalRowIdx = partStartRow + rowInPart;
                    const rowStartSample = globalRowIdx * samplesPerRow;
                    const rowEndSample = Math.min(rowStartSample + samplesPerRow, totalSamples);
                    
                    if (rowStartSample >= totalSamples) break;
                    
                    // Get buffer slice for this row
                    const rowBuffer = fullBuffer.slice(rowStartSample, rowEndSample);
                    if (rowBuffer.length === 0) continue;
                    
                    // Row dimensions
                    const graphX = 100;  // More space for time labels
                    const graphY = HEADER_HEIGHT + rowInPart * ROW_HEIGHT + 30;
                    const graphWidth = exportWidth - 120;
                    const graphHeight = ROW_HEIGHT - 50;
                    
                    // Time labels for this row (clear for doctors)
                    const rowStartTime = (rowStartSample / SAMPLING_RATE);
                    const rowEndTime = (rowEndSample / SAMPLING_RATE);
                    
                    // Calculate actual width for this row's data (partial rows don't fill full width)
                    const rowDataWidth = (rowBuffer.length / samplesPerRow) * graphWidth;
                    const isPartialRow = rowBuffer.length < samplesPerRow;
                    
                    exportCtx.fillStyle = '#1a5276';
                    exportCtx.font = 'bold 14px Arial';
                    exportCtx.fillText(formatTime(rowStartTime), 10, graphY + graphHeight / 2 + 5);
                    
                    // Position end time label at actual data end (not fixed right edge) for partial rows
                    if (isPartialRow) {
                        const endLabelX = graphX + rowDataWidth + 10;
                        exportCtx.fillText(formatTime(rowEndTime), endLabelX, graphY + graphHeight / 2 + 5);
                        
                        // Draw a vertical line to indicate where data ends
                        exportCtx.strokeStyle = '#aaaaaa';
                        exportCtx.lineWidth = 2;
                        exportCtx.setLineDash([5, 5]);
                        exportCtx.beginPath();
                        exportCtx.moveTo(graphX + rowDataWidth, graphY);
                        exportCtx.lineTo(graphX + rowDataWidth, graphY + graphHeight);
                        exportCtx.stroke();
                        exportCtx.setLineDash([]);
                        
                        // Add "END" label
                        exportCtx.fillStyle = '#888888';
                        exportCtx.font = 'italic 10px Arial';
                        exportCtx.fillText('(Recording End)', graphX + rowDataWidth + 10, graphY + graphHeight / 2 + 20);
                    } else {
                        exportCtx.fillText(formatTime(rowEndTime), exportWidth - 85, graphY + graphHeight / 2 + 5);
                    }
                    
                    // Row number label
                    exportCtx.fillStyle = '#7f8c8d';
                    exportCtx.font = '10px Arial';
                    exportCtx.fillText(`Row ${globalRowIdx + 1}`, 10, graphY - 5);
                    
                    // Graph border
                    exportCtx.strokeStyle = '#cccccc';
                    exportCtx.lineWidth = 1;
                    exportCtx.strokeRect(graphX, graphY, graphWidth, graphHeight);
                    
                    // Medical ECG grid (red)
                    const gridSpacingSmall = 15;
                    const gridSpacingLarge = 75;
                    
                    exportCtx.strokeStyle = '#ffcccc';
                    exportCtx.lineWidth = 0.5;
                    for (let x = graphX; x <= graphX + graphWidth; x += gridSpacingSmall) {
                        exportCtx.beginPath();
                        exportCtx.moveTo(x, graphY);
                        exportCtx.lineTo(x, graphY + graphHeight);
                        exportCtx.stroke();
                    }
                    for (let y = graphY; y <= graphY + graphHeight; y += gridSpacingSmall) {
                        exportCtx.beginPath();
                        exportCtx.moveTo(graphX, y);
                        exportCtx.lineTo(graphX + graphWidth, y);
                        exportCtx.stroke();
                    }
                    
                    // Large grid
                    exportCtx.strokeStyle = '#ff9999';
                    exportCtx.lineWidth = 1;
                    for (let x = graphX; x <= graphX + graphWidth; x += gridSpacingLarge) {
                        exportCtx.beginPath();
                        exportCtx.moveTo(x, graphY);
                        exportCtx.lineTo(x, graphY + graphHeight);
                        exportCtx.stroke();
                    }
                    
                    // Time markers along top of each row
                    exportCtx.fillStyle = '#666666';
                    exportCtx.font = '9px Arial';
                    const secondsInRow = (rowEndSample - rowStartSample) / SAMPLING_RATE;
                    
                    // Calculate actual width used by this row's data (maintain consistent scale)
                    // Full rows use full graphWidth, partial rows use proportional width
                    const actualRowWidth = (rowBuffer.length / samplesPerRow) * graphWidth;
                    
                    const timeMarkInterval = SECONDS_PER_ROW > 20 ? 5 : (SECONDS_PER_ROW > 10 ? 2 : 1);
                    // Only draw time markers up to the actual data extent
                    for (let t = 0; t <= secondsInRow; t += timeMarkInterval) {
                        const xPos = graphX + (t / SECONDS_PER_ROW) * graphWidth;
                        if (xPos <= graphX + actualRowWidth + 5) {  // Only within data range
                            const timeLabel = (rowStartTime + t).toFixed(1) + 's';
                            exportCtx.fillText(timeLabel, xPos - 10, graphY - 3);
                            
                            // Small tick mark
                            exportCtx.strokeStyle = '#999999';
                            exportCtx.lineWidth = 1;
                            exportCtx.beginPath();
                            exportCtx.moveTo(xPos, graphY);
                            exportCtx.lineTo(xPos, graphY + 5);
                            exportCtx.stroke();
                        }
                    }
                    
                    // Draw ECG signal for this row - MAINTAIN CONSISTENT SCALE (no stretching)
                    if (rowBuffer.length >= 2) {
                        exportCtx.strokeStyle = '#00aa66';
                        exportCtx.lineWidth = 1.5;
                        exportCtx.beginPath();
                        
                        for (let i = 0; i < rowBuffer.length; i++) {
                            // Use consistent pixels-per-sample ratio (based on full row samples)
                            // This prevents stretching of partial rows
                            const x = graphX + (i / samplesPerRow) * graphWidth;
                            const y = graphY + graphHeight - ((rowBuffer[i] - globalMinVal) / globalRange) * (graphHeight - 20) - 10;
                            
                            if (i === 0) {
                                exportCtx.moveTo(x, y);
                            } else {
                                exportCtx.lineTo(x, y);
                            }
                        }
                        exportCtx.stroke();
                        
                        // Draw R-peak markers for this row - MAINTAIN CONSISTENT SCALE
                        annotations.forEach(ann => {
                            const globalIdx = ann.sample_index - startSample;
                            if (globalIdx >= rowStartSample && globalIdx < rowEndSample) {
                                const localIdx = globalIdx - rowStartSample;
                                if (localIdx >= 0 && localIdx < rowBuffer.length) {
                                    // Use consistent pixels-per-sample ratio
                                    const x = graphX + (localIdx / samplesPerRow) * graphWidth;
                                    const y = graphY + graphHeight - ((rowBuffer[localIdx] - globalMinVal) / globalRange) * (graphHeight - 20) - 10;
                                    
                                    // Check for false detection
                                    const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                                    if (classResult && classResult.correct === false) {
                                        exportCtx.strokeStyle = '#cc8800';
                                        exportCtx.lineWidth = 2;
                                        exportCtx.beginPath();
                                        exportCtx.arc(x, y, 6, 0, Math.PI * 2);
                                        exportCtx.stroke();
                                    }
                                    
                                    // R-peak marker
                                    exportCtx.fillStyle = ann.beat_type === 'N' ? '#00aa66' : '#cc3333';
                                    exportCtx.beginPath();
                                    exportCtx.arc(x, y, 3, 0, Math.PI * 2);
                                    exportCtx.fill();
                                }
                            }
                        });
                    }
                }
                
                // Legend at bottom
                const legendY = exportHeight - 50;
                exportCtx.font = '11px Arial';
                exportCtx.fillStyle = '#00aa66';
                exportCtx.beginPath();
                exportCtx.arc(60, legendY, 5, 0, Math.PI * 2);
                exportCtx.fill();
                exportCtx.fillStyle = '#333333';
                exportCtx.fillText('Normal Beat', 72, legendY + 4);
                
                exportCtx.fillStyle = '#cc3333';
                exportCtx.beginPath();
                exportCtx.arc(180, legendY, 5, 0, Math.PI * 2);
                exportCtx.fill();
                exportCtx.fillStyle = '#333333';
                exportCtx.fillText('Abnormal Beat', 192, legendY + 4);
                
                exportCtx.strokeStyle = '#cc8800';
                exportCtx.lineWidth = 2;
                exportCtx.beginPath();
                exportCtx.arc(320, legendY, 7, 0, Math.PI * 2);
                exportCtx.stroke();
                exportCtx.fillStyle = '#333333';
                exportCtx.fillText('False Detection', 335, legendY + 4);
                
                // Scale info
                exportCtx.fillStyle = '#666666';
                exportCtx.font = '10px Arial';
                exportCtx.fillText(`Scale: ${SECONDS_PER_ROW}s per row | Sampling: ${SAMPLING_RATE}Hz`, 450, legendY + 4);
                
                // Create download link for this part
                const partSuffix = numParts > 1 ? `_part${partIdx + 1}` : '';
                const dataURL = exportCanvas.toDataURL('image/' + format, 0.95);
                const link = document.createElement('a');
                link.download = 'ecg_complete_' + timestamp.replace(/[:.]/g, '-') + partSuffix + '.' + format;
                link.href = dataURL;
                link.click();
                
                console.log(`[ECG] Exported part ${partIdx + 1}/${numParts} as ${format.toUpperCase()}`);
            }
            
            console.log('[ECG] Export complete: ' + numParts + ' file(s) generated');
        }
        
        // Helper function to format time as MM:SS.s for clear doctor readability
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(1);
            if (mins > 0) {
                return `${mins}:${secs.padStart(4, '0')}`;
            }
            return `${secs}s`;
        }
        
        // ============================================================
        // AUTO-BATCH FUNCTIONS
        // ============================================================
        
        // Check if it's time to auto-save a batch
        function checkAutoBatch() {
            if (!autoBatchEnabled) return;
            
            // Check if we've accumulated enough new data for a batch
            const unsavedSamples = currentIndex - lastBatchEndSample;
            if (unsavedSamples >= AUTO_BATCH_INTERVAL_SAMPLES) {
                saveBatch(lastBatchEndSample, currentIndex);
            }
        }
        
        // Save a batch of ECG data as an image (stored in memory, not downloaded)
        function saveBatch(startSample, endSample) {
            if (endSample <= startSample) return;
            
            const batchNum = savedBatches.length + 1;
            console.log(`[ECG] Auto-saving batch ${batchNum}: samples ${startSample} to ${endSample}`);
            
            // Generate batch image
            const batchCanvas = generateBatchCanvas(startSample, endSample, batchNum);
            const dataURL = batchCanvas.toDataURL('image/png', 0.95);
            
            // Store batch metadata
            savedBatches.push({
                batchNum: batchNum,
                startSample: startSample,
                endSample: endSample,
                startTime: startSample / SAMPLING_RATE,
                endTime: endSample / SAMPLING_RATE,
                dataURL: dataURL,
                timestamp: new Date().toISOString()
            });
            
            lastBatchEndSample = endSample;
            updateBatchStatus();
            
            console.log(`[ECG] Batch ${batchNum} saved (${((endSample - startSample) / SAMPLING_RATE).toFixed(1)}s)`);
        }
        
        // Generate canvas for a batch of data
        function generateBatchCanvas(startSample, endSample, batchNum) {
            const EXPORT_MAX_WIDTH = 10000;
            const ROW_HEIGHT = 250;
            const HEADER_HEIGHT = 90;
            const FOOTER_HEIGHT = 80;
            const SECONDS_PER_ROW = 30;
            
            const totalSamples = endSample - startSample;
            const samplesPerRow = Math.round(SECONDS_PER_ROW * SAMPLING_RATE);
            const numRows = Math.ceil(totalSamples / samplesPerRow);
            
            const exportWidth = EXPORT_MAX_WIDTH;
            const exportHeight = HEADER_HEIGHT + numRows * ROW_HEIGHT + FOOTER_HEIGHT;
            
            const exportCanvas = document.createElement('canvas');
            exportCanvas.width = exportWidth;
            exportCanvas.height = exportHeight;
            const exportCtx = exportCanvas.getContext('2d');
            
            // White background
            exportCtx.fillStyle = '#ffffff';
            exportCtx.fillRect(0, 0, exportWidth, exportHeight);
            
            // Get data buffer
            let buffer = [];
            for (let i = startSample; i < endSample && i < ecgData.length; i++) {
                buffer.push(ecgData[i]);
            }
            
            // Find global min/max
            const globalMinVal = Math.min(...buffer);
            const globalMaxVal = Math.max(...buffer);
            const globalRange = globalMaxVal - globalMinVal || 1;
            
            const modelName = document.getElementById('modelName').textContent;
            const timestamp = new Date().toISOString();
            
            // Header
            exportCtx.fillStyle = '#333333';
            exportCtx.font = 'bold 18px Arial';
            exportCtx.fillText(`ECG Recording - Batch ${batchNum}`, 20, 30);
            
            exportCtx.font = '12px Arial';
            exportCtx.fillStyle = '#666666';
            exportCtx.fillText('Model: ' + modelName, 20, 50);
            exportCtx.fillText('Saved: ' + timestamp, 20, 68);
            
            const startTime = (startSample / SAMPLING_RATE).toFixed(2);
            const endTime = (endSample / SAMPLING_RATE).toFixed(2);
            exportCtx.fillText(`Time Range: ${startTime}s - ${endTime}s | ${numRows} row(s)`, 300, 50);
            
            // Draw each row
            for (let rowIdx = 0; rowIdx < numRows; rowIdx++) {
                const rowStartSample = rowIdx * samplesPerRow;
                const rowEndSample = Math.min(rowStartSample + samplesPerRow, totalSamples);
                
                const rowBuffer = buffer.slice(rowStartSample, rowEndSample);
                if (rowBuffer.length === 0) continue;
                
                const graphX = 100;
                const graphY = HEADER_HEIGHT + rowIdx * ROW_HEIGHT + 30;
                const graphWidth = exportWidth - 120;
                const graphHeight = ROW_HEIGHT - 50;
                
                const rowStartTime = ((startSample + rowStartSample) / SAMPLING_RATE);
                const rowEndTime = ((startSample + rowEndSample) / SAMPLING_RATE);
                
                // Time labels
                exportCtx.fillStyle = '#1a5276';
                exportCtx.font = 'bold 14px Arial';
                exportCtx.fillText(formatTime(rowStartTime), 10, graphY + graphHeight / 2 + 5);
                exportCtx.fillText(formatTime(rowEndTime), exportWidth - 85, graphY + graphHeight / 2 + 5);
                
                // Row number
                exportCtx.fillStyle = '#7f8c8d';
                exportCtx.font = '10px Arial';
                exportCtx.fillText(`Row ${rowIdx + 1}`, 10, graphY - 5);
                
                // Graph border
                exportCtx.strokeStyle = '#cccccc';
                exportCtx.lineWidth = 1;
                exportCtx.strokeRect(graphX, graphY, graphWidth, graphHeight);
                
                // Medical grid
                exportCtx.strokeStyle = '#ffcccc';
                exportCtx.lineWidth = 0.5;
                for (let x = graphX; x <= graphX + graphWidth; x += 15) {
                    exportCtx.beginPath();
                    exportCtx.moveTo(x, graphY);
                    exportCtx.lineTo(x, graphY + graphHeight);
                    exportCtx.stroke();
                }
                for (let y = graphY; y <= graphY + graphHeight; y += 15) {
                    exportCtx.beginPath();
                    exportCtx.moveTo(graphX, y);
                    exportCtx.lineTo(graphX + graphWidth, y);
                    exportCtx.stroke();
                }
                
                // Draw ECG signal
                if (rowBuffer.length >= 2) {
                    exportCtx.strokeStyle = '#00aa66';
                    exportCtx.lineWidth = 1.5;
                    exportCtx.beginPath();
                    
                    for (let i = 0; i < rowBuffer.length; i++) {
                        const x = graphX + (i / samplesPerRow) * graphWidth;
                        const y = graphY + graphHeight - ((rowBuffer[i] - globalMinVal) / globalRange) * (graphHeight - 20) - 10;
                        
                        if (i === 0) {
                            exportCtx.moveTo(x, y);
                        } else {
                            exportCtx.lineTo(x, y);
                        }
                    }
                    exportCtx.stroke();
                    
                    // Draw R-peak markers
                    annotations.forEach(ann => {
                        const globalIdx = ann.sample_index - startSample;
                        if (globalIdx >= rowStartSample && globalIdx < rowEndSample) {
                            const localIdx = globalIdx - rowStartSample;
                            if (localIdx >= 0 && localIdx < rowBuffer.length) {
                                const x = graphX + (localIdx / samplesPerRow) * graphWidth;
                                const y = graphY + graphHeight - ((rowBuffer[localIdx] - globalMinVal) / globalRange) * (graphHeight - 20) - 10;
                                
                                const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                                if (classResult && classResult.correct === false) {
                                    exportCtx.strokeStyle = '#cc8800';
                                    exportCtx.lineWidth = 2;
                                    exportCtx.beginPath();
                                    exportCtx.arc(x, y, 6, 0, Math.PI * 2);
                                    exportCtx.stroke();
                                }
                                
                                exportCtx.fillStyle = ann.beat_type === 'N' ? '#00aa66' : '#cc3333';
                                exportCtx.beginPath();
                                exportCtx.arc(x, y, 3, 0, Math.PI * 2);
                                exportCtx.fill();
                            }
                        }
                    });
                }
            }
            
            // Legend
            const legendY = exportHeight - 50;
            exportCtx.font = '11px Arial';
            exportCtx.fillStyle = '#00aa66';
            exportCtx.beginPath();
            exportCtx.arc(60, legendY, 5, 0, Math.PI * 2);
            exportCtx.fill();
            exportCtx.fillStyle = '#333333';
            exportCtx.fillText('Normal', 72, legendY + 4);
            
            exportCtx.fillStyle = '#cc3333';
            exportCtx.beginPath();
            exportCtx.arc(140, legendY, 5, 0, Math.PI * 2);
            exportCtx.fill();
            exportCtx.fillStyle = '#333333';
            exportCtx.fillText('Abnormal', 152, legendY + 4);
            
            exportCtx.strokeStyle = '#cc8800';
            exportCtx.lineWidth = 2;
            exportCtx.beginPath();
            exportCtx.arc(240, legendY, 7, 0, Math.PI * 2);
            exportCtx.stroke();
            exportCtx.fillStyle = '#333333';
            exportCtx.fillText('False', 255, legendY + 4);
            
            exportCtx.fillStyle = '#666666';
            exportCtx.font = '10px Arial';
            exportCtx.fillText(`Sampling: ${SAMPLING_RATE}Hz | ${SECONDS_PER_ROW}s/row`, 320, legendY + 4);
            
            return exportCanvas;
        }
        
        // Update batch status display
        function updateBatchStatus() {
            const statusEl = document.getElementById('batchStatus');
            if (!statusEl) return;
            
            const totalSaved = savedBatches.reduce((sum, b) => sum + (b.endSample - b.startSample), 0);
            const savedSeconds = totalSaved / SAMPLING_RATE;
            const unsavedSeconds = (currentIndex - lastBatchEndSample) / SAMPLING_RATE;
            
            statusEl.innerHTML = `
                <span style="color: #00ff88;">📦 ${savedBatches.length} batch${savedBatches.length !== 1 ? 'es' : ''}</span>
                <span style="color: #888; margin-left: 10px;">(${savedSeconds.toFixed(0)}s saved)</span>
                ${unsavedSeconds > 10 ? `<span style="color: #ffd700; margin-left: 10px;">⏳ ${unsavedSeconds.toFixed(0)}s pending</span>` : ''}
            `;
        }
        
        // Download all saved batches as a single ZIP file
        async function downloadAllBatches() {
            if (savedBatches.length === 0) {
                alert('No batches saved yet. Recording auto-saves batches every 2 minutes.');
                return;
            }
            
            const totalBatches = savedBatches.length;
            const totalSeconds = savedBatches.reduce((sum, b) => sum + (b.endSample - b.startSample), 0) / SAMPLING_RATE;
            
            // Show creating ZIP status
            const statusEl = document.getElementById('batchStatus');
            if (statusEl) {
                statusEl.innerHTML = `
                    <span style="color: #ffaa00;">📦 Creating ZIP (${totalBatches} batch${totalBatches !== 1 ? 'es' : ''})...</span>
                    <span style="color: #888; margin-left: 10px;">(${totalSeconds.toFixed(0)}s total)</span>
                `;
            }
            
            console.log(`[ECG] Creating ZIP with ${totalBatches} batches...`);
            
            try {
                // Create ZIP file
                const zip = new JSZip();
                
                // Add each batch to the ZIP
                for (const batch of savedBatches) {
                    // Convert data URL to blob
                    const dataURL = batch.dataURL;
                    const base64Data = dataURL.split(',')[1];
                    const filename = `ecg_batch_${batch.batchNum}_${batch.timestamp.replace(/[:.]/g, '-')}.png`;
                    zip.file(filename, base64Data, {base64: true});
                }
                
                // Generate ZIP blob
                const zipBlob = await zip.generateAsync({type: 'blob'});
                
                // Create download link
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                const link = document.createElement('a');
                link.download = `ecg_recording_${timestamp}.zip`;
                link.href = URL.createObjectURL(zipBlob);
                link.click();
                
                // Cleanup
                setTimeout(() => URL.revokeObjectURL(link.href), 1000);
                
                console.log('[ECG] ZIP downloaded successfully');
                
                if (statusEl) {
                    statusEl.innerHTML = `
                        <span style="color: #00ff88;">✅ ZIP downloaded (${totalBatches} batch${totalBatches !== 1 ? 'es' : ''})!</span>
                        <span style="color: #888; margin-left: 10px;">(${totalSeconds.toFixed(0)}s total)</span>
                    `;
                }
            } catch (error) {
                console.error('[ECG] Error creating ZIP:', error);
                alert('Error creating ZIP file. Please try again.');
                if (statusEl) {
                    statusEl.innerHTML = `
                        <span style="color: #ff4444;">❌ Error creating ZIP</span>
                    `;
                }
            }
        }
        
        // Export only unsaved data (faster than full export)
        function exportUnsaved(format = 'png') {
            const unsavedStart = lastBatchEndSample;
            const unsavedEnd = currentIndex;
            
            if (unsavedEnd <= unsavedStart) {
                alert('No unsaved data to export. All data has been saved in batches.');
                return;
            }
            
            console.log(`[ECG] Exporting unsaved data: ${unsavedStart} to ${unsavedEnd}`);
            
            const batchCanvas = generateBatchCanvas(unsavedStart, unsavedEnd, savedBatches.length + 1);
            const dataURL = batchCanvas.toDataURL('image/' + format, 0.95);
            const link = document.createElement('a');
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            link.download = `ecg_unsaved_${timestamp}.${format}`;
            link.href = dataURL;
            link.click();
            
            console.log('[ECG] Unsaved data exported');
        }
        
        // Force save current pending data as a batch
        function forceSaveBatch() {
            const unsavedSamples = currentIndex - lastBatchEndSample;
            if (unsavedSamples < MIN_BATCH_SAMPLES) {
                alert(`Need at least ${MIN_BATCH_SECONDS} seconds of unsaved data to create a batch.`);
                return;
            }
            saveBatch(lastBatchEndSample, currentIndex);
        }
        
        // Draw beat waveform on the beat snapshot canvas
        function drawBeatWaveform(waveform, isAbnormal = false) {
            const width = beatCanvas.getBoundingClientRect().width;
            const height = beatCanvas.getBoundingClientRect().height;
            
            // Clear canvas
            beatCtx.fillStyle = '#0a0a1a';
            beatCtx.fillRect(0, 0, width, height);
            
            // Draw grid
            beatCtx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
            beatCtx.lineWidth = 1;
            for (let x = 0; x < width; x += 30) {
                beatCtx.beginPath();
                beatCtx.moveTo(x, 0);
                beatCtx.lineTo(x, height);
                beatCtx.stroke();
            }
            for (let y = 0; y < height; y += 30) {
                beatCtx.beginPath();
                beatCtx.moveTo(0, y);
                beatCtx.lineTo(width, y);
                beatCtx.stroke();
            }
            
            if (!waveform || waveform.length < 2) return;
            
            // Find min/max for scaling
            const minVal = Math.min(...waveform);
            const maxVal = Math.max(...waveform);
            const range = maxVal - minVal || 1;
            
            // Draw beat waveform
            beatCtx.strokeStyle = isAbnormal ? '#ff4757' : '#00ff88';
            beatCtx.lineWidth = 2;
            beatCtx.beginPath();
            
            for (let i = 0; i < waveform.length; i++) {
                const x = (i / waveform.length) * width;
                const y = height - ((waveform[i] - minVal) / range) * (height - 20) - 10;
                
                if (i === 0) {
                    beatCtx.moveTo(x, y);
                } else {
                    beatCtx.lineTo(x, y);
                }
            }
            beatCtx.stroke();
            
            // Draw R-peak marker at the correct position (varies by model: v2/v3/v5=70, v6=90)
            const rPeakX = (currentRPeakPos / waveform.length) * width;
            const rPeakY = height - ((waveform[Math.min(currentRPeakPos, waveform.length-1)] - minVal) / range) * (height - 20) - 10;
            beatCtx.fillStyle = '#ffcc00';
            beatCtx.beginPath();
            beatCtx.arc(rPeakX, rPeakY, 6, 0, Math.PI * 2);
            beatCtx.fill();
            beatCtx.fillStyle = '#ffcc00';
            beatCtx.font = '11px Arial';
            beatCtx.fillText('R-peak', rPeakX - 18, rPeakY - 10);
        }
        
        // Load data from server
        async function loadData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            ecgData = data.signal;
            annotations = data.annotations;
            console.log(`Loaded ${ecgData.length} ECG samples and ${annotations.length} annotations`);
            
            // Reset Y-axis tracking for new data
            globalMinVal = Infinity;
            globalMaxVal = -Infinity;
        }
        
        // Draw ECG signal
        function drawECG() {
            const width = canvas.getBoundingClientRect().width;
            const height = canvas.getBoundingClientRect().height;
            
            // Clear canvas
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, width, height);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x < width; x += 50) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            for (let y = 0; y < height; y += 50) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // Calculate display range based on view offset
            let endSample = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
            let startSample = Math.max(0, endSample - DISPLAY_SAMPLES);
            
            // Get display buffer from ecgData
            let buffer = [];
            for (let i = startSample; i < endSample && i < ecgData.length; i++) {
                buffer.push(ecgData[i]);
            }
            
            if (buffer.length < 2) return;
            
            // Find min/max for the current buffer
            const localMinVal = Math.min(...buffer);
            const localMaxVal = Math.max(...buffer);
            
            // Update global min/max - expand to fit largest signal seen, but never shrink
            // This ensures Y-axis scale remains consistent across entire recording
            if (localMinVal < globalMinVal) globalMinVal = localMinVal;
            if (localMaxVal > globalMaxVal) globalMaxVal = localMaxVal;
            
            // Use global values for scaling (stable Y-axis that expands but never shrinks)
            const minVal = globalMinVal;
            const maxVal = globalMaxVal;
            const range = maxVal - minVal || 1;
            
            // Dynamic height expansion based on signal amplitude and content
            // Count visible annotations to determine if we need more height
            let visibleAnnotations = 0;
            annotations.forEach(ann => {
                if (ann.sample_index > startSample && ann.sample_index <= endSample) {
                    visibleAnnotations++;
                }
            });
            
            // Expand height if many annotations or high signal variance
            // More annotations = more markers = need more height for clarity
            const baseHeight = MIN_GRAPH_HEIGHT;
            const heightPerAnnotation = 5;  // Add 5px per visible annotation (up to limit)
            const annotationBonus = Math.min(visibleAnnotations * heightPerAnnotation, 200);
            const desiredHeight = baseHeight + annotationBonus;
            
            // Update graph height (will only expand, never shrink)
            updateGraphHeight(desiredHeight);
            
            // Draw ECG line
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < buffer.length; i++) {
                const x = (i / DISPLAY_SAMPLES) * width;
                const y = height - ((buffer[i] - minVal) / range) * (height - 40) - 20;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Draw R-peak markers
            annotations.forEach(ann => {
                if (ann.sample_index > startSample && ann.sample_index <= endSample) {
                    const bufferIdx = ann.sample_index - startSample;
                    if (bufferIdx >= 0 && bufferIdx < buffer.length) {
                        const x = (bufferIdx / DISPLAY_SAMPLES) * width;
                        const y = height - ((buffer[bufferIdx] - minVal) / range) * (height - 40) - 20;
                        
                        // Check if this beat has a false detection
                        const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                        if (classResult && classResult.correct === false) {
                            // Yellow circle for false detection
                            ctx.strokeStyle = '#ffd700';
                            ctx.lineWidth = 3;
                            ctx.beginPath();
                            ctx.arc(x, y, 10, 0, Math.PI * 2);
                            ctx.stroke();
                        }
                        
                        // Draw marker
                        ctx.fillStyle = ann.beat_type === 'N' ? '#00ff88' : '#ff4757';
                        ctx.beginPath();
                        ctx.arc(x, y, 6, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });
            
            // Show "History Mode" indicator if not live
            if (!isLive) {
                ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
                ctx.font = 'bold 14px Arial';
                ctx.fillText('📜 VIEWING HISTORY', 10, 25);
            }
        }
        
        // Update time display
        function updateTime() {
            let displayIndex = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
            const seconds = displayIndex / SAMPLING_RATE;
            const minutes = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(3);
            document.getElementById('currentTime').textContent = 
                `${minutes}:${secs.padStart(6, '0')}`;
        }
        
        // Calculate BPM from recent beat intervals
        function calculateBPM(currentBeatSample) {
            beatTimes.push(currentBeatSample);
            
            // Keep only last 10 beats for smoothing
            if (beatTimes.length > 10) {
                beatTimes.shift();
            }
            
            if (beatTimes.length < 2) return null;
            
            // Calculate average interval from recent beats
            let totalInterval = 0;
            let count = 0;
            for (let i = 1; i < beatTimes.length; i++) {
                const interval = (beatTimes[i] - beatTimes[i-1]) / SAMPLING_RATE;
                // Only count reasonable intervals (30-200 BPM range)
                if (interval > 0.3 && interval < 2.0) {
                    totalInterval += interval;
                    count++;
                }
            }
            
            if (count === 0) return null;
            
            const avgInterval = totalInterval / count;
            return Math.round(60 / avgInterval);
        }
        
        // Check for beats and classify - with throttling for high-speed mode
        async function checkForBeats() {
            // Skip if already processing classifications
            if (isClassifying) return;
            
            // Calculate samples to check based on speed
            const samplesToCheck = Math.max(1, Math.round(speedMultiplier * (SAMPLING_RATE / 60)));
            const prevSample = currentIndex - samplesToCheck;
            
            // Collect beats to classify (avoid duplicates)
            const beatsToClassify = [];
            for (const ann of annotations) {
                if (ann.sample_index > prevSample && ann.sample_index <= currentIndex && 
                    ann.beat_type !== '+' && !processedBeats.has(ann.sample_index)) {
                    beatsToClassify.push(ann);
                }
            }
            
            if (beatsToClassify.length === 0) return;
            
            isClassifying = true;
            
            try {
                for (const ann of beatsToClassify) {
                    // Mark as processed immediately to prevent duplicates
                    processedBeats.add(ann.sample_index);
                    
                    // Limit processed beats set size to prevent memory issues
                    if (processedBeats.size > 5000) {
                        const toRemove = [...processedBeats].slice(0, 1000);
                        toRemove.forEach(v => processedBeats.delete(v));
                    }
                    
                    try {
                        const response = await fetch('/api/classify', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                r_peak: ann.sample_index,
                                beat_type: ann.beat_type
                            })
                        });
                        const result = await response.json();
                        console.log('[ECG] Beat at', ann.sample_index, ':', result.predicted);
                        addClassification(result);
                        
                        // Calculate heart rate
                        const bpm = calculateBPM(ann.sample_index);
                        if (bpm !== null && bpm > 0 && bpm < 300) {
                            document.getElementById('heartRate').textContent = bpm;
                        }
                    } catch (e) {
                        console.error('Classification error:', e);
                    }
                }
            } finally {
                isClassifying = false;
            }
        }
        
        // Add classification result
        function addClassification(result) {
            classifications.unshift(result);
            
            // Limit array size to prevent memory issues at high speed
            if (classifications.length > MAX_CLASSIFICATIONS) {
                classifications = classifications.slice(0, MAX_CLASSIFICATIONS);
            }
            
            // Track false detections
            if (result.correct === false) {
                falseDetections.unshift(result);
                // Limit false detections array
                if (falseDetections.length > MAX_FALSE_DETECTIONS) {
                    falseDetections = falseDetections.slice(0, MAX_FALSE_DETECTIONS);
                }
                updateFalseDetectionList();
            }
            
            // Update stats
            const total = classifications.filter(c => c.correct !== null).length;
            const normal = classifications.filter(c => c.predicted === 'NORMAL').length;
            const abnormal = classifications.filter(c => c.predicted === 'ABNORMAL').length;
            const correct = classifications.filter(c => c.correct === true).length;
            const known = classifications.filter(c => c.correct !== null).length;
            
            document.getElementById('totalBeats').textContent = classifications.length;
            document.getElementById('normalBeats').textContent = normal;
            document.getElementById('abnormalBeats').textContent = abnormal;
            document.getElementById('falseCount').textContent = falseDetections.length;
            if (known > 0) {
                document.getElementById('accuracy').textContent = 
                    Math.round((correct / known) * 100) + '%';
            }
            
            // Update current status
            const statusEl = document.getElementById('currentStatus');
            statusEl.textContent = result.predicted;
            statusEl.className = 'value ' + result.predicted.toLowerCase();
            
            // Update probability bar
            const prob = result.probability;
            const probBar = document.getElementById('probBar');
            probBar.style.width = (prob * 100) + '%';
            probBar.style.background = prob >= 0.5 ? '#ff4757' : '#00ff88';
            document.getElementById('probText').textContent = 
                `Abnormal Probability: ${(prob * 100).toFixed(1)}%`;
            
            // Update beat snapshot display
            if (result.beat_waveform) {
                currentBeatWaveform = result.beat_waveform;
                // Update R-peak position from result (v2/v3/v5=70, v6=90)
                currentRPeakPos = result.r_peak_pos_in_beat || 70;
                currentBeatLength = result.beat_length || 188;
                const isAbnormal = result.predicted === 'ABNORMAL';
                drawBeatWaveform(result.beat_waveform, isAbnormal);
                
                // Update beat info displays
                const beatTypeEl = document.getElementById('beatTypeDisplay');
                beatTypeEl.textContent = result.beat_type;
                beatTypeEl.style.color = result.beat_type === 'N' ? '#00ff88' : '#ff4757';
                
                const groundTruthEl = document.getElementById('groundTruthDisplay');
                groundTruthEl.textContent = result.ground_truth;
                groundTruthEl.style.color = result.ground_truth === 'NORMAL' ? '#00ff88' : '#ff4757';
                
                const predictionEl = document.getElementById('predictionDisplay');
                predictionEl.textContent = result.predicted;
                predictionEl.style.color = result.predicted === 'NORMAL' ? '#00ff88' : '#ff4757';
            }
            
            // Update list
            const listEl = document.getElementById('classificationList');
            if (classifications.length === 1) {
                listEl.innerHTML = '';
            }
            
            const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
            const item = document.createElement('div');
            const incorrectClass = result.correct === false ? ' style="border: 2px solid #ffd700;"' : '';
            item.className = 'classification-item ' + result.predicted.toLowerCase();
            if (result.correct === false) item.style.border = '2px solid #ffd700';
            item.style.cursor = 'pointer';
            item.onclick = () => navigateToTime(result.r_peak);
            item.innerHTML = `
                <div class="beat-info">
                    <div>Beat Type: ${result.beat_type} → ${result.predicted}</div>
                    <div class="beat-time">Time: ${time}s | Prob: ${(result.probability * 100).toFixed(1)}%</div>
                </div>
                <span class="prediction-badge ${result.predicted.toLowerCase()}">${result.predicted}</span>
            `;
            listEl.insertBefore(item, listEl.firstChild);
            
            // Keep only last 100 items
            while (listEl.children.length > 100) {
                listEl.removeChild(listEl.lastChild);
            }
        }
        
        // Update false detection list
        function updateFalseDetectionList() {
            const listEl = document.getElementById('falseDetectionList');
            
            if (falseDetections.length === 0) {
                listEl.innerHTML = '<p style="color: #888; text-align: center;">No false detections yet.</p>';
                return;
            }
            
            listEl.innerHTML = '';
            
            falseDetections.slice(0, 50).forEach(result => {
                const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
                const item = document.createElement('div');
                item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; margin-bottom: 6px; border-radius: 8px; background: rgba(255, 215, 0, 0.15); border-left: 3px solid #ffd700; cursor: pointer;';
                item.onclick = () => navigateToTime(result.r_peak);
                item.innerHTML = `
                    <div>
                        <span style="color: #ffd700; font-weight: bold;">${time}s</span>
                        <span style="color: #aaa; font-size: 11px; margin-left: 8px;">Expected: ${result.ground_truth} | Got: ${result.predicted}</span>
                    </div>
                `;
                item.onmouseover = () => { item.style.background = 'rgba(255, 215, 0, 0.3)'; item.style.transform = 'translateX(5px)'; };
                item.onmouseout = () => { item.style.background = 'rgba(255, 215, 0, 0.15)'; item.style.transform = 'none'; };
                listEl.appendChild(item);
            });
        }
        
        // Animation loop with proper timing
        let lastFrameTime = 0;
        const targetFPS = 60;
        const frameInterval = 1000 / targetFPS;
        let lastBatchCheckTime = 0;
        
        function animate(timestamp) {
            if (!isRunning) return;
            
            // Calculate time delta for proper timing
            const deltaTime = timestamp - lastFrameTime;
            
            if (deltaTime >= frameInterval) {
                lastFrameTime = timestamp - (deltaTime % frameInterval);
                
                // Calculate samples to advance: 1x speed = 360 samples/sec = 6 samples/frame at 60fps
                const samplesPerSecond = SAMPLING_RATE * speedMultiplier;
                const samplesToAdvance = Math.max(1, Math.round(samplesPerSecond / targetFPS));
                
                // Advance samples
                for (let i = 0; i < samplesToAdvance; i++) {
                    if (currentIndex < ecgData.length) {
                        currentIndex++;
                    }
                }
                
                // Update display if in live mode
                if (isLive) {
                    drawECG();
                    updateTime();
                }
                
                checkForBeats();
                
                // Check for auto-batch save periodically (not every frame)
                if (timestamp - lastBatchCheckTime > BATCH_CHECK_INTERVAL_MS) {
                    lastBatchCheckTime = timestamp;
                    checkAutoBatch();
                    updateBatchStatus();
                }
            }
            
            if (currentIndex < ecgData.length) {
                animationId = requestAnimationFrame(animate);
            } else {
                isRunning = false;
                document.getElementById('currentStatus').textContent = 'Complete!';
                // Final batch save on completion
                if (currentIndex - lastBatchEndSample > MIN_BATCH_SAMPLES) {
                    saveBatch(lastBatchEndSample, currentIndex);
                }
            }
        }
        
        // Control functions
        async function startSimulation() {
            if (ecgData.length === 0) {
                await loadData();
            }
            isRunning = true;
            lastFrameTime = performance.now();
            animationId = requestAnimationFrame(animate);
        }
        
        function stopSimulation() {
            isRunning = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            
            // AUTO-SAVE: Save any remaining unsaved data as final batch
            const unsavedSamples = currentIndex - lastBatchEndSample;
            if (unsavedSamples >= MIN_BATCH_SAMPLES) {
                console.log('[ECG] Auto-saving final batch on stop...');
                saveBatch(lastBatchEndSample, currentIndex);
            }
            
            // Update batch status - user can click Download button when ready
            updateBatchStatus();
        }
        
        async function resetSimulation() {
            stopSimulation();
            currentIndex = 0;
            classifications = [];
            falseDetections = [];
            beatTimes = [];
            currentBeatWaveform = null;
            viewOffset = 0;
            isLive = true;
            
            // Reset high-speed stability tracking
            isClassifying = false;
            classificationQueue = [];
            processedBeats.clear();
            
            // Reset Y-axis tracking (allow scale to adjust from start)
            globalMinVal = Infinity;
            globalMaxVal = -Infinity;
            
            // Reset graph height tracking
            maxGraphHeight = MIN_GRAPH_HEIGHT;
            
            // Reset backend state (clears beat_buffer for context-aware models)
            try {
                await fetch('/api/reset', { method: 'POST' });
            } catch (e) {
                console.error('Failed to reset backend:', e);
            }
            
            // Reset batch state
            savedBatches = [];
            lastBatchEndSample = 0;
            updateBatchStatus();
            
            document.getElementById('totalBeats').textContent = '0';
            document.getElementById('normalBeats').textContent = '0';
            document.getElementById('abnormalBeats').textContent = '0';
            document.getElementById('accuracy').textContent = '--';
            document.getElementById('heartRate').textContent = '--';
            document.getElementById('falseCount').textContent = '0';
            document.getElementById('currentStatus').textContent = 'Waiting...';
            document.getElementById('currentStatus').className = 'value';
            document.getElementById('probBar').style.width = '0%';
            document.getElementById('probText').textContent = 'Abnormal Probability: --';
            document.getElementById('classificationList').innerHTML = 
                '<p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>';
            document.getElementById('falseDetectionList').innerHTML = 
                '<p style="color: #888; text-align: center;">No false detections yet.</p>';
            document.getElementById('currentTime').textContent = '0:00.000';
            
            updateHistoryUI();
            
            // Reset beat snapshot
            document.getElementById('beatTypeDisplay').textContent = '--';
            document.getElementById('beatTypeDisplay').style.color = '#00ff88';
            document.getElementById('groundTruthDisplay').textContent = '--';
            document.getElementById('groundTruthDisplay').style.color = '#00ff88';
            document.getElementById('predictionDisplay').textContent = '--';
            document.getElementById('predictionDisplay').style.color = '#00ff88';
            
            // Clear beat canvas
            const width = beatCanvas.getBoundingClientRect().width;
            const height = beatCanvas.getBoundingClientRect().height;
            beatCtx.fillStyle = '#0a0a1a';
            beatCtx.fillRect(0, 0, width, height);
            
            drawECG();
        }
        
        // Load model info
        async function loadModelInfo() {
            try {
                const response = await fetch('/api/model_info');
                const info = await response.json();
                document.getElementById('modelName').textContent = info.name;
            } catch (e) {
                console.error('Failed to load model info:', e);
            }
        }
        
        // Initialize
        loadModelInfo();
        loadData().then(() => {
            drawECG();
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    return jsonify({
        'signal': ecg_data.tolist(),
        'annotations': annotations.to_dict('records')
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    r_peak = data['r_peak']
    beat_type = data['beat_type']
    
    result = extract_and_classify_beat(ecg_data, r_peak, beat_type)
    return jsonify(result)


@app.route('/api/model_info')
def get_model_info():
    return jsonify({
        'name': model_config['name'],
        'onnx_file': model_config['onnx_file'],
        'scaler_file': model_config['scaler_file'],
    })


@app.route('/api/reset', methods=['POST'])
def reset_backend():
    global beat_buffer
    beat_buffer = []
    return jsonify({
        'status': 'ok',
        'message': 'Backend state reset (beat_buffer cleared)'
    })


def main():
    parser = argparse.ArgumentParser(description='ECG Real-Time Classification Frontend')
    parser.add_argument('--model', '-m', type=str, default='v3', choices=['v2', 'v3', 'v5', 'v6'],
                        help='Model version to use: v2 (CNN), v3 (LSTM), v5 (Transformer), v6 (Context-Aware CNN1D). Default: v3')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port to run the server on. Default: 5000')
    parser.add_argument('--training-data', action='store_true',
                        help='Use demo training data instead of record 119. (Deprecated)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ECG Real-Time Classification Frontend")
    print("Using PyTorch ONNX Models")
    print("=" * 60)
    
    print(f"\nSelected model: {args.model.upper()}")
    if args.model == 'v6':
        print("  Context-Aware CNN1D: Uses 7-beat rolling buffer (3 prev + center + 3 next)")
        print("  Beat extraction: 200 samples (90 before + 110 after R-peak)")
        print("  Normalization: Flatten 7x200 → scale → reshape to (7, 200)")
        print("  First 3 beats will show 'WAITING' status until buffer is full")
    else:
        print(f"  Single-beat classification: 188 samples (70 before + 118 after R-peak)")
    
    use_record_119 = not args.training_data
    use_training_data = args.training_data
    
    print("  Data: Using MIT-BIH record 119 (excluded from training - true validation)")
    
    print("Loading data and model...")
    load_data(model_version=args.model, use_training_data=use_training_data, use_record_119=use_record_119)
    
    print(f"\nStarting web server on port {args.port}...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
