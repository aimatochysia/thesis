"""
ECG Real-Time Classification Frontend

A mini web-based frontend that simulates real-time ECG monitoring and classification.
Features:
- Real-time ECG signal visualization
- Automatic heartbeat detection at R-peaks
- AI model classification while time continues in background
- Live classification results display

Usage:
    python realtime_frontend.py
    Then open http://localhost:5000 in your browser
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, jsonify, request

# ONNX Runtime import for cross-platform inference without Keras dependency
try:
    import onnxruntime as ort
    USE_ONNX = True
except ImportError:
    print("Warning: ONNXRuntime not found. Falling back to TensorFlow/Keras.")
    print("Install ONNXRuntime for better cross-platform support: pip install onnxruntime")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        USE_ONNX = False
    except ImportError:
        print("Error: Neither ONNXRuntime nor TensorFlow is available.")
        print("Install one of them:")
        print("  pip install onnxruntime  (recommended, lightweight)")
        print("  pip install tensorflow   (heavier, but works if ONNX model not available)")
        sys.exit(1)

# Constants
BEAT_LENGTH = 188
PRE_SAMPLES = 70
POST_SAMPLES = 118
SAMPLING_RATE = 360  # Hz - MIT-BIH standard sampling rate
NORMAL_BEAT_TYPES = {'N'}
ABNORMAL_BEAT_TYPES = {'A', 'V', 'F', 'S', 'Q', '!', 'E', 'J', 'L', 'R'}

# Global state
app = Flask(__name__)
ecg_data = None
annotations = None
model = None
scaler = None
current_sample = 0
classification_results = []
is_running = False
speed_multiplier = 10  # Speed up simulation (10x faster)


def load_data():
    """Load ECG signal, annotations, model, and scaler."""
    global ecg_data, annotations, model, scaler
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # Load signal from original CSV
    signal_path = os.path.join(sample_dir, '100.csv')
    df = pd.read_csv(signal_path)
    df.columns = df.columns.str.strip().str.strip("'")
    ecg_data = df['MLII'].values.astype(np.float32)
    
    # Load annotations
    annotation_path = os.path.join(sample_dir, '100annotations.txt')
    annotations_list = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 4:
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
    
    # Model file paths (defined once for reuse)
    onnx_model_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    h5_model_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    keras_model_path = os.path.join(sample_dir, 'ecg_lstm_v3_final.keras')
    
    # Load model - try ONNX first (preferred), fallback to Keras
    if USE_ONNX:
        # Try to load ONNX model
        if os.path.exists(onnx_model_path):
            print(f"Loading ONNX model from: {onnx_model_path}")
            model = ort.InferenceSession(onnx_model_path)
            print("ONNX model loaded successfully (Keras-free inference)")
        elif os.path.exists(h5_model_path):
            print(f"ONNX model not found. To use ONNX (recommended for cross-platform):")
            print(f"  Convert {h5_model_path} to ONNX format")
            print("Attempting to load H5 model with TensorFlow...")
            import tensorflow as tf
            from tensorflow.keras.models import load_model as keras_load_model
            model = keras_load_model(h5_model_path)
            print("H5 model loaded with TensorFlow/Keras")
        else:
            raise FileNotFoundError(f"No model found. Looking for:\n  {onnx_model_path}\n  {h5_model_path}")
    else:
        # Use Keras model
        if os.path.exists(keras_model_path):
            model = load_model(keras_model_path)
        elif os.path.exists(h5_model_path):
            model = load_model(h5_model_path)
        else:
            raise FileNotFoundError(f"No Keras model found at:\n  {keras_model_path}\n  {h5_model_path}")
    
    # Load scaler
    scaler_path = os.path.join(sample_dir, 'scaler_v3.pkl')
    scaler = joblib.load(scaler_path)
    
    print(f"Loaded {len(ecg_data)} ECG samples")
    print(f"Loaded {len(annotations)} annotations")


def extract_and_classify_beat(signal, r_peak_idx, beat_type):
    """Extract beat at R-peak and classify it."""
    start_idx = r_peak_idx - PRE_SAMPLES
    end_idx = r_peak_idx + POST_SAMPLES
    
    # Handle edge cases
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
    
    # Normalize
    beat_2d = beat.reshape(1, -1)
    normalized = scaler.transform(beat_2d).flatten().astype(np.float32)
    
    # Classify - handle both ONNX and Keras models
    beat_input = normalized.reshape(1, BEAT_LENGTH, 1)
    
    # Check if model is an ONNX Runtime session (using hasattr to avoid NameError)
    if USE_ONNX and hasattr(model, 'run') and hasattr(model, 'get_inputs'):
        # ONNX model inference
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        proba = model.run([output_name], {input_name: beat_input})[0]
    else:
        # Keras model inference
        proba = model.predict(beat_input, verbose=0)
    
    if proba.shape[1] == 2:
        prob_abnormal = float(proba[0, 1])
    else:
        prob_abnormal = float(proba[0, 0])
    
    predicted_class = 1 if prob_abnormal >= 0.5 else 0
    predicted_label = "ABNORMAL" if predicted_class == 1 else "NORMAL"
    
    # Get ground truth
    if beat_type in NORMAL_BEAT_TYPES:
        ground_truth = "NORMAL"
    elif beat_type in ABNORMAL_BEAT_TYPES:
        ground_truth = "ABNORMAL"
    else:
        ground_truth = "UNKNOWN"
    
    return {
        'r_peak': r_peak_idx,
        'beat_type': beat_type,
        'ground_truth': ground_truth,
        'predicted': predicted_label,
        'probability': round(prob_abnormal, 4),
        'correct': ground_truth == predicted_label if ground_truth != "UNKNOWN" else None
    }


# HTML Template with embedded JavaScript for real-time visualization
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Real-Time Classification</title>
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
            height: 300px;
            background: #0a0a1a;
            border-radius: 10px;
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
            gap: 10px;
            color: #888;
        }
        #speedSlider {
            width: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🫀 ECG Real-Time Classification Monitor</h1>
        
        <div class="controls">
            <button id="startBtn" onclick="startSimulation()">▶ Start</button>
            <button id="stopBtn" onclick="stopSimulation()">⏹ Stop</button>
            <button id="resetBtn" onclick="resetSimulation()">🔄 Reset</button>
            <div class="speed-control">
                <span>Speed:</span>
                <input type="range" id="speedSlider" min="1" max="50" value="10">
                <span id="speedValue">10x</span>
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
        </div>
        
        <div class="ecg-container">
            <canvas id="ecgCanvas"></canvas>
            <div class="time-display">Time: <span id="currentTime">0:00.000</span></div>
        </div>
        
        <div class="results-container">
            <div class="panel">
                <h3>📊 Current Classification</h3>
                <div class="current-beat">
                    <div class="label">Latest Heartbeat Status</div>
                    <div class="value" id="currentStatus">Waiting...</div>
                    <div class="probability-bar">
                        <div class="probability-fill" id="probBar" style="width: 0%; background: #00ff88;"></div>
                    </div>
                    <div id="probText" style="margin-top: 10px; color: #888;">Abnormal Probability: --</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>📋 Classification History</h3>
                <div class="classification-list" id="classificationList">
                    <p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('ecgCanvas');
        const ctx = canvas.getContext('2d');
        
        let ecgData = [];
        let annotations = [];
        let currentIndex = 0;
        let isRunning = false;
        let animationId = null;
        let displayBuffer = [];
        let classifications = [];
        let lastBeatTime = 0;
        let speedMultiplier = 10;
        
        const SAMPLING_RATE = 360;
        const DISPLAY_SECONDS = 5;
        const DISPLAY_SAMPLES = SAMPLING_RATE * DISPLAY_SECONDS;
        
        // Resize canvas to be pixel-perfect
        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Speed slider
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            speedMultiplier = parseInt(e.target.value);
            document.getElementById('speedValue').textContent = speedMultiplier + 'x';
        });
        
        // Load data from server
        async function loadData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            ecgData = data.signal;
            annotations = data.annotations;
            console.log(`Loaded ${ecgData.length} ECG samples and ${annotations.length} annotations`);
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
            
            if (displayBuffer.length < 2) return;
            
            // Find min/max for scaling
            const minVal = Math.min(...displayBuffer);
            const maxVal = Math.max(...displayBuffer);
            const range = maxVal - minVal || 1;
            
            // Draw ECG line
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < displayBuffer.length; i++) {
                const x = (i / DISPLAY_SAMPLES) * width;
                const y = height - ((displayBuffer[i] - minVal) / range) * (height - 40) - 20;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Draw R-peak markers
            const startSample = currentIndex - displayBuffer.length;
            annotations.forEach(ann => {
                if (ann.sample_index > startSample && ann.sample_index <= currentIndex) {
                    const bufferIdx = ann.sample_index - startSample;
                    if (bufferIdx >= 0 && bufferIdx < displayBuffer.length) {
                        const x = (bufferIdx / DISPLAY_SAMPLES) * width;
                        const y = height - ((displayBuffer[bufferIdx] - minVal) / range) * (height - 40) - 20;
                        
                        // Draw marker
                        ctx.fillStyle = ann.beat_type === 'N' ? '#00ff88' : '#ff4757';
                        ctx.beginPath();
                        ctx.arc(x, y, 6, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });
        }
        
        // Update time display
        function updateTime() {
            const seconds = currentIndex / SAMPLING_RATE;
            const minutes = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(3);
            document.getElementById('currentTime').textContent = 
                `${minutes}:${secs.padStart(6, '0')}`;
        }
        
        // Check for beats and classify
        async function checkForBeats() {
            const prevSample = currentIndex - speedMultiplier;
            
            for (const ann of annotations) {
                if (ann.sample_index > prevSample && ann.sample_index <= currentIndex && ann.beat_type !== '+') {
                    // Found a beat! Classify it
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            r_peak: ann.sample_index,
                            beat_type: ann.beat_type
                        })
                    });
                    const result = await response.json();
                    addClassification(result);
                    
                    // Calculate heart rate
                    if (lastBeatTime > 0) {
                        const beatInterval = (ann.sample_index - lastBeatTime) / SAMPLING_RATE;
                        const bpm = Math.round(60 / beatInterval);
                        document.getElementById('heartRate').textContent = bpm;
                    }
                    lastBeatTime = ann.sample_index;
                }
            }
        }
        
        // Add classification result
        function addClassification(result) {
            classifications.unshift(result);
            
            // Update stats
            const total = classifications.length;
            const normal = classifications.filter(c => c.predicted === 'NORMAL').length;
            const abnormal = total - normal;
            const correct = classifications.filter(c => c.correct === true).length;
            const known = classifications.filter(c => c.correct !== null).length;
            
            document.getElementById('totalBeats').textContent = total;
            document.getElementById('normalBeats').textContent = normal;
            document.getElementById('abnormalBeats').textContent = abnormal;
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
            
            // Update list
            const listEl = document.getElementById('classificationList');
            if (classifications.length === 1) {
                listEl.innerHTML = '';
            }
            
            const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
            const item = document.createElement('div');
            item.className = 'classification-item ' + result.predicted.toLowerCase();
            item.innerHTML = `
                <div class="beat-info">
                    <div>Beat Type: ${result.beat_type} → ${result.predicted}</div>
                    <div class="beat-time">Time: ${time}s | Prob: ${(result.probability * 100).toFixed(1)}%</div>
                </div>
                <span class="prediction-badge ${result.predicted.toLowerCase()}">${result.predicted}</span>
            `;
            listEl.insertBefore(item, listEl.firstChild);
            
            // Keep only last 50 items
            while (listEl.children.length > 50) {
                listEl.removeChild(listEl.lastChild);
            }
        }
        
        // Animation loop
        function animate() {
            if (!isRunning) return;
            
            // Advance samples based on speed
            for (let i = 0; i < speedMultiplier; i++) {
                if (currentIndex < ecgData.length) {
                    displayBuffer.push(ecgData[currentIndex]);
                    currentIndex++;
                    
                    // Keep buffer at display size
                    while (displayBuffer.length > DISPLAY_SAMPLES) {
                        displayBuffer.shift();
                    }
                }
            }
            
            drawECG();
            updateTime();
            checkForBeats();
            
            if (currentIndex < ecgData.length) {
                animationId = requestAnimationFrame(animate);
            } else {
                isRunning = false;
                document.getElementById('currentStatus').textContent = 'Complete!';
            }
        }
        
        // Control functions
        async function startSimulation() {
            if (ecgData.length === 0) {
                await loadData();
            }
            isRunning = true;
            animate();
        }
        
        function stopSimulation() {
            isRunning = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }
        
        function resetSimulation() {
            stopSimulation();
            currentIndex = 0;
            displayBuffer = [];
            classifications = [];
            lastBeatTime = 0;
            
            document.getElementById('totalBeats').textContent = '0';
            document.getElementById('normalBeats').textContent = '0';
            document.getElementById('abnormalBeats').textContent = '0';
            document.getElementById('accuracy').textContent = '--';
            document.getElementById('heartRate').textContent = '--';
            document.getElementById('currentStatus').textContent = 'Waiting...';
            document.getElementById('currentStatus').className = 'value';
            document.getElementById('probBar').style.width = '0%';
            document.getElementById('probText').textContent = 'Abnormal Probability: --';
            document.getElementById('classificationList').innerHTML = 
                '<p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>';
            document.getElementById('currentTime').textContent = '0:00.000';
            
            drawECG();
        }
        
        // Initialize
        loadData().then(() => {
            drawECG();
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    """Return ECG signal and annotations as JSON."""
    return jsonify({
        'signal': ecg_data.tolist(),
        'annotations': annotations.to_dict('records')
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a single beat."""
    data = request.json
    r_peak = data['r_peak']
    beat_type = data['beat_type']
    
    result = extract_and_classify_beat(ecg_data, r_peak, beat_type)
    return jsonify(result)


def main():
    """Run the real-time frontend."""
    print("=" * 60)
    print("ECG Real-Time Classification Frontend")
    print("=" * 60)
    
    print("\nLoading data...")
    load_data()
    
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
