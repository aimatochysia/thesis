"""
ECG Real-Time Classification Backend Application

A thesis-quality Flask application for real-time ECG arrhythmia detection.
Features modular architecture with clear separation of concerns:

- ECGStreamer: Handles signal loading and real-time simulation
- InferenceEngine: ONNX model loading and beat classification
- EvaluationLayer: Ground truth comparison and metrics calculation

API Endpoints:
- GET /: Serves the frontend HTML/JS
- GET /ecg/stream: Returns next ECG window with samples, timestamps, position
- POST /ecg/infer: Classifies a beat given R-peak position
- GET /ecg/status: Returns current system status (time, model info, metrics)
- GET /ecg/annotations: Returns annotations in a sample range
- POST /ecg/control: Control playback (start, stop, reset, speed)

Designed for VPS deployment with persistent backend state.

Usage:
    python app.py              # Default: v3 (LSTM) model
    python app.py --model v6   # Context-Aware CNN1D
    python app.py --port 8080  # Custom port

Then open http://localhost:5000 in your browser.
"""

import os
import argparse
from flask import Flask, render_template_string, jsonify, request

# Import backend modules
from backend import ECGStreamer, InferenceEngine, EvaluationLayer

# Flask application
app = Flask(__name__)

# Global state (managed by Flask application context)
streamer: ECGStreamer = None
engine: InferenceEngine = None
evaluator: EvaluationLayer = None


def init_backend(model_version: str = 'v3', record: str = '119') -> None:
    """
    Initialize backend modules.
    
    Args:
        model_version: Model to use ('v2', 'v3', 'v5', 'v6')
        record: MIT-BIH record number to use (default: '119' - excluded from v6 training)
    """
    global streamer, engine, evaluator
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # Initialize ECG Streamer
    signal_path = os.path.join(sample_dir, f'{record}.csv')
    annotation_path = os.path.join(sample_dir, f'{record}annotations.txt')
    
    streamer = ECGStreamer(signal_path, annotation_path)
    print(f"✓ ECG Streamer loaded: {streamer.total_samples} samples, "
          f"{streamer.total_duration:.1f}s duration")
    
    # Initialize Inference Engine
    engine = InferenceEngine(model_version, sample_dir)
    model_info = engine.get_model_info()
    print(f"✓ Inference Engine loaded: {model_info['name']}")
    print(f"  - Beat length: {model_info['beat_length']} samples")
    print(f"  - Context-aware: {model_info['context_aware']}")
    if model_info['context_aware']:
        print(f"  - Context window: {model_info['context_window_size']} beats")
    
    # Initialize Evaluation Layer
    evaluator = EvaluationLayer()
    print("✓ Evaluation Layer initialized")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template_string(get_frontend_html())


@app.route('/ecg/stream')
def ecg_stream():
    """
    Get ECG window for streaming display.
    
    Query Parameters:
        window_seconds: Window duration in seconds (default: 5.0)
        end_sample: End sample index (default: current position)
    
    Returns:
        JSON with samples[], timestamps[], start_index, end_index, sampling_rate
    """
    window_seconds = request.args.get('window_seconds', 5.0, type=float)
    end_sample = request.args.get('end_sample', None, type=int)
    
    window_samples = int(window_seconds * streamer.sampling_rate)
    window = streamer.get_window(window_samples, end_sample)
    
    return jsonify(window)


@app.route('/ecg/infer', methods=['POST'])
def ecg_infer():
    """
    Classify a beat at the given R-peak position.
    
    Request Body (JSON):
        r_peak: R-peak sample index
        beat_type: Optional ground truth beat type for evaluation
    
    Returns:
        JSON with predicted, probability, r_peak, beat_waveform, etc.
    """
    data = request.json
    r_peak = data.get('r_peak')
    beat_type = data.get('beat_type', 'N')  # Default to N if not provided
    
    if r_peak is None:
        return jsonify({'error': 'r_peak is required'}), 400
    
    # Get signal from streamer
    signal = streamer.get_full_signal()
    
    # Run inference
    result = engine.classify_beat(signal, r_peak)
    
    # Add to evaluation if beat_type provided and not WAITING
    if result['predicted'] != 'WAITING':
        eval_result = evaluator.add_result(
            r_peak=r_peak,
            beat_type=beat_type,
            predicted=result['predicted'],
            probability=result['probability'],
            beat_waveform=result.get('beat_waveform'),
            r_peak_pos_in_beat=result.get('r_peak_pos_in_beat', 70),
            beat_length=result.get('beat_length', 188)
        )
        if eval_result:
            result['ground_truth'] = eval_result.ground_truth
            result['correct'] = eval_result.correct
    
    return jsonify(result)


@app.route('/ecg/status')
def ecg_status():
    """
    Get current system status.
    
    Returns:
        JSON with:
            - simulation: playback state (running, position, speed)
            - model: model information
            - metrics: current performance metrics
            - signal: signal information (total samples, duration)
    """
    position = streamer.get_current_position()
    model_info = engine.get_model_info()
    metrics = evaluator.get_metrics()
    
    return jsonify({
        'simulation': {
            'running': streamer.is_running,
            'current_sample': position['absolute_index'],
            'current_time': round(position['time_seconds'], 3),
            'progress': round(position['progress'], 4),
            'speed': streamer.speed_multiplier
        },
        'model': model_info,
        'metrics': metrics,
        'signal': {
            'total_samples': position['total_samples'],
            'total_duration': round(position['total_seconds'], 1),
            'sampling_rate': streamer.sampling_rate
        }
    })


@app.route('/ecg/annotations')
def ecg_annotations():
    """
    Get annotations in a sample range.
    
    Query Parameters:
        start: Start sample index (default: 0)
        end: End sample index (default: current position)
    
    Returns:
        JSON list of annotations with sample_index, beat_type, time
    """
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', None, type=int)
    
    if end is None:
        end = streamer.get_current_position()['absolute_index']
    
    annotations = streamer.get_annotations_in_range(start, end)
    return jsonify(annotations)


@app.route('/ecg/control', methods=['POST'])
def ecg_control():
    """
    Control playback state.
    
    Request Body (JSON):
        action: 'start', 'stop', 'reset', or 'set_speed'
        speed: Speed multiplier (required for 'set_speed')
    
    Returns:
        JSON with new status
    """
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        streamer.start()
    elif action == 'stop':
        streamer.stop()
    elif action == 'reset':
        streamer.reset()
        engine.reset_buffer()
        evaluator.reset()
    elif action == 'set_speed':
        speed = data.get('speed', 1.0)
        streamer.set_speed(speed)
    else:
        return jsonify({'error': f'Unknown action: {action}'}), 400
    
    return jsonify({'success': True, 'running': streamer.is_running})


@app.route('/ecg/results')
def ecg_results():
    """
    Get classification results and false detections.
    
    Query Parameters:
        count: Number of results to return (default: 50)
    
    Returns:
        JSON with recent_results[] and false_detections[]
    """
    count = request.args.get('count', 50, type=int)
    
    return jsonify({
        'recent_results': evaluator.get_recent_results(count),
        'false_detections': evaluator.get_false_detections(count)
    })


@app.route('/ecg/data')
def ecg_data():
    """
    Get full signal and annotations for initial load.
    
    Returns:
        JSON with signal[] and annotations[]
    """
    return jsonify({
        'signal': streamer.get_full_signal().tolist(),
        'annotations': streamer.get_all_annotations().to_dict('records')
    })


@app.route('/api/data')
def api_data():
    """Legacy endpoint for backward compatibility."""
    return ecg_data()


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """Legacy endpoint for backward compatibility."""
    return ecg_infer()


@app.route('/api/model_info')
def api_model_info():
    """Legacy endpoint for backward compatibility."""
    return jsonify(engine.get_model_info())


# ============================================================
# FRONTEND HTML TEMPLATE
# ============================================================

def get_frontend_html():
    """Return the frontend HTML template."""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Real-Time Classification</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
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
            flex-wrap: wrap;
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
        #startBtn { background: linear-gradient(45deg, #00ff88, #00cc6a); color: #1a1a2e; }
        #stopBtn { background: linear-gradient(45deg, #ff4757, #ff3838); color: white; }
        #resetBtn { background: linear-gradient(45deg, #5352ed, #3742fa); color: white; }
        button:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3); }
        
        .stats-bar {
            display: flex;
            justify-content: space-around;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat-item { text-align: center; min-width: 80px; }
        .stat-value { font-size: 28px; font-weight: bold; color: #00ff88; }
        .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
        
        .ecg-container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        #ecgCanvas { width: 100%; height: 300px; background: #0a0a1a; border-radius: 10px; }
        .time-display {
            text-align: center;
            font-size: 24px;
            color: #00ff88;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
        }
        
        .nav-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .nav-btn {
            padding: 5px 15px;
            font-size: 12px;
            border-radius: 15px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            cursor: pointer;
        }
        .nav-btn:hover { background: rgba(0,255,136,0.2); }
        .nav-btn.live { background: linear-gradient(45deg, #ff4757, #ff3838); border: none; font-weight: bold; }
        
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
        .speed-btn.active { background: rgba(0,255,136,0.3); border-color: #00ff88; }
        .speed-btn:hover { background: rgba(0,255,136,0.2); }
        
        .model-badge {
            background: linear-gradient(45deg, #00ff88, #00cc6a);
            color: #1a1a2e;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: bold;
        }
        
        .beat-snapshot-container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        #beatCanvas { width: 100%; height: 150px; background: #0a0a1a; border-radius: 10px; }
        
        .results-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) { .results-container { grid-template-columns: 1fr; } }
        
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
        
        .classification-list { max-height: 300px; overflow-y: auto; }
        .classification-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .classification-item:hover { transform: translateX(5px); }
        .classification-item.normal { background: rgba(0, 255, 136, 0.2); border-left: 4px solid #00ff88; }
        .classification-item.abnormal { background: rgba(255, 71, 87, 0.2); border-left: 4px solid #ff4757; }
        .classification-item.false { border: 2px solid #ffd700; }
        
        .prediction-badge {
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .prediction-badge.normal { background: #00ff88; color: #1a1a2e; }
        .prediction-badge.abnormal { background: #ff4757; color: white; }
        
        .current-beat { text-align: center; padding: 30px; }
        .current-beat .label { font-size: 14px; color: #888; margin-bottom: 10px; }
        .current-beat .value { font-size: 48px; font-weight: bold; }
        .current-beat .value.normal { color: #00ff88; }
        .current-beat .value.abnormal { color: #ff4757; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .probability-bar {
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin-top: 15px;
            overflow: hidden;
        }
        .probability-fill { height: 100%; border-radius: 10px; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🫀 ECG Real-Time Classification Monitor</h1>
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
            <div class="nav-controls">
                <button class="nav-btn" onclick="scrollHistory(-5)">⏪ -5s</button>
                <button class="nav-btn" onclick="scrollHistory(-1)">◀ -1s</button>
                <button class="nav-btn live" id="liveBtn" onclick="goToLive()">🔴 Live</button>
                <button class="nav-btn" id="fwdBtn" onclick="scrollHistory(1)" disabled>▶ +1s</button>
                <button class="nav-btn" id="fwd5Btn" onclick="scrollHistory(5)" disabled>⏩ +5s</button>
            </div>
        </div>
        
        <div class="beat-snapshot-container">
            <h3 style="color: #00ff88; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 255, 136, 0.3); padding-bottom: 10px;">💓 Current Beat Snapshot (Input to ONNX Model)</h3>
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px;">
                    <canvas id="beatCanvas"></canvas>
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
        
        <div class="panel" style="margin-top: 20px; border-left: 4px solid #ffd700;">
            <h3 style="color: #ffd700 !important;">⚠️ False Detections</h3>
            <div class="classification-list" id="falseDetectionList">
                <p style="color: #888; text-align: center;">No false detections yet.</p>
            </div>
        </div>
    </div>
    
    <script>
        // Canvas setup
        const canvas = document.getElementById('ecgCanvas');
        const ctx = canvas.getContext('2d');
        const beatCanvas = document.getElementById('beatCanvas');
        const beatCtx = beatCanvas.getContext('2d');
        
        // State
        let ecgData = [];
        let annotations = [];
        let currentIndex = 0;
        let isRunning = false;
        let animationId = null;
        let classifications = [];
        let falseDetections = [];
        let speedMultiplier = 1;
        let currentBeatWaveform = null;
        let currentRPeakPos = 70;
        let viewOffset = 0;
        let isLive = true;
        
        const SAMPLING_RATE = 360;
        const DISPLAY_SECONDS = 5;
        const DISPLAY_SAMPLES = SAMPLING_RATE * DISPLAY_SECONDS;
        
        // Resize canvas
        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            const beatRect = beatCanvas.getBoundingClientRect();
            beatCanvas.width = beatRect.width * window.devicePixelRatio;
            beatCanvas.height = beatRect.height * window.devicePixelRatio;
            beatCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            if (currentBeatWaveform) drawBeatWaveform(currentBeatWaveform);
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Speed control
        function setSpeed(speed) {
            speedMultiplier = speed;
            document.getElementById('speedValue').textContent = speed + 'x';
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent === speed + 'x') btn.classList.add('active');
            });
            
            fetch('/ecg/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'set_speed', speed: speed})
            });
        }
        
        // History navigation
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
            if (targetOffset >= 0) { goToLive(); return; }
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
            indicator.style.display = isLive ? 'none' : 'inline';
            fwdBtn.disabled = isLive;
            fwd5Btn.disabled = isLive;
        }
        
        // Draw ECG
        function drawECG() {
            const width = canvas.getBoundingClientRect().width;
            const height = canvas.getBoundingClientRect().height;
            
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, width, height);
            
            // Grid
            ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
            ctx.lineWidth = 1;
            for (let x = 0; x < width; x += 50) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 0; y < height; y += 50) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }
            
            let endSample = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
            let startSample = Math.max(0, endSample - DISPLAY_SAMPLES);
            
            let buffer = ecgData.slice(startSample, endSample);
            if (buffer.length < 2) return;
            
            const minVal = Math.min(...buffer);
            const maxVal = Math.max(...buffer);
            const range = maxVal - minVal || 1;
            
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < buffer.length; i++) {
                const x = (i / DISPLAY_SAMPLES) * width;
                const y = height - ((buffer[i] - minVal) / range) * (height - 40) - 20;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
            
            // R-peak markers
            annotations.forEach(ann => {
                if (ann.sample_index > startSample && ann.sample_index <= endSample) {
                    const bufferIdx = ann.sample_index - startSample;
                    if (bufferIdx >= 0 && bufferIdx < buffer.length) {
                        const x = (bufferIdx / DISPLAY_SAMPLES) * width;
                        const y = height - ((buffer[bufferIdx] - minVal) / range) * (height - 40) - 20;
                        
                        const classResult = classifications.find(c => c.r_peak === ann.sample_index);
                        if (classResult && classResult.correct === false) {
                            ctx.strokeStyle = '#ffd700';
                            ctx.lineWidth = 3;
                            ctx.beginPath();
                            ctx.arc(x, y, 10, 0, Math.PI * 2);
                            ctx.stroke();
                        }
                        
                        ctx.fillStyle = ann.beat_type === 'N' ? '#00ff88' : '#ff4757';
                        ctx.beginPath();
                        ctx.arc(x, y, 6, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            });
            
            if (!isLive) {
                ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
                ctx.font = 'bold 14px Arial';
                ctx.fillText('📜 VIEWING HISTORY', 10, 25);
            }
        }
        
        // Draw beat waveform
        function drawBeatWaveform(waveform, isAbnormal = false) {
            const width = beatCanvas.getBoundingClientRect().width;
            const height = beatCanvas.getBoundingClientRect().height;
            
            beatCtx.fillStyle = '#0a0a1a';
            beatCtx.fillRect(0, 0, width, height);
            
            beatCtx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
            beatCtx.lineWidth = 1;
            for (let x = 0; x < width; x += 30) {
                beatCtx.beginPath(); beatCtx.moveTo(x, 0); beatCtx.lineTo(x, height); beatCtx.stroke();
            }
            for (let y = 0; y < height; y += 30) {
                beatCtx.beginPath(); beatCtx.moveTo(0, y); beatCtx.lineTo(width, y); beatCtx.stroke();
            }
            
            if (!waveform || waveform.length < 2) return;
            
            const minVal = Math.min(...waveform);
            const maxVal = Math.max(...waveform);
            const range = maxVal - minVal || 1;
            
            beatCtx.strokeStyle = isAbnormal ? '#ff4757' : '#00ff88';
            beatCtx.lineWidth = 2;
            beatCtx.beginPath();
            
            for (let i = 0; i < waveform.length; i++) {
                const x = (i / waveform.length) * width;
                const y = height - ((waveform[i] - minVal) / range) * (height - 20) - 10;
                if (i === 0) beatCtx.moveTo(x, y); else beatCtx.lineTo(x, y);
            }
            beatCtx.stroke();
            
            const rPeakX = (currentRPeakPos / waveform.length) * width;
            const rPeakY = height - ((waveform[Math.min(currentRPeakPos, waveform.length-1)] - minVal) / range) * (height - 20) - 10;
            beatCtx.fillStyle = '#ffcc00';
            beatCtx.beginPath();
            beatCtx.arc(rPeakX, rPeakY, 6, 0, Math.PI * 2);
            beatCtx.fill();
            beatCtx.font = '11px Arial';
            beatCtx.fillText('R-peak', rPeakX - 18, rPeakY - 10);
        }
        
        // Update time display
        function updateTime() {
            let displayIndex = isLive ? currentIndex : Math.max(0, currentIndex + Math.round(viewOffset * SAMPLING_RATE));
            const seconds = displayIndex / SAMPLING_RATE;
            const minutes = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(3);
            document.getElementById('currentTime').textContent = `${minutes}:${secs.padStart(6, '0')}`;
        }
        
        // Load data
        async function loadData() {
            const response = await fetch('/ecg/data');
            const data = await response.json();
            ecgData = data.signal;
            annotations = data.annotations;
            console.log(`Loaded ${ecgData.length} samples and ${annotations.length} annotations`);
        }
        
        // Load model info
        async function loadModelInfo() {
            try {
                const response = await fetch('/ecg/status');
                const status = await response.json();
                document.getElementById('modelName').textContent = status.model.name;
            } catch (e) {
                console.error('Failed to load model info:', e);
            }
        }
        
        // Check for beats and classify
        // At 60 FPS, we check (speedMultiplier * 6) samples per frame
        // Formula: samples_per_frame = SAMPLING_RATE * speedMultiplier / FPS
        async function checkForBeats() {
            const TARGET_FPS = 60;  // Animation frame rate
            const samplesToCheck = Math.max(1, Math.round(speedMultiplier * (SAMPLING_RATE / TARGET_FPS)));
            const prevSample = currentIndex - samplesToCheck;
            
            for (const ann of annotations) {
                if (ann.sample_index > prevSample && ann.sample_index <= currentIndex && ann.beat_type !== '+') {
                    try {
                        const response = await fetch('/ecg/infer', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ r_peak: ann.sample_index, beat_type: ann.beat_type })
                        });
                        const result = await response.json();
                        if (result.predicted !== 'WAITING') {
                            addClassification(result);
                        }
                    } catch (e) {
                        console.error('Classification error:', e);
                    }
                }
            }
        }
        
        // Add classification result
        function addClassification(result) {
            classifications.unshift(result);
            
            if (result.correct === false) {
                falseDetections.unshift(result);
                updateFalseDetectionList();
            }
            
            // Update stats
            const total = classifications.length;
            const normal = classifications.filter(c => c.predicted === 'NORMAL').length;
            const abnormal = classifications.filter(c => c.predicted === 'ABNORMAL').length;
            const correct = classifications.filter(c => c.correct === true).length;
            
            document.getElementById('totalBeats').textContent = total;
            document.getElementById('normalBeats').textContent = normal;
            document.getElementById('abnormalBeats').textContent = abnormal;
            document.getElementById('falseCount').textContent = falseDetections.length;
            if (total > 0) {
                document.getElementById('accuracy').textContent = Math.round((correct / total) * 100) + '%';
            }
            
            // Update current status
            const statusEl = document.getElementById('currentStatus');
            statusEl.textContent = result.predicted;
            statusEl.className = 'value ' + result.predicted.toLowerCase();
            
            const prob = result.probability;
            const probBar = document.getElementById('probBar');
            probBar.style.width = (prob * 100) + '%';
            probBar.style.background = prob >= 0.5 ? '#ff4757' : '#00ff88';
            document.getElementById('probText').textContent = `Abnormal Probability: ${(prob * 100).toFixed(1)}%`;
            
            // Update beat snapshot
            if (result.beat_waveform) {
                currentBeatWaveform = result.beat_waveform;
                currentRPeakPos = result.r_peak_pos_in_beat || 70;
                drawBeatWaveform(result.beat_waveform, result.predicted === 'ABNORMAL');
                
                document.getElementById('beatTypeDisplay').textContent = result.beat_type || '--';
                document.getElementById('beatTypeDisplay').style.color = (result.beat_type === 'N') ? '#00ff88' : '#ff4757';
                document.getElementById('groundTruthDisplay').textContent = result.ground_truth || '--';
                document.getElementById('groundTruthDisplay').style.color = (result.ground_truth === 'NORMAL') ? '#00ff88' : '#ff4757';
                document.getElementById('predictionDisplay').textContent = result.predicted;
                document.getElementById('predictionDisplay').style.color = (result.predicted === 'NORMAL') ? '#00ff88' : '#ff4757';
            }
            
            // Update list
            const listEl = document.getElementById('classificationList');
            if (classifications.length === 1) listEl.innerHTML = '';
            
            const time = (result.r_peak / SAMPLING_RATE).toFixed(2);
            const item = document.createElement('div');
            item.className = 'classification-item ' + result.predicted.toLowerCase();
            if (result.correct === false) item.classList.add('false');
            item.onclick = () => navigateToTime(result.r_peak);
            item.innerHTML = `
                <div class="beat-info">
                    <div>Beat Type: ${result.beat_type || '?'} → ${result.predicted}</div>
                    <div class="beat-time" style="color: #888; font-size: 12px;">Time: ${time}s | Prob: ${(result.probability * 100).toFixed(1)}%</div>
                </div>
                <span class="prediction-badge ${result.predicted.toLowerCase()}">${result.predicted}</span>
            `;
            listEl.insertBefore(item, listEl.firstChild);
            while (listEl.children.length > 100) listEl.removeChild(listEl.lastChild);
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
                item.onmouseover = () => { item.style.background = 'rgba(255, 215, 0, 0.3)'; };
                item.onmouseout = () => { item.style.background = 'rgba(255, 215, 0, 0.15)'; };
                listEl.appendChild(item);
            });
        }
        
        // Animation loop
        let lastFrameTime = 0;
        const targetFPS = 60;
        const frameInterval = 1000 / targetFPS;
        
        function animate(timestamp) {
            if (!isRunning) return;
            
            const deltaTime = timestamp - lastFrameTime;
            if (deltaTime >= frameInterval) {
                lastFrameTime = timestamp - (deltaTime % frameInterval);
                
                const samplesPerSecond = SAMPLING_RATE * speedMultiplier;
                const samplesToAdvance = Math.max(1, Math.round(samplesPerSecond / targetFPS));
                
                for (let i = 0; i < samplesToAdvance; i++) {
                    if (currentIndex < ecgData.length) currentIndex++;
                }
                
                if (isLive) {
                    drawECG();
                    updateTime();
                }
                
                checkForBeats();
            }
            
            if (currentIndex < ecgData.length) {
                animationId = requestAnimationFrame(animate);
            } else {
                isRunning = false;
                document.getElementById('currentStatus').textContent = 'Complete!';
            }
        }
        
        // Controls
        async function startSimulation() {
            if (ecgData.length === 0) await loadData();
            isRunning = true;
            lastFrameTime = performance.now();
            animationId = requestAnimationFrame(animate);
            fetch('/ecg/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'start'})
            });
        }
        
        function stopSimulation() {
            isRunning = false;
            if (animationId) cancelAnimationFrame(animationId);
            fetch('/ecg/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'stop'})
            });
        }
        
        function resetSimulation() {
            stopSimulation();
            currentIndex = 0;
            classifications = [];
            falseDetections = [];
            currentBeatWaveform = null;
            viewOffset = 0;
            isLive = true;
            
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
            document.getElementById('classificationList').innerHTML = '<p style="color: #888; text-align: center;">No classifications yet. Start the simulation!</p>';
            document.getElementById('falseDetectionList').innerHTML = '<p style="color: #888; text-align: center;">No false detections yet.</p>';
            document.getElementById('currentTime').textContent = '0:00.000';
            document.getElementById('beatTypeDisplay').textContent = '--';
            document.getElementById('groundTruthDisplay').textContent = '--';
            document.getElementById('predictionDisplay').textContent = '--';
            
            updateHistoryUI();
            
            const width = beatCanvas.getBoundingClientRect().width;
            const height = beatCanvas.getBoundingClientRect().height;
            beatCtx.fillStyle = '#0a0a1a';
            beatCtx.fillRect(0, 0, width, height);
            
            drawECG();
            
            fetch('/ecg/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'reset'})
            });
        }
        
        // Initialize
        loadModelInfo();
        loadData().then(() => drawECG());
    </script>
</body>
</html>
'''


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Run the ECG real-time classification backend."""
    parser = argparse.ArgumentParser(description='ECG Real-Time Classification Backend')
    parser.add_argument('--model', '-m', type=str, default='v3',
                        choices=['v2', 'v3', 'v5', 'v6'],
                        help='Model version: v2 (CNN), v3 (LSTM), v5 (Transformer), v6 (Context-Aware). Default: v3')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port to run the server on. Default: 5000')
    parser.add_argument('--record', '-r', type=str, default='119',
                        help='MIT-BIH record number to use. Default: 119 (excluded from v6 training)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to. Default: 127.0.0.1')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ECG Real-Time Classification Backend")
    print("Modular Architecture for Thesis-Quality Deployment")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Record: {args.record}")
    print(f"  Host: {args.host}:{args.port}")
    
    print("\nInitializing backend modules...")
    init_backend(model_version=args.model, record=args.record)
    
    print(f"\n" + "=" * 60)
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
