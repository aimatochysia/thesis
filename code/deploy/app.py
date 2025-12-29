"""
ECG Real-Time Classification Backend Application

A thesis-quality Flask application for real-time ECG arrhythmia detection.
Features modular architecture with clear separation of concerns:

- ECGStreamer: Handles signal loading and real-time simulation
- InferenceEngine: ONNX model loading and beat classification
- EvaluationLayer: Ground truth comparison and metrics calculation

API Endpoints:
- GET /: Serves the frontend index.html
- GET /ecg/stream: Returns next ECG window with samples, timestamps, position
- POST /ecg/infer: Classifies a beat given R-peak position
- GET /ecg/status: Returns current system status (time, model info, metrics)
- GET /ecg/annotations: Returns annotations in a sample range
- POST /ecg/control: Control playback (start, stop, reset, speed)

Designed for VPS deployment with persistent backend state.

Architecture:
- Frontend: Node.js Express server (frontend/) - serves static files
- Backend: Python Flask API (this file) - handles ECG processing and inference

Usage (Standalone - backend serves frontend):
    python app.py              # Default: v3 (LSTM) model
    python app.py --model v6   # Context-Aware CNN1D
    python app.py --port 8080  # Custom port

Usage (Separated - with Node.js frontend):
    # Terminal 1: Start backend
    python app.py --model v6 --port 5000
    
    # Terminal 2: Start frontend
    cd frontend && npm start

Then open http://localhost:5000 (standalone) or http://localhost:3000 (separated).
"""

import os
import argparse
from flask import Flask, send_from_directory, jsonify, request

# Import backend modules
from backend import ECGStreamer, InferenceEngine, EvaluationLayer

# Flask application - serve static files from frontend/public
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'public')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

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
    return send_from_directory(FRONTEND_DIR, 'index.html')


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
    print(f"  Frontend: {FRONTEND_DIR}")
    
    print("\nInitializing backend modules...")
    init_backend(model_version=args.model, record=args.record)
    
    print(f"\n" + "=" * 60)
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
