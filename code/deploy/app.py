import os
import argparse
from flask import Flask, send_from_directory, jsonify, request

from backend import ECGStreamer, InferenceEngine, EvaluationLayer

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'public')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

streamer: ECGStreamer = None
engine: InferenceEngine = None
evaluator: EvaluationLayer = None


def init_backend(model_version: str = 'v3', record: str = '119') -> None:
    global streamer, engine, evaluator
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    signal_path = os.path.join(sample_dir, f'{record}.csv')
    annotation_path = os.path.join(sample_dir, f'{record}annotations.txt')
    
    streamer = ECGStreamer(signal_path, annotation_path)
    print(f"✓ ECG Streamer loaded: {streamer.total_samples} samples, "
          f"{streamer.total_duration:.1f}s duration")
    
    engine = InferenceEngine(model_version, sample_dir)
    model_info = engine.get_model_info()
    print(f"✓ Inference Engine loaded: {model_info['name']}")
    print(f"  - Beat length: {model_info['beat_length']} samples")
    print(f"  - Context-aware: {model_info['context_aware']}")
    if model_info['context_aware']:
        print(f"  - Context window: {model_info['context_window_size']} beats")
    
    evaluator = EvaluationLayer()
    print("✓ Evaluation Layer initialized")


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/ecg/stream')
def ecg_stream():
    window_seconds = request.args.get('window_seconds', 5.0, type=float)
    end_sample = request.args.get('end_sample', None, type=int)
    
    window_samples = int(window_seconds * streamer.sampling_rate)
    window = streamer.get_window(window_samples, end_sample)
    
    return jsonify(window)


@app.route('/ecg/infer', methods=['POST'])
def ecg_infer():
    data = request.json
    r_peak = data.get('r_peak')
    beat_type = data.get('beat_type', 'N')
    
    if r_peak is None:
        return jsonify({'error': 'r_peak is required'}), 400
    
    signal = streamer.get_full_signal()
    
    result = engine.classify_beat(signal, r_peak)
    
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
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', None, type=int)
    
    if end is None:
        end = streamer.get_current_position()['absolute_index']
    
    annotations = streamer.get_annotations_in_range(start, end)
    return jsonify(annotations)


@app.route('/ecg/control', methods=['POST'])
def ecg_control():
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
    count = request.args.get('count', 50, type=int)
    
    return jsonify({
        'recent_results': evaluator.get_recent_results(count),
        'false_detections': evaluator.get_false_detections(count)
    })


@app.route('/ecg/data')
def ecg_data():
    return jsonify({
        'signal': streamer.get_full_signal().tolist(),
        'annotations': streamer.get_all_annotations().to_dict('records')
    })


@app.route('/api/data')
def api_data():
    return ecg_data()


@app.route('/api/classify', methods=['POST'])
def api_classify():
    return ecg_infer()


@app.route('/api/model_info')
def api_model_info():
    return jsonify(engine.get_model_info())


def main():
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
