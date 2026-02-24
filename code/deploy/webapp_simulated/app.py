
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request

try:
    import onnxruntime as ort
    USE_ONNX = True
except ImportError:
    print("Error: ONNXRuntime not found.")
    print("Install ONNXRuntime for ONNX model inference: pip install onnxruntime")
    sys.exit(1)

MODEL_CONFIGS = {
    'context_aware': {
        'name': 'Context-Aware CNN1D',
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

BEAT_LENGTH_context_aware = 200
PRE_SAMPLES_context_aware = 90
POST_SAMPLES_context_aware = 110
CONTEXT_WINDOW_SIZE = 7
SAMPLING_RATE = 360
NORMAL_BEAT_TYPE = 'N'

app = Flask(__name__, static_folder='static', template_folder='templates')
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


def load_data(model_version='context_aware', use_training_data=False, use_record_119=True):
    global ecg_data, annotations, model, scaler, model_config, beat_buffer

    beat_buffer = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'model')

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
        print(f"Unknown model version '{model_version}'. Using context_aware (Context-Aware CNN1D) as default.")
        model_version = 'context_aware'

    model_config = MODEL_CONFIGS[model_version]
    print(f"\nLoading {model_config['name']} model...")

    onnx_model_path = os.path.join(sample_dir, model_config['onnx_file'])
    if os.path.exists(onnx_model_path):
        print(f"Loading ONNX model from: {onnx_model_path}")
        model = ort.InferenceSession(onnx_model_path)
        print(f"[OK] {model_config['name']} ONNX model loaded successfully")
    else:
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    scaler_path = os.path.join(sample_dir, model_config['scaler_file'])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler loaded from: {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")



def extract_beat_context_aware(signal, r_peak_idx):
    start_idx = r_peak_idx - PRE_SAMPLES_context_aware
    end_idx = r_peak_idx + POST_SAMPLES_context_aware

    if start_idx < 0:
        pad_before = -start_idx
        beat = np.zeros(BEAT_LENGTH_context_aware, dtype=np.float32)
        available = signal[:end_idx]
        beat[pad_before:pad_before + len(available)] = available
    elif end_idx > len(signal):
        beat = np.zeros(BEAT_LENGTH_context_aware, dtype=np.float32)
        available = signal[start_idx:]
        beat[:len(available)] = available
    else:
        beat = signal[start_idx:end_idx].astype(np.float32)

    return beat


def extract_and_classify_beat(signal, r_peak_idx, beat_type):
    global beat_buffer

    beat = extract_beat_context_aware(signal, r_peak_idx)

    beat_buffer.append((beat, beat_type, r_peak_idx))

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
            'beat_waveform': beat.tolist(),
            'buffer_size': len(beat_buffer),
            'context_aware': True
        }

    # get the center beat which is the real obne being classified, noy the latest input
    center_idx = CONTEXT_WINDOW_SIZE // 2
    center_beat = beat_buffer[center_idx][0]
    center_beat_type = beat_buffer[center_idx][1]
    center_r_peak = beat_buffer[center_idx][2]

    context_beats = np.stack([b for b, _, _ in beat_buffer], axis=0)

    flat_size = CONTEXT_WINDOW_SIZE * BEAT_LENGTH_context_aware
    context_flat = context_beats.reshape(1, flat_size)

    normalized = scaler.transform(context_flat).astype(np.float32)

    context_input = normalized.reshape(1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH_context_aware)

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

    result = {
        'r_peak': center_r_peak,
        'beat_type': center_beat_type,
        'ground_truth': ground_truth,
        'predicted': predicted_label,
        'probability': round(prob_abnormal, 4),
        'correct': ground_truth == predicted_label,
        'beat_waveform': center_beat.tolist(),
        'r_peak_pos_in_beat': PRE_SAMPLES_context_aware,
        'beat_length': BEAT_LENGTH_context_aware,
        'context_aware': True,
        'buffer_size': len(beat_buffer)
    }

    return result


@app.route('/')
def index():
    return render_template('index.html')


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


def ensure_loaded():
    """Load data and model if not already loaded (for gunicorn compatibility)."""
    if ecg_data is None:
        print("Loading data and model...")
        load_data(model_version='context_aware')


def main():
    parser = argparse.ArgumentParser(description='ECG Real-Time Classification Frontend')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port to run the server on. Default: 5000')
    parser.add_argument('--training-data', action='store_true',
                        help='Use demo training data instead of record 119. (Deprecated)')
    args = parser.parse_args()

    print("\nModel: Context-Aware CNN1D ")

    use_record_119 = not args.training_data
    use_training_data = args.training_data

    print("  Data: Using MIT-BIH record 119 (excluded from training - true validation)")

    print("Loading data and model...")
    load_data(model_version='context_aware', use_training_data=use_training_data, use_record_119=use_record_119)

    host = '0.0.0.0' if os.environ.get('DOCKER') else '127.0.0.1'
    print(f"\nStarting web server on port {args.port}...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host=host, port=args.port, debug=False, threaded=True)


# Load data on module import so gunicorn workers have the model ready.
# When run directly via `python app.py`, main() will reload with CLI args.
ensure_loaded()


if __name__ == '__main__':
    main()
