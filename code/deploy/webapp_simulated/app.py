
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


def load_data(model_version='v3', use_training_data=False, use_record_119=True):
    global ecg_data, annotations, model, scaler, model_config, beat_buffer

    beat_buffer = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, '..', 'sample')

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
        print(f"[OK] {model_config['name']} ONNX model loaded successfully")
    else:
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    scaler_path = os.path.join(sample_dir, model_config['scaler_file'])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler loaded from: {scaler_path}")
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

        # get the CENTER beat (index 3) which is the one being classified
        center_beat = beat_buffer[3][0]
        center_beat_type = beat_buffer[3][1]
        center_r_peak = beat_buffer[3][2]

        context_beats = np.stack([b for b, _, _ in beat_buffer], axis=0)

        flat_size = CONTEXT_WINDOW_SIZE * BEAT_LENGTH_V6
        context_flat = context_beats.reshape(1, flat_size)

        normalized = scaler.transform(context_flat).astype(np.float32)

        context_input = normalized.reshape(1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH_V6)

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
        beat_waveform_to_return = center_beat.tolist()
        r_peak_to_return = center_r_peak
    else:
        r_peak_pos_in_beat = PRE_SAMPLES
        beat_waveform_to_return = raw_beat.tolist()
        r_peak_to_return = r_peak_idx

    result = {
        'r_peak': r_peak_to_return,
        'beat_type': center_beat_type,
        'ground_truth': ground_truth,
        'predicted': predicted_label,
        'probability': round(prob_abnormal, 4),
        'correct': ground_truth == predicted_label,
        'beat_waveform': beat_waveform_to_return,
        'r_peak_pos_in_beat': r_peak_pos_in_beat,
        'beat_length': BEAT_LENGTH_V6 if is_context_aware else BEAT_LENGTH
    }

    if is_context_aware:
        result['context_aware'] = True
        result['buffer_size'] = len(beat_buffer)

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
        print("  Normalization: Flatten 7x200 -> scale -> reshape to (7, 200)")
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
