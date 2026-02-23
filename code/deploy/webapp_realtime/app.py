import os
import sys
import argparse
import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, sosfilt
from flask import Flask, render_template, jsonify, request

try:
    import onnxruntime as ort
    USE_ONNX = True
except ImportError:
    print("Error: ONNXRuntime not found.")
    print("Install ONNXRuntime: pip install onnxruntime")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model configs (same as simulated webapp)
# ---------------------------------------------------------------------------
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
PRE_SAMPLES_V6 = 90
POST_SAMPLES_V6 = 110
CONTEXT_WINDOW_SIZE = 7
DEFAULT_SAMPLING_RATE = 360

# ---------------------------------------------------------------------------
# Pan-Tompkins configuration defaults
# ---------------------------------------------------------------------------
PT_CONFIG = {
    "bandpass_low_hz": 0.5,
    "bandpass_high_hz": 40.0,
    "derivative_kernel": np.array([-1, -2, 0, 2, 1], dtype=np.float32),
    "mwi_window_sec": 0.12,
    "refractory_ms": 200,
    "adaptive_threshold_alpha": 0.01,
    "initial_threshold": 0.0,
    "pre_frac": 0.35,
    "post_frac": 0.65,
    "pre_min_sec": 0.08,
    "pre_max_sec": 0.35,
    "post_min_sec": 0.16,
    "post_max_sec": 0.60,
    "rr_history_len": 8,
    "model_input_len": 188,
    "NORMALIZE_MODE": "per_beat_standardize",
    "target_baseline": 950.0,
    "global_scale": 100.0,
    "per_beat_eps": 1e-6,
}


# ---------------------------------------------------------------------------
# Signal processing utilities (from deployment.py)
# ---------------------------------------------------------------------------
def butter_bandpass_sos(lowcut, highcut, fs, order=2):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos


def preprocess_stream(signal, fs, bp_low, bp_high):
    sos = butter_bandpass_sos(bp_low, bp_high, fs, order=2)
    filtered = sosfilt(sos, signal)
    return filtered


# ---------------------------------------------------------------------------
# Pan-Tompkins R-peak detector (from deployment.py)
# ---------------------------------------------------------------------------
class PanTompkinsDetector:
    def __init__(self, fs, derivative_kernel, mwi_window_sec,
                 refractory_ms, adaptive_alpha, init_threshold=0.0):
        self.fs = fs
        self.deriv_kernel = derivative_kernel.astype(np.float32)
        self.mwi_window = int(max(1, round(mwi_window_sec * fs)))
        self.refractory_samples = int(round(refractory_ms * fs / 1000.0))
        self.alpha = adaptive_alpha

        self.filtered_buffer = []
        self.deriv_buffer = []
        self.square_buffer = []
        self.mwi_buffer = []

        self.threshold = init_threshold
        self.peak_mean = 0.0
        self.noise_mean = 0.0
        self.last_peak_index = -(10 ** 9)

    def step(self, filtered_sample):
        self.filtered_buffer.append(filtered_sample)
        idx = len(self.filtered_buffer) - 1

        deriv = self._compute_derivative(idx)
        self.deriv_buffer.append(deriv)

        sq = deriv * deriv
        self.square_buffer.append(sq)

        mwi_val = self._moving_window_integral(idx)
        self.mwi_buffer.append(mwi_val)

        if len(self.mwi_buffer) < self.mwi_window * 4:
            mean_mwi = np.mean(self.mwi_buffer)
            self.threshold = mean_mwi * 1.5
            return None

        detected = False
        if mwi_val > self.threshold:
            if idx - self.last_peak_index > self.refractory_samples:
                detected = True
                self.peak_mean = (1 - self.alpha) * self.peak_mean + self.alpha * mwi_val
                self.last_peak_index = idx
        else:
            self.noise_mean = (1 - self.alpha) * self.noise_mean + self.alpha * mwi_val

        self.threshold = self.noise_mean + 0.5 * (self.peak_mean - self.noise_mean)

        if detected:
            r_index = self._local_peak_refinement(idx)
            self.last_peak_index = r_index
            return r_index

        return None

    def _compute_derivative(self, idx):
        k = len(self.deriv_kernel)
        if idx < k - 1:
            return 0.0
        segment = self.filtered_buffer[idx - (k - 1): idx + 1]
        return float(np.dot(segment, self.deriv_kernel))

    def _moving_window_integral(self, idx):
        w = self.mwi_window
        start = max(0, idx - w + 1)
        segment = self.square_buffer[start: idx + 1]
        return float(np.mean(segment)) if segment else 0.0

    def _local_peak_refinement(self, idx, search_radius_samples=10):
        start = max(0, idx - search_radius_samples)
        end = min(len(self.filtered_buffer), idx + search_radius_samples + 1)
        segment = np.array(self.filtered_buffer[start:end], dtype=np.float32)
        local_max_offset = int(np.argmax(segment))
        return start + local_max_offset


# ---------------------------------------------------------------------------
# Adaptive beat segmenter (from deployment.py)
# ---------------------------------------------------------------------------
class RRAdaptiveSegmenter:
    def __init__(self, fs, pre_frac, post_frac,
                 pre_min_sec, pre_max_sec, post_min_sec, post_max_sec,
                 rr_history_len):
        self.fs = fs
        self.pre_frac = pre_frac
        self.post_frac = post_frac
        self.pre_min_samples = int(round(pre_min_sec * fs))
        self.pre_max_samples = int(round(pre_max_sec * fs))
        self.post_min_samples = int(round(post_min_sec * fs))
        self.post_max_samples = int(round(post_max_sec * fs))
        self.rr_hist_len = rr_history_len
        self.rr_history = []
        self.default_rr_samples = int(round(0.8 * fs))

    def update_rr(self, rr_samples):
        self.rr_history.append(rr_samples)
        if len(self.rr_history) > self.rr_hist_len:
            self.rr_history.pop(0)

    def median_rr(self):
        if not self.rr_history:
            return self.default_rr_samples
        return int(np.median(self.rr_history))

    def compute_window(self, r_index, total_len):
        rr = self.median_rr()
        pre = int(round(self.pre_frac * rr))
        post = int(round(self.post_frac * rr))
        pre = max(self.pre_min_samples, min(self.pre_max_samples, pre))
        post = max(self.post_min_samples, min(self.post_max_samples, post))
        start = max(0, r_index - pre)
        end = min(total_len, r_index + post)
        return start, end


# ---------------------------------------------------------------------------
# Beat resampling and normalization (from deployment.py)
# ---------------------------------------------------------------------------
def resample_linear(x, target_len):
    if len(x) == target_len:
        return x.astype(np.float32)
    xi = np.linspace(0, 1, num=len(x), endpoint=True)
    xo = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(xo, xi, x).astype(np.float32)


def normalize_beat(x, mode, target_baseline, global_scale, eps):
    if mode == "baseline_shift_scale":
        return (x - target_baseline) / (global_scale if global_scale != 0 else 1.0)
    elif mode == "per_beat_standardize":
        m = float(np.mean(x))
        s = float(np.std(x))
        s = s if s > eps else 1.0
        return (x - m) / s
    else:
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Flask app and global state
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global state
uploaded_signal = None
uploaded_fs = DEFAULT_SAMPLING_RATE
uploaded_filename = None
filtered_signal = None
processing_results = None
model_session = None
model_scaler = None
active_model_config = None
active_model_version = None
beat_buffer = []


def load_model(model_version='v3'):
    global model_session, model_scaler, active_model_config, beat_buffer, active_model_version

    beat_buffer = []

    if model_version not in MODEL_CONFIGS:
        print("Unknown model version '{}'. Using v3 (LSTM) as default.".format(model_version))
        model_version = 'v3'

    active_model_config = MODEL_CONFIGS[model_version]
    active_model_version = model_version
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, '..', 'sample')

    onnx_path = os.path.join(sample_dir, active_model_config['onnx_file'])
    if os.path.exists(onnx_path):
        model_session = ort.InferenceSession(onnx_path)
        print("[OK] Loaded {} from: {}".format(active_model_config['name'], onnx_path))
    else:
        raise FileNotFoundError("ONNX model not found: {}".format(onnx_path))

    scaler_path = os.path.join(sample_dir, active_model_config['scaler_file'])
    if os.path.exists(scaler_path):
        model_scaler = joblib.load(scaler_path)
        print("[OK] Scaler loaded from: {}".format(scaler_path))
    else:
        raise FileNotFoundError("Scaler not found: {}".format(scaler_path))


# ---------------------------------------------------------------------------
# ECG processing pipeline
# ---------------------------------------------------------------------------
def detect_r_peaks(signal, fs):
    """Run Pan-Tompkins R-peak detection on filtered signal."""
    filtered = preprocess_stream(
        signal, fs,
        PT_CONFIG["bandpass_low_hz"],
        PT_CONFIG["bandpass_high_hz"]
    )

    detector = PanTompkinsDetector(
        fs=fs,
        derivative_kernel=PT_CONFIG["derivative_kernel"],
        mwi_window_sec=PT_CONFIG["mwi_window_sec"],
        refractory_ms=PT_CONFIG["refractory_ms"],
        adaptive_alpha=PT_CONFIG["adaptive_threshold_alpha"],
        init_threshold=PT_CONFIG["initial_threshold"]
    )

    r_peaks = []
    for i in range(len(filtered)):
        r_index = detector.step(float(filtered[i]))
        if r_index is not None:
            r_peaks.append(r_index)

    return np.array(r_peaks, dtype=np.int32), filtered


def extract_beat_v6(signal, r_peak_idx):
    """Extract a 200-sample beat for v6 context-aware model."""
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


def classify_beat_single(beat_188, input_shape):
    """Classify a single 188-sample beat with the ONNX model."""
    beat_2d = beat_188.reshape(1, -1)
    normalized = model_scaler.transform(beat_2d).flatten().astype(np.float32)
    model_input = normalized.reshape(input_shape)

    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    output = model_session.run([output_name], {input_name: model_input})[0]

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

    return max(0.0, min(1.0, prob_abnormal))


def classify_beat_context(beat_list):
    """Classify with v6 context-aware model using 7-beat window."""
    context_beats = np.stack(beat_list, axis=0)
    flat_size = CONTEXT_WINDOW_SIZE * BEAT_LENGTH_V6
    context_flat = context_beats.reshape(1, flat_size)
    normalized = model_scaler.transform(context_flat).astype(np.float32)
    context_input = normalized.reshape(1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH_V6)

    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    output = model_session.run([output_name], {input_name: context_input})[0]

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

    return max(0.0, min(1.0, prob_abnormal))


def process_ecg(signal, fs):
    """Full pipeline: filter -> detect R-peaks -> segment -> classify."""
    global filtered_signal, beat_buffer

    r_peaks, filtered = detect_r_peaks(signal, fs)
    filtered_signal = filtered

    is_context = active_model_config.get('context_aware', False)
    beat_length = active_model_config['beat_length']
    input_shape = active_model_config['input_shape']

    segmenter = RRAdaptiveSegmenter(
        fs=fs,
        pre_frac=PT_CONFIG["pre_frac"],
        post_frac=PT_CONFIG["post_frac"],
        pre_min_sec=PT_CONFIG["pre_min_sec"],
        pre_max_sec=PT_CONFIG["pre_max_sec"],
        post_min_sec=PT_CONFIG["post_min_sec"],
        post_max_sec=PT_CONFIG["post_max_sec"],
        rr_history_len=PT_CONFIG["rr_history_len"]
    )

    results = []
    beat_buffer = []
    last_r = None

    for i, r_idx in enumerate(r_peaks):
        r_idx = int(r_idx)

        if last_r is not None:
            segmenter.update_rr(r_idx - last_r)
        last_r = r_idx

        if is_context:
            beat = extract_beat_v6(filtered, r_idx)
            beat_buffer.append(beat)
            if len(beat_buffer) > CONTEXT_WINDOW_SIZE:
                beat_buffer = beat_buffer[-CONTEXT_WINDOW_SIZE:]

            if len(beat_buffer) < CONTEXT_WINDOW_SIZE:
                results.append({
                    'beat_index': i,
                    'r_peak': r_idx,
                    'time_sec': round(r_idx / fs, 4),
                    'predicted': 'WAITING',
                    'probability': 0.0,
                    'beat_waveform': beat.tolist(),
                    'r_peak_pos_in_beat': PRE_SAMPLES_V6,
                    'beat_length': BEAT_LENGTH_V6,
                    'context_aware': True,
                })
                continue

            center_idx_in_buf = CONTEXT_WINDOW_SIZE // 2
            center_beat = beat_buffer[center_idx_in_buf]
            center_r_peak = r_peaks[i - (CONTEXT_WINDOW_SIZE - 1 - center_idx_in_buf)]

            prob = classify_beat_context([b for b in beat_buffer])
            pred_label = "ABNORMAL" if prob >= 0.5 else "NORMAL"

            results.append({
                'beat_index': i - (CONTEXT_WINDOW_SIZE - 1 - center_idx_in_buf),
                'r_peak': int(center_r_peak),
                'time_sec': round(int(center_r_peak) / fs, 4),
                'predicted': pred_label,
                'probability': round(prob, 4),
                'beat_waveform': center_beat.tolist(),
                'r_peak_pos_in_beat': PRE_SAMPLES_V6,
                'beat_length': BEAT_LENGTH_V6,
                'context_aware': True,
            })
        else:
            start, end = segmenter.compute_window(r_idx, total_len=len(filtered))
            window = filtered[start:end]
            beat_188 = resample_linear(window, beat_length)

            prob = classify_beat_single(beat_188, input_shape)
            pred_label = "ABNORMAL" if prob >= 0.5 else "NORMAL"

            results.append({
                'beat_index': i,
                'r_peak': r_idx,
                'time_sec': round(r_idx / fs, 4),
                'predicted': pred_label,
                'probability': round(prob, 4),
                'beat_waveform': beat_188.tolist(),
                'window_start': start,
                'window_end': end,
                'r_peak_pos_in_beat': r_idx - start,
                'beat_length': beat_length,
            })

    return {
        'r_peaks': r_peaks.tolist(),
        'beats': results,
        'total_beats': len(results),
        'normal_count': sum(1 for b in results if b['predicted'] == 'NORMAL'),
        'abnormal_count': sum(1 for b in results if b['predicted'] == 'ABNORMAL'),
        'waiting_count': sum(1 for b in results if b['predicted'] == 'WAITING'),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_ecg():
    global uploaded_signal, uploaded_fs, uploaded_filename, processing_results, filtered_signal

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ecg_column = request.form.get('ecg_column', '').strip() or None
    try:
        fs = int(request.form.get('sampling_rate', DEFAULT_SAMPLING_RATE))
    except (ValueError, TypeError):
        fs = DEFAULT_SAMPLING_RATE

    try:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.strip("'")

        if ecg_column and ecg_column in df.columns:
            signal = df[ecg_column].values
        else:
            # auto-detect first numeric column
            signal = None
            detected_col = None
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    signal = df[col].values
                    detected_col = col
                    break
            if signal is None:
                return jsonify({'error': 'No numeric column found in CSV'}), 400
            ecg_column = detected_col

        uploaded_signal = signal.astype(np.float32)
        uploaded_fs = fs
        uploaded_filename = f.filename
        processing_results = None
        filtered_signal = None

        duration_sec = len(uploaded_signal) / fs

        return jsonify({
            'status': 'ok',
            'filename': f.filename,
            'samples': len(uploaded_signal),
            'sampling_rate': fs,
            'ecg_column': ecg_column,
            'duration_sec': round(duration_sec, 2),
            'columns': list(df.columns),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/process', methods=['POST'])
def process():
    global processing_results

    if uploaded_signal is None:
        return jsonify({'error': 'No ECG signal uploaded yet'}), 400

    data = request.get_json(silent=True) or {}
    model_version = data.get('model_version', None)

    if model_version and model_version != active_model_version:
        try:
            load_model(model_version)
        except Exception as e:
            return jsonify({'error': 'Failed to load model: {}'.format(str(e))}), 500

    try:
        processing_results = process_ecg(uploaded_signal, uploaded_fs)
        return jsonify(processing_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream')
def stream():
    if uploaded_signal is None:
        return jsonify({'error': 'No signal loaded'}), 400

    try:
        start = int(request.args.get('start', 0))
        end = int(request.args.get('end', len(uploaded_signal)))
    except (ValueError, TypeError):
        start = 0
        end = len(uploaded_signal)

    start = max(0, start)
    end = min(len(uploaded_signal), end)

    sig = uploaded_signal[start:end]

    # downsample if too many points for display
    max_points = 10000
    if len(sig) > max_points:
        factor = int(math.ceil(len(sig) / max_points))
        sig = sig[::factor]

    fil = None
    if filtered_signal is not None:
        fil_seg = filtered_signal[start:end]
        if len(fil_seg) > max_points:
            factor = int(math.ceil(len(fil_seg) / max_points))
            fil_seg = fil_seg[::factor]
        fil = fil_seg.tolist()

    return jsonify({
        'signal': sig.tolist(),
        'filtered': fil,
        'start': start,
        'end': end,
        'total_samples': len(uploaded_signal),
        'sampling_rate': uploaded_fs,
    })


@app.route('/api/status')
def status():
    info = {
        'has_signal': uploaded_signal is not None,
        'filename': uploaded_filename,
        'total_samples': len(uploaded_signal) if uploaded_signal is not None else 0,
        'sampling_rate': uploaded_fs,
        'processed': processing_results is not None,
    }
    if processing_results:
        info['total_beats'] = processing_results['total_beats']
        info['normal_count'] = processing_results['normal_count']
        info['abnormal_count'] = processing_results['abnormal_count']
    return jsonify(info)


@app.route('/api/model_info')
def model_info():
    if active_model_config is None:
        return jsonify({'error': 'No model loaded'}), 400
    return jsonify({
        'name': active_model_config['name'],
        'onnx_file': active_model_config['onnx_file'],
        'scaler_file': active_model_config['scaler_file'],
        'beat_length': active_model_config['beat_length'],
        'context_aware': active_model_config.get('context_aware', False),
    })


@app.route('/api/change_model', methods=['POST'])
def change_model():
    global processing_results
    data = request.get_json(silent=True) or {}
    version = data.get('model_version', 'v3')
    try:
        load_model(version)
        processing_results = None
        return jsonify({'status': 'ok', 'model': active_model_config['name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ECG Real-Input Classification Webapp')
    parser.add_argument('--model', '-m', type=str, default='v3',
                        choices=['v2', 'v3', 'v5', 'v6'],
                        help='Model version: v2 (CNN), v3 (LSTM), v5 (Transformer), v6 (Context-Aware). Default: v3')
    parser.add_argument('--port', '-p', type=int, default=5001,
                        help='Port to run the server on. Default: 5001')
    args = parser.parse_args()

    print("=" * 60)
    print("ECG Real-Input Classification Webapp")
    print("Built-in Pan-Tompkins R-peak Detection")
    print("Using PyTorch ONNX Models")
    print("=" * 60)

    print("\nSelected model: {}".format(args.model.upper()))
    print("Loading model...")
    load_model(model_version=args.model)

    print("\nStarting web server on port {}...".format(args.port))
    print("Open your browser and go to: http://localhost:{}".format(args.port))
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
