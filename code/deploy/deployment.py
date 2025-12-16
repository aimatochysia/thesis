"""
End-to-end ECG streaming-to-beat pipeline with RR-adaptive segmentation and per-beat classification.

What this script does:
- Loads continuous ECG recordings (CSV or array) and emulates real-time streaming.
- Applies low-compute preprocessing (bandpass + optional notch), Pan–Tompkins-like R-peak detection.
- Maintains RR history and uses RR-adaptive windows with sensible clamps to segment single beats.
- Resamples each segmented beat to match the model's expected input length (188).
- Normalizes beats to match the training distribution.
- Loads a classification model (PyTorch ONNX, Keras .h5, or scikit-learn .pkl) and predicts per-beat normal/abnormal.
- Produces plots:
  - Continuous signal with detected R-peaks and segmented beat windows.
  - A grid of beats showing the model's prediction per beat.
- Outputs a simple CSV with per-beat timestamps, indices, predictions, and scores.

Dependencies:
- numpy, scipy, pandas, matplotlib
- onnxruntime (for PyTorch ONNX models - recommended)
- tensorflow (for .h5 Keras models), or scikit-learn + joblib (for .pkl models)

Usage with PyTorch ONNX models:
    python deployment.py --input_csv data/ecg.csv --onnx_model sample/ecg_lstm_v3_pytorch_final.onnx \\
                         --scaler sample/scaler_v3_pytorch.pkl --model_version v3

Notes and assumptions:
- Sampling rate fs should be known or provided. If unknown, the script assumes fs=360 Hz (common in datasets).
- This script is intended for offline replay with real-time emulation (sleep disabled by default, configurable).
- For very long datasets, the plotting functions downsample to keep figures responsive.
"""

import os
import sys
import math
import time
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt, iirnotch

# Optional TF/Sklearn imports are protected to allow running either path
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import joblib
except Exception:
    joblib = None


# -------------------------------
# Configuration (adjust as needed)
# -------------------------------

CONFIG = {
    # Data/source configuration
    "input_csv_path": None,  # e.g., "data/long_ecg_record.csv" containing one column "ecg" or multiple columns; set via CLI or code
    "ecg_column": None,      # If None, first numeric column will be used
    "fs": 360,               # Sampling rate (Hz). Adjust to your dataset if known.

    # Preprocessing pipeline
    "bandpass_low_hz": 0.5,  # Bandpass lower cutoff
    "bandpass_high_hz": 40.0,  # Bandpass upper cutoff
    "use_notch": False,
    "notch_freq_hz": 50.0,   # 50 or 60 Hz depending on region
    "notch_q": 30.0,

    # Pan–Tompkins-like detector parameters
    "derivative_kernel": np.array([-1, -2, 0, 2, 1], dtype=np.float32),  # simple FIR derivative
    "mwi_window_sec": 0.12,       # moving window integration size
    "refractory_ms": 200,         # minimum time between beats
    "adaptive_threshold_alpha": 0.01,  # smoothing factor for threshold updates
    "initial_threshold": 0.0,     # will be set adaptive on start

    # RR-adaptive window fractions and clamps
    "pre_frac": 0.35,             # fraction of RR for pre-R window
    "post_frac": 0.65,            # fraction of RR for post-R window
    "pre_min_sec": 0.08,
    "pre_max_sec": 0.35,
    "post_min_sec": 0.16,
    "post_max_sec": 0.60,
    "rr_history_len": 8,          # beats to keep for RR median

    # Model input normalization and resampling
    "model_input_len": 188,       # expected input length for v2 (and others)
    "target_baseline": 950.0,     # center beats near this baseline
    "NORMALIZE_MODE": "per_beat_standardize",  # "baseline_shift_scale" or "per_beat_standardize" or "none"
    "global_scale": 100.0,        # scale divisor after baseline shift; adjust to match training
    "per_beat_eps": 1e-6,

    # Fallback behavior
    "search_back_sec": 1.2,       # if no R for this duration, run search-back
    "critical_pause_sec": 2.0,    # raise alert flag if no R for this duration
    "enable_sliding_window_fallback": True,
    "fallback_stride_samples": 32,

    # Classification model paths (ONNX is now preferred)
    "onnx_path": None,            # e.g., "sample/ecg_lstm_v3_pytorch_final.onnx"
    "scaler_path": None,          # e.g., "sample/scaler_v3_pytorch.pkl"
    "model_version": "v3",        # v2, v3, or v5
    "keras_h5_path": None,        # e.g., "models/v2_cnn.h5" (legacy)
    "sklearn_pkl_path": None,     # e.g., "models/ensemble.pkl" (legacy)

    # Output/reporting
    "output_csv_path": "outputs/per_beat_predictions.csv",
    "plots_dir": "outputs/plots",
    "plot_max_points": 200000,    # downsample for plotting
    "save_plots": True,

    # Real-time emulation
    "enable_sleep": False,        # set True to sleep per sample for fs timing emulation
}


# -------------------------------
# Utility functions
# -------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def downsample_for_plot(x: np.ndarray, max_points: int) -> np.ndarray:
    if x.size <= max_points:
        return x
    factor = int(math.ceil(x.size / max_points))
    return x[::factor]


def butter_bandpass_sos(lowcut, highcut, fs, order=2):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos


def notch_sos(freq, q, fs):
    b, a = iirnotch(w0=freq/(fs/2), Q=q)  # iirnotch uses normalized frequency
    # Convert to sos via tf? We'll use direct filter with b,a for notch to keep simple.
    return b, a


# -------------------------------
# Detector and preprocessing
# -------------------------------

class PanTompkinsDetector:
    """
    Simplified Pan–Tompkins-like detector suitable for streaming.
    - Maintains derivative, squaring, moving window integration (MWI).
    - Adaptive thresholding with refractory period.
    """
    def __init__(self, fs: int, derivative_kernel: np.ndarray, mwi_window_sec: float,
                 refractory_ms: int, adaptive_alpha: float, init_threshold: float = 0.0):
        self.fs = fs
        self.deriv_kernel = derivative_kernel.astype(np.float32)
        self.mwi_window = int(max(1, round(mwi_window_sec * fs)))
        self.refractory_samples = int(round(refractory_ms * fs / 1000.0))
        self.alpha = adaptive_alpha

        # Buffers
        self.raw_buffer = []
        self.filtered_buffer = []
        self.deriv_buffer = []
        self.square_buffer = []
        self.mwi_buffer = []

        # Adaptive threshold state
        self.threshold = init_threshold
        self.peak_mean = 0.0
        self.noise_mean = 0.0
        self.last_peak_index = -10**9

    def step(self, filtered_sample: float) -> Optional[int]:
        """
        Process one sample; return R-peak index if detected, else None.
        """
        # Update buffers
        self.filtered_buffer.append(filtered_sample)
        idx = len(self.filtered_buffer) - 1

        # Derivative via small FIR over filtered_buffer
        deriv = self._compute_derivative(idx)
        self.deriv_buffer.append(deriv)

        sq = deriv * deriv
        self.square_buffer.append(sq)

        mwi_val = self._moving_window_integral(idx)
        self.mwi_buffer.append(mwi_val)

        # Initialize threshold after some samples
        if len(self.mwi_buffer) < self.mwi_window * 4:
            # Initialize with a small multiple of mean
            mean_mwi = np.mean(self.mwi_buffer)
            self.threshold = mean_mwi * 1.5
            return None

        # Adaptive thresholding
        detected = False
        if mwi_val > self.threshold:
            # Check refractory
            if idx - self.last_peak_index > self.refractory_samples:
                detected = True
                # Update peak_mean
                self.peak_mean = (1 - self.alpha) * self.peak_mean + self.alpha * mwi_val
                # Set last_peak_index tentatively (we'll refine with local search)
                self.last_peak_index = idx
        else:
            # Update noise_mean
            self.noise_mean = (1 - self.alpha) * self.noise_mean + self.alpha * mwi_val

        # Update threshold
        # Typical Pan-Tompkins uses threshold halfway between noise and signal estimates
        self.threshold = self.noise_mean + 0.5 * (self.peak_mean - self.noise_mean)

        if detected:
            # Local peak refinement on filtered signal in a small neighborhood
            r_index = self._local_peak_refinement(idx)
            self.last_peak_index = r_index
            return r_index

        return None

    def _compute_derivative(self, idx: int) -> float:
        k = len(self.deriv_kernel)
        if idx < k - 1:
            return 0.0
        # Convolution of derivative kernel centered at idx
        segment = self.filtered_buffer[idx - (k - 1): idx + 1]
        return float(np.dot(segment, self.deriv_kernel))

    def _moving_window_integral(self, idx: int) -> float:
        w = self.mwi_window
        start = max(0, idx - w + 1)
        segment = self.square_buffer[start: idx + 1]
        # Mean of squared derivative over window
        return float(np.mean(segment)) if segment else 0.0

    def _local_peak_refinement(self, idx: int, search_radius_samples: int = 10) -> int:
        # Find local maximum in filtered signal near idx
        start = max(0, idx - search_radius_samples)
        end = min(len(self.filtered_buffer), idx + search_radius_samples + 1)
        segment = np.array(self.filtered_buffer[start:end], dtype=np.float32)
        local_max_offset = int(np.argmax(segment))
        return start + local_max_offset


def preprocess_stream(signal: np.ndarray, fs: int, bp_low: float, bp_high: float,
                      use_notch: bool, notch_freq: float, notch_q: float) -> np.ndarray:
    """
    Apply bandpass and optional notch filtering to the entire signal (offline convenience).
    For true streaming, you'd implement causal steps; this function is okay for offline replay.
    """
    sos = butter_bandpass_sos(bp_low, bp_high, fs, order=2)
    filtered = sosfilt(sos, signal)

    if use_notch:
        b, a = notch_sos(notch_freq, notch_q, fs)
        filtered = sosfilt(np.array([b, a], dtype=object), filtered)  # fallback; or use lfilter(b,a,filtered)

    return filtered


# -------------------------------
# Segmentation and normalization
# -------------------------------

class RRAdaptiveSegmenter:
    def __init__(self, fs: int, pre_frac: float, post_frac: float,
                 pre_min_sec: float, pre_max_sec: float, post_min_sec: float, post_max_sec: float,
                 rr_history_len: int):
        self.fs = fs
        self.pre_frac = pre_frac
        self.post_frac = post_frac
        self.pre_min_samples = int(round(pre_min_sec * fs))
        self.pre_max_samples = int(round(pre_max_sec * fs))
        self.post_min_samples = int(round(post_min_sec * fs))
        self.post_max_samples = int(round(post_max_sec * fs))
        self.rr_hist_len = rr_history_len
        self.rr_history: List[int] = []
        self.default_rr_samples = int(round(0.8 * fs))  # default RR ~0.8 s

    def update_rr(self, rr_samples: int):
        self.rr_history.append(rr_samples)
        if len(self.rr_history) > self.rr_hist_len:
            self.rr_history.pop(0)

    def median_rr(self) -> int:
        if not self.rr_history:
            return self.default_rr_samples
        return int(np.median(self.rr_history))

    def compute_window(self, r_index: int, total_len: int) -> Tuple[int, int]:
        rr = self.median_rr()
        pre = int(round(self.pre_frac * rr))
        post = int(round(self.post_frac * rr))

        pre = max(self.pre_min_samples, min(self.pre_max_samples, pre))
        post = max(self.post_min_samples, min(self.post_max_samples, post))

        start = max(0, r_index - pre)
        end = min(total_len, r_index + post)
        return start, end


def resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Simple linear resampling to target_len samples.
    """
    if len(x) == target_len:
        return x.astype(np.float32)
    xi = np.linspace(0, 1, num=len(x), endpoint=True)
    xo = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(xo, xi, x).astype(np.float32)


def normalize_beat(x: np.ndarray, mode: str, target_baseline: float, global_scale: float, eps: float) -> np.ndarray:
    """
    Normalize beat to match training expectations.
    - "baseline_shift_scale": subtract target_baseline, divide by global_scale
    - "per_beat_standardize": per-beat mean 0, std 1
    - "none": no change
    """
    if mode == "baseline_shift_scale":
        return (x - target_baseline) / (global_scale if global_scale != 0 else 1.0)
    elif mode == "per_beat_standardize":
        m = float(np.mean(x))
        s = float(np.std(x))
        s = s if s > eps else 1.0
        return (x - m) / s
    else:
        return x.astype(np.float32)


# -------------------------------
# Model loading and prediction
# -------------------------------

# ONNX Runtime import for PyTorch ONNX models
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

# Model configurations for PyTorch ONNX models
MODEL_CONFIGS = {
    'v2': {
        'name': 'CNN (v2)',
        'onnx_file': 'ecg_cnn_v2_pytorch_final.onnx',
        'scaler_file': 'scaler_v2_pytorch.pkl',
        'input_shape': (1, 1, 188),  # CNN: (batch, channels, length)
    },
    'v3': {
        'name': 'LSTM (v3)',
        'onnx_file': 'ecg_lstm_v3_pytorch_final.onnx',
        'scaler_file': 'scaler_v3_pytorch.pkl',
        'input_shape': (1, 188, 1),  # LSTM: (batch, timesteps, features)
    },
    'v5': {
        'name': 'Transformer (v5)',
        'onnx_file': 'ecg_transformer_v5_pytorch_final.onnx',
        'scaler_file': 'scaler_v5_pytorch.pkl',
        'input_shape': (1, 188, 1),  # Transformer: (batch, timesteps, features)
    },
}


class BeatClassifier:
    def __init__(self, keras_h5_path: Optional[str] = None, sklearn_pkl_path: Optional[str] = None,
                 onnx_path: Optional[str] = None, onnx_input_shape: Optional[Tuple] = None,
                 scaler_path: Optional[str] = None):
        self.keras_model = None
        self.sklearn_model = None
        self.onnx_session = None
        self.onnx_input_shape = onnx_input_shape
        self.scaler = None
        self.model_type = None

        # Priority: ONNX > Keras > sklearn
        if onnx_path and ONNX_AVAILABLE and os.path.isfile(onnx_path):
            self.onnx_session = ort.InferenceSession(onnx_path)
            self.model_type = "onnx"
            print(f"Loaded ONNX model from: {onnx_path}")
        elif keras_h5_path and tf is not None and os.path.isfile(keras_h5_path):
            self.keras_model = tf.keras.models.load_model(keras_h5_path)
            self.model_type = "keras"
        elif sklearn_pkl_path and joblib is not None and os.path.isfile(sklearn_pkl_path):
            self.sklearn_model = joblib.load(sklearn_pkl_path)
            self.model_type = "sklearn"
        else:
            raise RuntimeError("No valid model file found or required libraries not available.")
        
        # Load scaler if provided
        if scaler_path and joblib is not None and os.path.isfile(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from: {scaler_path}")

    def preprocess_beat(self, beat_188: np.ndarray) -> np.ndarray:
        """
        Apply scaler transformation if available.
        """
        if self.scaler is not None:
            # Use the scaler fitted on training data
            beat_2d = beat_188.reshape(1, -1)
            return self.scaler.transform(beat_2d).flatten().astype(np.float32)
        return beat_188.astype(np.float32)

    def predict_proba(self, beat_188: np.ndarray, use_scaler: bool = True) -> float:
        """
        Returns probability of abnormal class (1).
        If use_scaler=True and scaler is available, applies scaler first.
        """
        # Apply scaler if requested and available
        if use_scaler and self.scaler is not None:
            beat_188 = self.preprocess_beat(beat_188)
        
        if self.model_type == "onnx":
            # Reshape according to model-specific input shape
            if self.onnx_input_shape:
                x = beat_188.reshape(self.onnx_input_shape).astype(np.float32)
            else:
                # Default: assume LSTM/Transformer shape
                x = beat_188.reshape(1, -1, 1).astype(np.float32)
            
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            proba = self.onnx_session.run([output_name], {input_name: x})[0]
            
            # Handle 2-class output from PyTorch models
            if proba.shape[1] == 2:
                return float(proba[0, 1])
            else:
                return float(proba[0, 0])
        elif self.model_type == "keras":
            # Expect shape (1, 188, 1) for Conv1D-based models
            x = beat_188.reshape(1, -1, 1).astype(np.float32)
            prob = float(self.keras_model.predict(x, verbose=0).squeeze())
            return prob
        elif self.model_type == "sklearn":
            # Expect shape (1, 188) for traditional ML
            x = beat_188.reshape(1, -1).astype(np.float32)
            if hasattr(self.sklearn_model, "predict_proba"):
                prob = float(self.sklearn_model.predict_proba(x)[0, 1])
            else:
                # Fallback: decision_function or predict
                df = getattr(self.sklearn_model, "decision_function", None)
                if df:
                    score = float(df(x))
                    # Map decision score to [0,1] via logistic approximation
                    prob = 1.0 / (1.0 + math.exp(-score))
                else:
                    pred = int(self.sklearn_model.predict(x)[0])
                    prob = float(pred)
            return prob
        else:
            raise RuntimeError("Unknown model type.")


# -------------------------------
# Main pipeline
# -------------------------------

def load_signal_from_csv(path: str, ecg_column: Optional[str]) -> np.ndarray:
    df = pd.read_csv(path)
    if ecg_column and ecg_column in df.columns:
        signal = df[ecg_column].values
    else:
        # Pick first numeric column
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                signal = df[col].values
                break
        else:
            raise ValueError("No numeric column found in CSV.")
    return signal.astype(np.float32)


def emulate_stream_and_classify(
    signal: np.ndarray,
    fs: int,
    config: Dict,
    classifier: BeatClassifier,
    plot: bool = True
) -> Dict:
    """
    Emulate streaming, detect R, segment beats, classify, and collect outputs.
    Returns a dict with results including beats, timestamps, predictions, etc.
    """
    # Preprocess entire signal (offline convenience)
    filtered = preprocess_stream(signal, fs, config["bandpass_low_hz"], config["bandpass_high_hz"],
                                 config["use_notch"], config["notch_freq_hz"], config["notch_q"])

    detector = PanTompkinsDetector(
        fs=fs,
        derivative_kernel=config["derivative_kernel"],
        mwi_window_sec=config["mwi_window_sec"],
        refractory_ms=config["refractory_ms"],
        adaptive_alpha=config["adaptive_threshold_alpha"],
        init_threshold=config["initial_threshold"]
    )

    segmenter = RRAdaptiveSegmenter(
        fs=fs,
        pre_frac=config["pre_frac"],
        post_frac=config["post_frac"],
        pre_min_sec=config["pre_min_sec"],
        pre_max_sec=config["pre_max_sec"],
        post_min_sec=config["post_min_sec"],
        post_max_sec=config["post_max_sec"],
        rr_history_len=config["rr_history_len"]
    )

    results = {
        "r_indices": [],
        "beat_windows": [],        # (start, end)
        "beats_raw": [],           # segmented raw (preprocessed filtered)
        "beats_resampled": [],     # resampled to 188
        "beats_normalized": [],
        "pred_probs": [],
        "pred_labels": [],
        "timestamps_sec": [],
    }

    last_r_index = None
    last_r_time_sec = -1e9
    search_back_samples = int(round(config["search_back_sec"] * fs))
    critical_pause_samples = int(round(config["critical_pause_sec"] * fs))

    # Streaming emulation loop
    for i in range(len(filtered)):
        sample = filtered[i]

        r_index = detector.step(sample)
        if config["enable_sleep"]:
            time.sleep(1.0 / fs)

        # Periodic checks: search-back if long gap
        if last_r_index is not None:
            gap = i - last_r_index
            if gap > search_back_samples:
                # Search-back: naive approach — find local maxima in recent buffer
                # Here, since we process offline already, detector handles adaptive thresholding.
                # We can try a lower threshold recheck, but to keep things simple we rely on detector state.
                pass

            if gap > critical_pause_samples:
                # Critical pause — log an alert event (no classification)
                # In a real system, you'd raise immediate alert. Here we just log.
                # print(f"[ALERT] Critical pause detected at sample {i}")
                pass

        if r_index is not None:
            # Compute RR if previous beat exists
            if last_r_index is not None:
                rr = r_index - last_r_index
                segmenter.update_rr(rr)
            last_r_index = r_index
            last_r_time_sec = r_index / fs

            # RR-adaptive window
            start, end = segmenter.compute_window(r_index, total_len=len(filtered))
            window = filtered[start:end]

            # Resample to 188
            beat_188 = resample_linear(window, config["model_input_len"])

            # Normalize
            beat_norm = normalize_beat(
                beat_188,
                mode=config["NORMALIZE_MODE"],
                target_baseline=config["target_baseline"],
                global_scale=config["global_scale"],
                eps=config["per_beat_eps"]
            )

            # Predict - if classifier has scaler, pass raw beat; otherwise use normalized
            if hasattr(classifier, 'scaler') and classifier.scaler is not None:
                # Use scaler in classifier (PyTorch ONNX models)
                prob_abnormal = classifier.predict_proba(beat_188, use_scaler=True)
            else:
                # Use manual normalization
                prob_abnormal = classifier.predict_proba(beat_norm, use_scaler=False)
            label = int(prob_abnormal >= 0.5)

            # Store results
            results["r_indices"].append(r_index)
            results["timestamps_sec"].append(last_r_time_sec)
            results["beat_windows"].append((start, end))
            results["beats_raw"].append(window.astype(np.float32))
            results["beats_resampled"].append(beat_188.astype(np.float32))
            results["beats_normalized"].append(beat_norm.astype(np.float32))
            results["pred_probs"].append(prob_abnormal)
            results["pred_labels"].append(label)

    # Convert lists to arrays where convenient
    results["r_indices"] = np.array(results["r_indices"], dtype=np.int32)
    results["timestamps_sec"] = np.array(results["timestamps_sec"], dtype=np.float32)
    results["pred_probs"] = np.array(results["pred_probs"], dtype=np.float32)
    results["pred_labels"] = np.array(results["pred_labels"], dtype=np.int32)

    if plot and config["save_plots"]:
        ensure_dir(config["plots_dir"])
        plot_continuous_with_beats(signal, filtered, fs, results, config)
        plot_beats_grid(results, config)

    return results


# -------------------------------
# Plotting
# -------------------------------

def plot_continuous_with_beats(signal: np.ndarray, filtered: np.ndarray, fs: int, results: Dict, config: Dict):
    """
    Plot the continuous raw and filtered signals with detected R-peaks and segmentation windows.
    """
    raw_ds = downsample_for_plot(signal, config["plot_max_points"])
    fil_ds = downsample_for_plot(filtered, config["plot_max_points"])
    t_raw = np.arange(len(raw_ds)) / fs
    t_fil = np.arange(len(fil_ds)) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(t_raw, raw_ds, label="Raw ECG", alpha=0.5)
    plt.plot(t_fil, fil_ds, label="Filtered ECG", alpha=0.8)

    # R-peaks (downsample-safe marking by converting sample indices to time)
    for r_idx in results["r_indices"]:
        t = r_idx / fs
        plt.axvline(t, color="red", alpha=0.3, linestyle="--")

    # Segmentation windows
    for (start, end) in results["beat_windows"]:
        plt.axvspan(start / fs, end / fs, color="green", alpha=0.08)

    plt.title("Continuous ECG with detected R-peaks and beat segmentation windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    out_path = os.path.join(config["plots_dir"], "continuous_with_beats.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_beats_grid(results: Dict, config: Dict, max_beats: int = 50):
    """
    Plot a grid of segmented beats (resampled to 188) with predicted probabilities/labels.
    """
    n = len(results["beats_resampled"])
    if n == 0:
        return

    sel = np.arange(n)
    if n > max_beats:
        sel = np.linspace(0, n - 1, num=max_beats, dtype=int)

    cols = 5
    rows = int(math.ceil(len(sel) / cols))
    plt.figure(figsize=(cols * 3, rows * 2.5))

    for i, idx in enumerate(sel):
        beat = results["beats_resampled"][idx]
        prob = results["pred_probs"][idx]
        lab = results["pred_labels"][idx]
        t = np.arange(len(beat)) / config["fs"]

        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(beat, color="black", linewidth=1.0)
        ax.set_title(f"Beat {idx} | P(abn)={prob:.2f} | Label={lab}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Sampled segmented beats with model predictions")
    out_path = os.path.join(config["plots_dir"], "beats_grid.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------------
# Output CSV
# -------------------------------

def save_results_csv(results: Dict, config: Dict):
    rows = []
    for i in range(len(results["r_indices"])):
        start, end = results["beat_windows"][i]
        rows.append({
            "beat_index": i,
            "r_sample_index": int(results["r_indices"][i]),
            "timestamp_sec": float(results["timestamps_sec"][i]),
            "window_start": int(start),
            "window_end": int(end),
            "pred_prob_abnormal": float(results["pred_probs"][i]),
            "pred_label": int(results["pred_labels"][i]),
        })
    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(config["output_csv_path"]))
    df.to_csv(config["output_csv_path"], index=False)


# -------------------------------
# CLI / Entry point
# -------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ECG streaming-to-beat deployment pipeline with PyTorch ONNX models")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to continuous ECG CSV file")
    parser.add_argument("--ecg_column", type=str, default=None, help="Column name containing ECG values (default: first numeric col)")
    parser.add_argument("--fs", type=int, default=CONFIG["fs"], help="Sampling rate (Hz)")
    
    # Model options (ONNX preferred)
    parser.add_argument("--onnx_model", type=str, default=None, help="Path to PyTorch ONNX model (e.g., sample/ecg_lstm_v3_pytorch_final.onnx)")
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler pickle (e.g., sample/scaler_v3_pytorch.pkl)")
    parser.add_argument("--model_version", type=str, default="v3", choices=["v2", "v3", "v5"],
                        help="Model version for input shape: v2 (CNN), v3 (LSTM), v5 (Transformer)")
    
    # Legacy model options
    parser.add_argument("--keras_h5", type=str, default=None, help="Path to Keras .h5 model (legacy)")
    parser.add_argument("--sklearn_pkl", type=str, default=None, help="Path to scikit-learn .pkl model (legacy)")
    
    # Output options
    parser.add_argument("--output_csv", type=str, default=CONFIG["output_csv_path"], help="Output CSV path")
    parser.add_argument("--plots_dir", type=str, default=CONFIG["plots_dir"], help="Directory to save plots")
    parser.add_argument("--normalize_mode", type=str, default=CONFIG["NORMALIZE_MODE"],
                        choices=["baseline_shift_scale", "per_beat_standardize", "none"], help="Beat normalization mode")
    parser.add_argument("--target_baseline", type=float, default=CONFIG["target_baseline"], help="Target baseline value")
    parser.add_argument("--global_scale", type=float, default=CONFIG["global_scale"], help="Global scaling divisor")
    parser.add_argument("--enable_sleep", action="store_true", help="Emulate real-time by sleeping per sample")
    args = parser.parse_args()

    # Update config from CLI
    CONFIG["input_csv_path"] = args.input_csv
    CONFIG["ecg_column"] = args.ecg_column
    CONFIG["fs"] = args.fs
    CONFIG["onnx_path"] = args.onnx_model
    CONFIG["scaler_path"] = args.scaler
    CONFIG["model_version"] = args.model_version
    CONFIG["keras_h5_path"] = args.keras_h5
    CONFIG["sklearn_pkl_path"] = args.sklearn_pkl
    CONFIG["output_csv_path"] = args.output_csv
    CONFIG["plots_dir"] = args.plots_dir
    CONFIG["NORMALIZE_MODE"] = args.normalize_mode
    CONFIG["target_baseline"] = args.target_baseline
    CONFIG["global_scale"] = args.global_scale
    CONFIG["enable_sleep"] = args.enable_sleep

    # Load signal
    signal = load_signal_from_csv(CONFIG["input_csv_path"], CONFIG["ecg_column"])

    # Get model input shape if using ONNX
    onnx_input_shape = None
    if args.onnx_model or args.model_version:
        model_version = args.model_version or 'v3'
        if model_version in MODEL_CONFIGS:
            onnx_input_shape = MODEL_CONFIGS[model_version]['input_shape']

    # Load classifier (ONNX preferred, fallback to Keras/sklearn)
    classifier = BeatClassifier(
        keras_h5_path=CONFIG["keras_h5_path"],
        sklearn_pkl_path=CONFIG["sklearn_pkl_path"],
        onnx_path=CONFIG["onnx_path"],
        onnx_input_shape=onnx_input_shape,
        scaler_path=CONFIG["scaler_path"]
    )

    # Run pipeline
    results = emulate_stream_and_classify(signal, CONFIG["fs"], CONFIG, classifier, plot=True)

    # Save outputs
    save_results_csv(results, CONFIG)

    print(f"Completed. Saved per-beat predictions to {CONFIG['output_csv_path']} and plots to {CONFIG['plots_dir']}.")


if __name__ == "__main__":
    main()
