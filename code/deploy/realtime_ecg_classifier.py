"""
Real-time ECG Classifier

This script performs real-time ECG classification following these steps:
1. Simulates real-time ECG streaming from processed data
2. Applies preprocessing in streaming fashion
3. Detects R-peaks and segments heartbeats
4. Classifies each heartbeat using an AI model (normal/abnormal)
5. Outputs real-time predictions

The script can work with:
- Pre-processed CSV data (from ecg_data_processor.py)
- Raw ECG signals (synthetic or loaded from file)
- A trained classification model (.h5 Keras or .pkl scikit-learn)

If no model is available, a simple threshold-based classifier is used for demonstration.

Usage:
    python realtime_ecg_classifier.py --input processed_data.csv --mode csv
    python realtime_ecg_classifier.py --duration 60 --mode synthetic
"""

import os
import sys
import time
import math
from typing import Optional, Tuple, List, Dict
from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


# Configuration
CONFIG = {
    "fs": 360,                    # Sampling rate (Hz)
    "bandpass_low_hz": 0.4,       # Bandpass lower cutoff
    "bandpass_high_hz": 40.0,     # Bandpass upper cutoff
    "filter_order": 5,            # Butterworth filter order
    
    # R-peak detection
    "smoothing_window_sec": 0.1,  # Smoothing window (100 ms)
    "refractory_ms": 200,         # Minimum time between beats
    "threshold_k": 2.5,           # Threshold multiplier
    
    # Beat segmentation
    "pre_r_sec": 0.25,            # Seconds before R-peak
    "post_r_sec": 0.45,           # Seconds after R-peak
    "model_input_len": 188,       # Expected input length for model
    
    # Normalization (matching training data baseline ~950)
    "target_baseline": 950.0,
    "global_scale": 100.0,
    
    # Real-time simulation
    "stream_delay_sec": 0.0,      # Delay between samples (0 = no delay)
    "batch_size": 100,            # Process samples in batches
    
    # Model paths (optional)
    "keras_h5_path": None,
    "sklearn_pkl_path": None,
}

# ECG waveform amplitude constants for synthetic signal generation
ECG_WAVEFORM = {
    "baseline": 950,           # Baseline amplitude
    "p_wave_amp": 15,          # P-wave amplitude
    "q_wave_amp": 20,          # Q-wave amplitude (negative deflection)
    "r_peak_normal_amp": 250,  # Normal R-peak amplitude
    "r_peak_abnormal_min": 150,  # Abnormal R-peak minimum
    "r_peak_abnormal_max": 350,  # Abnormal R-peak maximum
    "s_wave_amp": 40,          # S-wave amplitude (negative deflection)
    "t_wave_normal_amp": 30,   # Normal T-wave amplitude
    "t_wave_abnormal_min": 10, # Abnormal T-wave minimum
    "t_wave_abnormal_max": 50, # Abnormal T-wave maximum
    "noise_std": 2,            # Standard deviation of noise
}


# ========== Signal Processing Functions ==========

def butter_bandpass_sos(lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """Create Butterworth bandpass filter coefficients."""
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos


def smooth_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving window smoothing."""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def resample_beat(beat: np.ndarray, target_len: int) -> np.ndarray:
    """Resample beat to target length using linear interpolation."""
    if len(beat) == target_len:
        return beat.astype(np.float32)
    xi = np.linspace(0, 1, num=len(beat), endpoint=True)
    xo = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(xo, xi, beat).astype(np.float32)


def normalize_beat(beat: np.ndarray, baseline: float = 950.0, scale: float = 100.0) -> np.ndarray:
    """Normalize beat for model input."""
    return (beat - baseline) / scale


# ========== Real-time R-peak Detector ==========

class StreamingRPeakDetector:
    """
    Streaming R-peak detector using differentiation, squaring, and adaptive thresholding.
    Uses a simplified Pan-Tompkins approach suitable for streaming.
    """
    
    def __init__(self, fs: int, smoothing_window_sec: float = 0.1,
                 refractory_ms: int = 200, threshold_k: float = 2.5):
        self.fs = fs
        self.smoothing_window = int(smoothing_window_sec * fs)
        self.refractory_samples = int(refractory_ms * fs / 1000)
        self.threshold_k = threshold_k
        
        # Buffers - larger for better statistics
        buffer_size = max(10000, fs * 10)  # At least 10 seconds
        self.raw_buffer = deque(maxlen=buffer_size)
        self.filtered_buffer = deque(maxlen=buffer_size)
        self.smoothed_buffer = deque(maxlen=buffer_size)
        
        # State
        self.last_peak_idx = -10000
        self.sample_count = 0
        self.threshold = 0.0
        self.peak_mean = 0.0
        self.noise_mean = 0.0
        self.initialized = False
        
        # Filter coefficients
        self.sos = butter_bandpass_sos(CONFIG["bandpass_low_hz"], 
                                        CONFIG["bandpass_high_hz"], fs, 
                                        CONFIG["filter_order"])
        
    def process_sample(self, sample: float) -> Optional[int]:
        """
        Process one sample and return R-peak index if detected.
        """
        self.raw_buffer.append(sample)
        self.filtered_buffer.append(sample)
        
        # Need minimum samples for processing
        if len(self.filtered_buffer) < self.smoothing_window + 2:
            self.sample_count += 1
            return None
        
        # Differentiate using recent samples
        recent = list(self.filtered_buffer)
        n = len(recent)
        
        # Compute derivative, square, and smooth in one pass
        deriv = recent[n-1] - recent[n-2]
        squared = deriv ** 2
        
        # Moving window integration
        start_idx = max(0, n - self.smoothing_window)
        window_samples = recent[start_idx:]
        
        # Compute derivative and square for window
        derivs = np.diff(window_samples) ** 2
        smoothed = np.mean(derivs) if len(derivs) > 0 else 0
        
        self.smoothed_buffer.append(smoothed)
        
        # Initialize threshold after collecting enough data
        if not self.initialized and len(self.smoothed_buffer) >= self.fs:  # After 1 second
            all_smoothed = list(self.smoothed_buffer)
            self.threshold = np.median(all_smoothed) + self.threshold_k * np.std(all_smoothed)
            self.peak_mean = np.percentile(all_smoothed, 90)
            self.noise_mean = np.percentile(all_smoothed, 50)
            self.initialized = True
        
        if not self.initialized:
            self.sample_count += 1
            return None
        
        current_idx = self.sample_count
        
        # Check for peak
        detected = None
        
        # Above threshold and past refractory period
        if smoothed > self.threshold:
            if current_idx - self.last_peak_idx > self.refractory_samples:
                # Confirm this is a local maximum
                recent_smoothed = list(self.smoothed_buffer)[-20:]
                if len(recent_smoothed) >= 3:
                    if recent_smoothed[-2] >= recent_smoothed[-1] and recent_smoothed[-2] >= recent_smoothed[-3]:
                        self.last_peak_idx = current_idx - 1
                        detected = current_idx - 1
                        
                        # Update peak mean
                        self.peak_mean = 0.9 * self.peak_mean + 0.1 * smoothed
        else:
            # Update noise mean
            self.noise_mean = 0.95 * self.noise_mean + 0.05 * smoothed
        
        # Adaptive threshold update
        self.threshold = self.noise_mean + 0.35 * (self.peak_mean - self.noise_mean)
        
        self.sample_count += 1
        return detected


# ========== Beat Segmenter ==========

class StreamingBeatSegmenter:
    """
    Segments heartbeats from a streaming signal using detected R-peaks.
    """
    
    def __init__(self, fs: int, pre_sec: float = 0.25, post_sec: float = 0.45,
                 target_len: int = 188):
        self.fs = fs
        self.pre_samples = int(pre_sec * fs)
        self.post_samples = int(post_sec * fs)
        self.target_len = target_len
        
        # Signal buffer - needs to hold enough for segmentation
        buffer_size = self.pre_samples + self.post_samples + 1000
        self.signal_buffer = deque(maxlen=buffer_size)
        self.sample_count = 0
        
        # Pending R-peaks waiting for enough post-R samples
        self.pending_peaks = []
        
    def add_sample(self, sample: float, r_peak_detected: bool = False) -> Optional[Dict]:
        """
        Add a sample to the buffer and optionally mark an R-peak.
        Returns a segmented beat if one is ready.
        """
        self.signal_buffer.append(sample)
        
        if r_peak_detected:
            self.pending_peaks.append(self.sample_count)
        
        self.sample_count += 1
        
        # Check if any pending peaks are ready for segmentation
        if not self.pending_peaks:
            return None
        
        oldest_peak = self.pending_peaks[0]
        samples_since_peak = self.sample_count - oldest_peak
        
        # Need enough samples after peak
        if samples_since_peak < self.post_samples:
            return None
        
        # Pop this peak and segment
        self.pending_peaks.pop(0)
        
        # Calculate indices relative to buffer
        buffer_list = list(self.signal_buffer)
        buffer_end_sample = self.sample_count - 1
        buffer_start_sample = buffer_end_sample - len(buffer_list) + 1
        
        peak_buffer_idx = oldest_peak - buffer_start_sample
        start_idx = peak_buffer_idx - self.pre_samples
        end_idx = peak_buffer_idx + self.post_samples
        
        # Check bounds
        if start_idx < 0 or end_idx > len(buffer_list):
            return None
        
        # Extract beat
        beat_raw = np.array(buffer_list[start_idx:end_idx])
        
        # Resample
        beat_resampled = resample_beat(beat_raw, self.target_len)
        
        # Normalize
        beat_normalized = normalize_beat(beat_resampled)
        
        return {
            "r_peak_idx": oldest_peak,
            "timestamp_sec": oldest_peak / self.fs,
            "beat_raw": beat_raw,
            "beat_resampled": beat_resampled,
            "beat_normalized": beat_normalized
        }


# ========== Classifier ==========

class HeartbeatClassifier:
    """
    Classifies heartbeats as normal (0) or abnormal (1).
    """
    
    def __init__(self, keras_path: Optional[str] = None, sklearn_path: Optional[str] = None):
        self.model = None
        self.model_type = None
        
        # Try to load Keras model
        if keras_path and os.path.isfile(keras_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(keras_path)
                self.model_type = "keras"
                print(f"Loaded Keras model from {keras_path}")
            except Exception as e:
                print(f"Failed to load Keras model: {e}")
        
        # Try to load sklearn model
        if self.model is None and sklearn_path and os.path.isfile(sklearn_path):
            try:
                import joblib
                self.model = joblib.load(sklearn_path)
                self.model_type = "sklearn"
                print(f"Loaded sklearn model from {sklearn_path}")
            except Exception as e:
                print(f"Failed to load sklearn model: {e}")
        
        if self.model is None:
            print("No model loaded - using threshold-based classifier for demonstration")
            self.model_type = "threshold"
    
    def classify(self, beat_normalized: np.ndarray) -> Tuple[int, float]:
        """
        Classify a normalized heartbeat.
        
        Returns:
            Tuple of (label, probability)
            label: 0 = normal, 1 = abnormal
            probability: probability of being abnormal
        """
        if self.model_type == "keras":
            x = beat_normalized.reshape(1, -1, 1).astype(np.float32)
            prob = float(self.model.predict(x, verbose=0).squeeze())
            label = int(prob >= 0.5)
            return label, prob
        
        elif self.model_type == "sklearn":
            x = beat_normalized.reshape(1, -1).astype(np.float32)
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba(x)[0, 1])
            else:
                prob = float(self.model.predict(x)[0])
            label = int(prob >= 0.5)
            return label, prob
        
        else:
            # Threshold-based classifier for demonstration
            # Abnormal beats often have unusual amplitude patterns
            beat_std = np.std(beat_normalized)
            beat_range = np.max(beat_normalized) - np.min(beat_normalized)
            
            # Simple heuristic: high variability might indicate abnormality
            # This is just for demonstration - real classification needs trained model
            abnormality_score = (beat_std - 0.5) / 0.3  # Normalized score
            prob = 1 / (1 + np.exp(-abnormality_score))  # Sigmoid
            
            # Add some small variance for demonstration (deterministic based on beat properties)
            # Use beat hash for reproducibility instead of random noise
            beat_hash = hash(tuple(beat_normalized[:10].astype(int))) % 100
            variance = (beat_hash - 50) / 500  # Small deterministic variance
            prob = min(1.0, max(0.0, prob + variance))
            label = int(prob >= 0.5)
            
            return label, prob


# ========== Real-time Pipeline ==========

class RealtimeECGPipeline:
    """
    Complete real-time ECG processing and classification pipeline.
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = CONFIG
        
        self.config = config
        self.fs = config["fs"]
        
        # Initialize components
        self.detector = StreamingRPeakDetector(
            fs=self.fs,
            smoothing_window_sec=config["smoothing_window_sec"],
            refractory_ms=config["refractory_ms"],
            threshold_k=config["threshold_k"]
        )
        
        self.segmenter = StreamingBeatSegmenter(
            fs=self.fs,
            pre_sec=config["pre_r_sec"],
            post_sec=config["post_r_sec"],
            target_len=config["model_input_len"]
        )
        
        self.classifier = HeartbeatClassifier(
            keras_path=config.get("keras_h5_path"),
            sklearn_path=config.get("sklearn_pkl_path")
        )
        
        # Results
        self.results = []
        self.total_samples = 0
        self.total_beats = 0
    
    def process_sample(self, sample: float) -> Optional[Dict]:
        """
        Process a single ECG sample through the full pipeline.
        
        Returns:
            Dict with classification results if a beat was classified, else None
        """
        # Detect R-peak
        r_peak_idx = self.detector.process_sample(sample)
        r_peak_detected = r_peak_idx is not None
        
        # Segment beat
        beat_data = self.segmenter.add_sample(sample, r_peak_detected)
        
        self.total_samples += 1
        
        if beat_data is None:
            return None
        
        # Classify
        label, prob = self.classifier.classify(beat_data["beat_normalized"])
        
        result = {
            "beat_index": self.total_beats,
            "r_peak_sample": beat_data["r_peak_idx"],
            "timestamp_sec": beat_data["timestamp_sec"],
            "pred_label": label,
            "pred_prob": prob,
            "category": "Normal" if label == 0 else "Abnormal",
        }
        
        self.results.append(result)
        self.total_beats += 1
        
        return result
    
    def process_stream(self, signal: np.ndarray, realtime: bool = False) -> List[Dict]:
        """
        Process an entire signal stream.
        
        Args:
            signal: ECG signal array
            realtime: If True, simulate real-time processing with delays
        """
        delay = self.config["stream_delay_sec"] if realtime else 0
        
        print(f"Processing {len(signal)} samples ({len(signal)/self.fs:.1f} seconds)...")
        print(f"{'='*60}")
        
        for i, sample in enumerate(signal):
            result = self.process_sample(sample)
            
            if result is not None:
                self._print_result(result)
            
            if realtime and delay > 0:
                time.sleep(delay)
        
        print(f"{'='*60}")
        print(f"Processing complete!")
        print(f"Total samples: {self.total_samples}")
        print(f"Total beats detected: {self.total_beats}")
        
        normal_count = sum(1 for r in self.results if r["pred_label"] == 0)
        abnormal_count = len(self.results) - normal_count
        
        print(f"Normal beats: {normal_count} ({100*normal_count/max(1,len(self.results)):.1f}%)")
        print(f"Abnormal beats: {abnormal_count} ({100*abnormal_count/max(1,len(self.results)):.1f}%)")
        
        return self.results
    
    def _print_result(self, result: Dict):
        """Print a classification result."""
        symbol = "✓" if result["pred_label"] == 0 else "✗"
        
        print(f"Beat #{result['beat_index']:4d} | "
              f"Time: {result['timestamp_sec']:8.2f}s | "
              f"Category: {result['category']:8s} | "
              f"Confidence: {result['pred_prob']:.2f} {symbol}")
    
    def save_results(self, output_path: str):
        """Save results to CSV."""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# ========== Data Loading Functions ==========

def generate_synthetic_ecg(duration_sec: int, fs: int, heart_rate_bpm: float = 75,
                           abnormality_rate: float = 0.1) -> np.ndarray:
    """
    Generate synthetic ECG signal for testing.
    """
    num_samples = int(duration_sec * fs)
    
    beat_interval = 60.0 / heart_rate_bpm
    rr_variability = 0.05
    
    ecg = np.zeros(num_samples)
    
    current_time = 0.5
    while current_time < duration_sec - 1:
        current_rr = beat_interval * (1 + np.random.uniform(-rr_variability, rr_variability))
        r_idx = int(current_time * fs)
        
        if r_idx >= num_samples:
            break
        
        # Occasionally create abnormal beats
        is_abnormal = np.random.random() < abnormality_rate
        
        # R-peak amplitude (abnormal might have different amplitude)
        if not is_abnormal:
            r_amplitude = ECG_WAVEFORM["r_peak_normal_amp"]
        else:
            r_amplitude = np.random.choice([ECG_WAVEFORM["r_peak_abnormal_min"], 
                                            ECG_WAVEFORM["r_peak_abnormal_max"]])
        
        # P-wave
        p_start = max(0, r_idx - int(0.2 * fs))
        p_end = max(0, r_idx - int(0.12 * fs))
        for i in range(p_start, min(p_end, num_samples)):
            offset = i - p_start
            width = p_end - p_start
            if width > 0:
                ecg[i] += ECG_WAVEFORM["p_wave_amp"] * np.sin(np.pi * offset / width)
        
        # Q-wave
        q_start = max(0, r_idx - int(0.04 * fs))
        for i in range(q_start, min(r_idx, num_samples)):
            offset = i - q_start
            width = r_idx - q_start
            if width > 0:
                ecg[i] -= ECG_WAVEFORM["q_wave_amp"] * (offset / width)
        
        # R-peak
        if r_idx < num_samples:
            ecg[r_idx] += r_amplitude
        
        # S-wave
        s_end = min(num_samples, r_idx + int(0.04 * fs))
        for i in range(r_idx + 1, s_end):
            offset = i - r_idx
            width = s_end - r_idx
            if width > 0:
                ecg[i] -= ECG_WAVEFORM["s_wave_amp"] * (1 - offset / width)
        
        # T-wave
        t_start = r_idx + int(0.1 * fs)
        t_end = min(num_samples, r_idx + int(0.35 * fs))
        if not is_abnormal:
            t_amplitude = ECG_WAVEFORM["t_wave_normal_amp"]
        else:
            t_amplitude = np.random.choice([ECG_WAVEFORM["t_wave_abnormal_min"],
                                            ECG_WAVEFORM["t_wave_abnormal_max"]])
        for i in range(t_start, t_end):
            offset = i - t_start
            width = t_end - t_start
            if width > 0:
                ecg[i] += t_amplitude * np.sin(np.pi * offset / width)
        
        current_time += current_rr
    
    ecg += ECG_WAVEFORM["baseline"]
    ecg += np.random.normal(0, ECG_WAVEFORM["noise_std"], num_samples)
    
    return ecg


def load_processed_csv(csv_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load processed ECG data from CSV (output of ecg_data_processor.py).
    Returns the amplitude data as a 2D array and the metadata DataFrame.
    """
    df = pd.read_csv(csv_path)
    
    # Extract amplitude columns
    amp_cols = [c for c in df.columns if c.startswith('amp_')]
    
    if not amp_cols:
        raise ValueError("No amplitude columns found in CSV")
    
    beats_array = df[amp_cols].values
    
    return beats_array, df


# ========== Main Entry Point ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time ECG Classifier")
    parser.add_argument("--mode", type=str, choices=["synthetic", "csv", "raw"],
                        default="synthetic",
                        help="Data source mode: synthetic, csv (processed), or raw")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file path (for csv or raw mode)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds for synthetic mode")
    parser.add_argument("--fs", type=int, default=360,
                        help="Sampling frequency")
    parser.add_argument("--heart_rate", type=float, default=75,
                        help="Heart rate for synthetic mode")
    parser.add_argument("--abnormality_rate", type=float, default=0.15,
                        help="Rate of abnormal beats for synthetic mode")
    parser.add_argument("--realtime", action="store_true",
                        help="Simulate real-time processing with delays")
    parser.add_argument("--output", type=str, default="realtime_predictions.csv",
                        help="Output CSV path for results")
    parser.add_argument("--keras_model", type=str, default=None,
                        help="Path to Keras .h5 model")
    parser.add_argument("--sklearn_model", type=str, default=None,
                        help="Path to sklearn .pkl model")
    
    args = parser.parse_args()
    
    # Update config
    CONFIG["fs"] = args.fs
    CONFIG["keras_h5_path"] = args.keras_model
    CONFIG["sklearn_pkl_path"] = args.sklearn_model
    
    if args.realtime:
        CONFIG["stream_delay_sec"] = 1.0 / args.fs  # Real-time at sampling rate
    
    print("=" * 60)
    print("Real-time ECG Classifier")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Sampling frequency: {args.fs} Hz")
    
    # Load or generate data
    if args.mode == "synthetic":
        print(f"Duration: {args.duration} seconds")
        print(f"Heart rate: {args.heart_rate} BPM")
        print(f"Abnormality rate: {args.abnormality_rate * 100:.1f}%")
        
        signal = generate_synthetic_ecg(
            duration_sec=args.duration,
            fs=args.fs,
            heart_rate_bpm=args.heart_rate,
            abnormality_rate=args.abnormality_rate
        )
        print(f"Generated {len(signal)} samples")
        
        # Create pipeline and process
        pipeline = RealtimeECGPipeline(CONFIG)
        pipeline.process_stream(signal, realtime=args.realtime)
        pipeline.save_results(args.output)
        
    elif args.mode == "csv":
        if not args.input:
            print("Error: --input required for csv mode")
            sys.exit(1)
        
        print(f"Loading processed data from: {args.input}")
        beats_array, df = load_processed_csv(args.input)
        
        print(f"Loaded {len(beats_array)} beats")
        
        # Classify each beat directly (already processed)
        classifier = HeartbeatClassifier(
            keras_path=args.keras_model,
            sklearn_path=args.sklearn_model
        )
        
        results = []
        print("=" * 60)
        
        for i, beat in enumerate(beats_array):
            # Normalize the beat
            beat_normalized = normalize_beat(beat)
            
            # Classify
            label, prob = classifier.classify(beat_normalized)
            
            result = {
                "beat_index": i,
                "timestamp_sec": df.iloc[i].get("timeframe", i),
                "pred_label": label,
                "pred_prob": prob,
                "category": "Normal" if label == 0 else "Abnormal",
            }
            
            if "record_id" in df.columns:
                result["record_id"] = df.iloc[i]["record_id"]
            
            results.append(result)
            
            symbol = "✓" if label == 0 else "✗"
            print(f"Beat #{i:4d} | "
                  f"Time: {result['timestamp_sec']:8.2f}s | "
                  f"Category: {result['category']:8s} | "
                  f"Confidence: {prob:.2f} {symbol}")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        
        print("=" * 60)
        print("Classification complete!")
        
        normal_count = sum(1 for r in results if r["pred_label"] == 0)
        abnormal_count = len(results) - normal_count
        
        print(f"Total beats: {len(results)}")
        print(f"Normal beats: {normal_count} ({100*normal_count/max(1,len(results)):.1f}%)")
        print(f"Abnormal beats: {abnormal_count} ({100*abnormal_count/max(1,len(results)):.1f}%)")
        print(f"Results saved to: {args.output}")
        
    elif args.mode == "raw":
        if not args.input:
            print("Error: --input required for raw mode")
            sys.exit(1)
        
        print(f"Loading raw ECG from: {args.input}")
        
        # Load raw signal from CSV (single column or first numeric column)
        df = pd.read_csv(args.input)
        
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                signal = df[col].values.astype(np.float32)
                break
        else:
            print("Error: No numeric column found in CSV")
            sys.exit(1)
        
        print(f"Loaded {len(signal)} samples")
        
        # Create pipeline and process
        pipeline = RealtimeECGPipeline(CONFIG)
        pipeline.process_stream(signal, realtime=args.realtime)
        pipeline.save_results(args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
