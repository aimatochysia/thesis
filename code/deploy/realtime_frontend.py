"""
ECG Real-Time Classification Frontend

A mini web-based frontend that simulates real-time ECG monitoring and classification.
Features:
- Real-time ECG signal visualization with scrollable history
- Pan-Tompkins R-peak detection (simulates real deployment - no annotation file needed)
- AI model classification using PyTorch ONNX models (v2/v3/v5/v6)
- Live classification results display
- Ground truth comparison (when detected R-peak matches annotation within tolerance)
- False detection log with clickable navigation
- Beat waveform snapshot showing exact input to ONNX model

R-PEAK DETECTION:
- Uses Pan-Tompkins algorithm for real-time R-peak detection
- Does NOT rely on annotation file for R-peak positions
- Annotations are only used for ground truth comparison (if detected R-peak is within 100ms)
- If no matching annotation is found, ground truth shows as "UNKNOWN"

PREPROCESSING (matches training exactly):
- v2/v3/v5: 188 samples per beat (70 before + 118 after R-peak), single beat classification
- v6 (Context-Aware): 200 samples per beat (90 before + 110 after R-peak), 7-beat context window
  - Flatten to (1, 1400) → scale with training scaler → reshape to (1, 7, 200)
  - Uses record 119 by default (excluded from training for true validation)

DATA SOURCES:
- All models (v2/v3/v5/v6): Now use 119.csv by default (MIT-BIH record 119)
  Record 119 was excluded from v6 training, providing true test data for all models
- --training-data: Use demo_training_signal.csv (deprecated, kept for backward compatibility)

Usage:
    python realtime_frontend.py              # Uses v3 (LSTM) by default with record 119
    python realtime_frontend.py --model v2   # Use CNN model with record 119
    python realtime_frontend.py --model v3   # Use LSTM model with record 119
    python realtime_frontend.py --model v5   # Use Transformer model with record 119
    python realtime_frontend.py --model v6   # Use Context-Aware CNN1D (7-beat rolling buffer)
    
    All models now use MIT-BIH record 119 by default for consistent testing.
    Record 119 was excluded from v6 training, providing true validation data.
    
    Then open http://localhost:5000 in your browser
"""

import os
import sys
import argparse
from typing import Optional
from collections import deque
import bisect
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, jsonify, request
from scipy.signal import butter, sosfilt, find_peaks

# ONNX Runtime import for cross-platform inference (PyTorch models exported to ONNX)
try:
    import onnxruntime as ort
    USE_ONNX = True
except ImportError:
    print("Error: ONNXRuntime not found.")
    print("Install ONNXRuntime for ONNX model inference: pip install onnxruntime")
    sys.exit(1)

# Model configurations for v2, v3, v5, v6 PyTorch ONNX models
MODEL_CONFIGS = {
    'v2': {
        'name': 'CNN (v2)',
        'onnx_file': 'ecg_cnn_v2_pytorch_final.onnx',
        'scaler_file': 'scaler_v2_pytorch.pkl',
        'input_shape': (1, 1, 188),  # CNN: (batch, channels, length)
        'beat_length': 188,
        'context_aware': False,
    },
    'v3': {
        'name': 'LSTM (v3)',
        'onnx_file': 'ecg_lstm_v3_pytorch_final.onnx',
        'scaler_file': 'scaler_v3_pytorch.pkl',
        'input_shape': (1, 188, 1),  # LSTM: (batch, timesteps, features)
        'beat_length': 188,
        'context_aware': False,
    },
    'v5': {
        'name': 'Transformer (v5)',
        'onnx_file': 'ecg_transformer_v5_pytorch_final.onnx',
        'scaler_file': 'scaler_v5_pytorch.pkl',
        'input_shape': (1, 188, 1),  # Transformer: (batch, timesteps, features)
        'beat_length': 188,
        'context_aware': False,
    },
    'v6': {
        'name': 'Context-Aware CNN1D (v6)',
        'onnx_file': 'context_ecg_model.onnx',
        'scaler_file': 'context_ecg_scaler.pkl',
        'input_shape': (1, 7, 200),  # (batch, channels=7_beats_as_channels, length=200)
        'beat_length': 200,
        'context_aware': True,
        'context_window_size': 7,
        'pre_r_samples': 90,
        'post_r_samples': 110,
    },
}

# Constants
BEAT_LENGTH = 188  # Default beat length (v2, v3, v5)
BEAT_LENGTH_V6 = 200  # v6 beat length
PRE_SAMPLES = 70
POST_SAMPLES = 118
PRE_SAMPLES_V6 = 90
POST_SAMPLES_V6 = 110
CONTEXT_WINDOW_SIZE = 7  # v6: 3 previous + 1 center + 3 subsequent beats
SAMPLING_RATE = 360  # Hz - MIT-BIH standard sampling rate
# Beat type classification: 'N' is Normal, anything else is Abnormal
NORMAL_BEAT_TYPE = 'N'

# R-peak detection tolerance (samples) for matching detected R-peaks to annotations
R_PEAK_TOLERANCE = 36  # 100ms at 360Hz - detected R-peak must be within 100ms of annotation


# -------------------------------
# Batch R-peak Detector with Look-Ahead
# -------------------------------
# 
# Designed for real-time deployment with look-ahead:
# 1. Batch processing for efficiency at all playback speeds
# 2. Uses scipy.signal.find_peaks for reliable detection
# 3. Detection runs ahead of display position
# 4. Beats classified when complete window available
#
# Based on Pan & Tompkins principles with batch optimizations.

def butter_bandpass_sos(lowcut, highcut, fs, order=2):
    """Design a bandpass filter using second-order sections."""
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos


class BatchRPeakDetector:
    """
    Batch R-peak detector optimized for real-time ECG classification.
    
    Key features:
    - Batch processing: handles chunks of samples efficiently
    - Uses scipy.signal.find_peaks for reliable detection
    - Short warmup (360 samples = 1 second) for faster start
    - Optimized for MLII lead at 360 Hz
    
    Detection approach:
    1. Bandpass filter 5-15 Hz (QRS isolation)
    2. Squared derivative for peak enhancement
    3. Moving window integration
    4. scipy.signal.find_peaks with height and distance constraints
    5. R-peak refinement in original signal
    """
    
    # Detection parameters (OPTIMIZED for MIT-BIH 360Hz MLII)
    MIN_DISTANCE_MS = 280  # 280ms minimum between beats (max ~214 BPM)
    MWI_WINDOW_MS = 80  # 80ms moving window integration
    HEIGHT_PERCENTILE = 60  # Lowered for better sensitivity
    SEARCH_WINDOW_MS = 40  # 40ms search window for R-peak refinement
    WARMUP_MS = 1000  # 1 second warmup
    
    def __init__(self, fs: int = 360):
        """Initialize detector."""
        self.fs = fs
        
        # Convert ms parameters to samples
        self.min_distance = int(self.MIN_DISTANCE_MS * fs / 1000)
        self.mwi_window = int(self.MWI_WINDOW_MS * fs / 1000)
        self.search_window = int(self.SEARCH_WINDOW_MS * fs / 1000)
        self.warmup_samples = int(self.WARMUP_MS * fs / 1000)
        
        # Filter design
        self.sos = butter(2, [5, 15], btype='bandpass', fs=fs, output='sos')
        
        # Signal storage
        self.signal_buffer = []  # Raw signal
        self.processed_up_to = 0  # Last processed sample index
        self.detected_peaks = set()  # All detected R-peak indices (set for O(1) lookup)
        self.last_peak_idx = -self.min_distance * 2
        
        # Threshold (will be set during warmup)
        self.threshold = None
        self.initialized = False

    def process_batch(self, start_idx: int, end_idx: int, signal: np.ndarray) -> list:
        """
        Process a batch of samples and detect R-peaks.
        
        This is the main detection method - efficient batch processing.
        
        Args:
            start_idx: Starting sample index to process
            end_idx: Ending sample index (exclusive)
            signal: Full ECG signal array
            
        Returns:
            List of newly detected R-peak indices
        """
        # Ensure we have enough signal
        if end_idx > len(signal):
            end_idx = len(signal)
        
        if end_idx <= start_idx:
            return []
        
        # Need warmup period
        if end_idx < self.warmup_samples:
            return []
        
        # Get segment with overlap for edge handling
        overlap = self.mwi_window + self.search_window + 10
        seg_start = max(0, start_idx - overlap)
        segment = signal[seg_start:end_idx].astype(np.float32)
        
        if len(segment) < self.mwi_window + 10:
            return []
        
        # Apply bandpass filter
        filtered = sosfilt(self.sos, segment)
        
        # Compute derivative (5-point)
        deriv = np.zeros_like(filtered)
        for i in range(4, len(filtered)):
            deriv[i] = (filtered[i] - filtered[i-4]) + 2 * (filtered[i-1] - filtered[i-3])
        deriv /= 8.0
        
        # Square and MWI
        squared = deriv ** 2
        mwi = np.convolve(squared, np.ones(self.mwi_window) / self.mwi_window, mode='same')
        
        # Initialize or update threshold
        if not self.initialized:
            self.threshold = np.percentile(mwi, self.HEIGHT_PERCENTILE) * 0.35
            self.initialized = True
        
        # Find peaks using scipy
        peaks, _ = find_peaks(
            mwi,
            height=self.threshold,
            distance=self.min_distance
        )
        
        # Convert to global indices and filter
        new_peaks = []
        for peak_local in peaks:
            peak_global = seg_start + peak_local
            
            # Skip if before the range we're processing
            if peak_global < start_idx:
                continue
            
            # Skip if already detected
            if peak_global in self.detected_peaks:
                continue
            
            # Skip if too close to last peak
            if peak_global - self.last_peak_idx < self.min_distance:
                continue
            
            # Refine R-peak location in original signal
            refined = self._refine_peak(peak_global, signal)
            
            # Check distance again after refinement
            if refined - self.last_peak_idx < self.min_distance:
                continue
            
            # Check if already detected (after refinement)
            if refined in self.detected_peaks:
                continue
            
            # Accept this peak
            self.detected_peaks.add(refined)
            self.last_peak_idx = refined
            new_peaks.append(refined)
            
            # Update threshold
            if peak_local < len(mwi):
                self.threshold = 0.85 * self.threshold + 0.15 * mwi[peak_local] * 0.35
        
        return new_peaks

    def _refine_peak(self, global_idx: int, signal: np.ndarray) -> int:
        """Refine R-peak location by finding maximum in original signal."""
        search_start = max(0, global_idx - self.search_window)
        search_end = min(len(signal), global_idx + self.search_window + 1)
        
        if search_end <= search_start:
            return global_idx
        
        segment = signal[search_start:search_end]
        
        # R-peaks in MLII are typically positive
        local_max_idx = int(np.argmax(segment))
        return search_start + local_max_idx

    def get_all_peaks(self) -> list:
        """Get all detected peaks as sorted list."""
        return sorted(self.detected_peaks)

    def reset(self):
        """Reset detector state."""
        self.__init__(self.fs)


# Main detector class - alias for backward compatibility
# Main detector class - alias for backward compatibility
RealtimeRPeakDetector = BatchRPeakDetector
HamiltonTompkinsDetector = BatchRPeakDetector
PanTompkinsDetector = BatchRPeakDetector

# Global R-peak detector instance
rpeak_detector = None
filtered_signal = None  # Raw ECG signal (detector handles filtering internally)

# Pre-sorted annotation indices for efficient binary search
annotation_indices = None  # Sorted list of annotation sample indices


# Global state
app = Flask(__name__)
ecg_data = None
annotations = None
model = None
scaler = None
model_config = None  # Current model configuration
current_sample = 0
classification_results = []
is_running = False
speed_multiplier = 10  # Speed up simulation (10x faster)

# Rolling beat buffer for v6 context-aware model
beat_buffer = []  # List of (beat_waveform, beat_type) tuples


def load_data(model_version='v3', use_training_data=False, use_record_119=True):
    """Load ECG signal, annotations, model, and scaler.
    
    Args:
        model_version: Which model to use ('v2', 'v3', 'v5', 'v6')
        use_training_data: If True, use demo data from training set (deprecated).
                          All models now use 119.csv by default.
        use_record_119: If True (default), use record 119 (excluded from training - true test).
                       This is now the default for ALL models (v2, v3, v5, v6).
    
    Preprocessing (matches training exactly):
    - v2/v3/v5: 188 samples per beat (70 before + 118 after R-peak)
    - v6: 200 samples per beat (90 before + 110 after R-peak), 7-beat context window
    - Normalization: Uses the same scaler trained on training data ONLY
    
    R-Peak Detection:
    - Uses Pan-Tompkins algorithm to detect R-peaks in real-time
    - Does NOT rely on annotation file for R-peak positions
    - Annotations are only used for ground truth comparison (if matched within tolerance)
    
    NOTE: All models now use MIT-BIH record 119 by default for consistent testing.
    Record 119 was excluded from v6 training, and using it for all models provides
    a fair comparison on unseen real ECG data.
    """
    global ecg_data, annotations, model, scaler, model_config, beat_buffer
    global rpeak_detector, filtered_signal
    
    # Reset beat buffer for v6 context-aware model
    beat_buffer = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # All models now use record 119 by default (the reserved test record)
    # This ensures consistent comparison across v2, v3, v5, and v6
    print(f"{MODEL_CONFIGS[model_version]['name']}: Using record 119 (excluded from training) for validation")
    
    # Choose data source - default is now record 119 for all models
    if use_record_119:
        # Use MIT-BIH record 119 - excluded from v6 training, provides true test for all models
        signal_path = os.path.join(sample_dir, '119.csv')
        annotation_path = os.path.join(sample_dir, '119annotations.txt')
        print("Using MIT-BIH record 119 (excluded from training - true test data)")
    elif use_training_data:
        # Use demo data created from training set - deprecated, kept for backward compatibility
        signal_path = os.path.join(sample_dir, 'demo_training_signal.csv')
        annotation_path = os.path.join(sample_dir, 'demo_training_annotations.txt')
        if not os.path.exists(signal_path):
            print("Warning: Training demo data not found, falling back to record 119")
            signal_path = os.path.join(sample_dir, '119.csv')
            annotation_path = os.path.join(sample_dir, '119annotations.txt')
    else:
        # Fallback to record 119
        signal_path = os.path.join(sample_dir, '119.csv')
        annotation_path = os.path.join(sample_dir, '119annotations.txt')
        print("Using MIT-BIH record 119 (excluded from training - true test data)")
    
    # Load signal
    df = pd.read_csv(signal_path)
    df.columns = df.columns.str.strip().str.strip("'")
    ecg_data = df['MLII'].values.astype(np.float32)
    
    # Raw signal - BatchRPeakDetector handles filtering internally
    filtered_signal = ecg_data
    
    # Initialize batch R-peak detector (optimized for 1x speed with look-ahead)
    rpeak_detector = BatchRPeakDetector(fs=SAMPLING_RATE)
    print("✓ Batch R-peak detector initialized")
    print("   - Optimized for real-time with look-ahead detection")
    print("   - 280ms refractory period (max ~214 BPM)")
    print("   - 1 second warmup for threshold calibration")
    print("   - Internal bandpass filter 5-15 Hz")
    
    # Load annotations (used only for ground truth comparison, NOT for R-peak positions)
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
    
    # Create sorted annotation index for efficient binary search
    annotation_indices = sorted(annotations['sample_index'].tolist())
    
    print(f"  NOTE: Annotations used for ground truth validation only, NOT for R-peak detection")
    
    # Get model configuration
    if model_version not in MODEL_CONFIGS:
        print(f"Unknown model version '{model_version}'. Using v3 (LSTM) as default.")
        model_version = 'v3'
    
    model_config = MODEL_CONFIGS[model_version]
    print(f"\nLoading {model_config['name']} model...")
    
    # Load ONNX model
    onnx_model_path = os.path.join(sample_dir, model_config['onnx_file'])
    if os.path.exists(onnx_model_path):
        print(f"Loading ONNX model from: {onnx_model_path}")
        model = ort.InferenceSession(onnx_model_path)
        print(f"✓ {model_config['name']} ONNX model loaded successfully")
    else:
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    # Load scaler
    scaler_path = os.path.join(sample_dir, model_config['scaler_file'])
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    print(f"\nLoaded {len(ecg_data)} ECG samples")
    print(f"Loaded {len(annotations)} annotations (for ground truth only)")


def find_matching_annotation(detected_r_peak: int, tolerance: int = R_PEAK_TOLERANCE):
    """
    Find an annotation that matches a detected R-peak within tolerance.
    Uses binary search for O(log n) efficiency.
    
    This is used for ground truth comparison only. The detected R-peak is from
    the Pan-Tompkins algorithm, and we try to find a matching annotation to
    determine the ground truth label.
    
    Args:
        detected_r_peak: Sample index of detected R-peak
        tolerance: Maximum distance (samples) between detected and annotated R-peak
        
    Returns:
        dict with 'beat_type' and 'sample_index' if found, None otherwise
    """
    if annotations is None or len(annotations) == 0:
        return None
    
    # Use binary search to find candidate annotations within tolerance range
    ann_indices = annotations['sample_index'].values
    
    # Find insertion point for the detected R-peak
    idx = bisect.bisect_left(ann_indices, detected_r_peak)
    
    # Check nearby annotations (before and after insertion point)
    best_match = None
    best_distance = tolerance + 1
    
    for check_idx in [idx - 1, idx, idx + 1]:
        if 0 <= check_idx < len(ann_indices):
            distance = abs(ann_indices[check_idx] - detected_r_peak)
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match = check_idx
    
    if best_match is not None:
        ann = annotations.iloc[best_match]
        return {
            'beat_type': ann['beat_type'],
            'sample_index': int(ann['sample_index']),
            'distance': int(best_distance)
        }
    
    return None


def extract_beat_v6(signal, r_peak_idx):
    """Extract beat for v6 context-aware model.
    
    PREPROCESSING (matches training exactly):
    - Beat length: 200 samples (90 before R-peak + 110 after R-peak)
    - This matches the dataset creator: PRE_R_SAMPLES=90, POST_R_SAMPLES=110
    - Edge cases handled with zero padding
    """
    start_idx = r_peak_idx - PRE_SAMPLES_V6  # 90 samples before R-peak
    end_idx = r_peak_idx + POST_SAMPLES_V6    # 110 samples after R-peak
    
    # Handle edge cases with zero padding
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
    """Extract beat at R-peak and classify it using PyTorch ONNX model."""
    global beat_buffer
    
    # Check if using v6 context-aware model
    is_context_aware = model_config.get('context_aware', False)
    
    if is_context_aware:
        # V6: Extract 200-sample beat and add to rolling buffer
        beat = extract_beat_v6(signal, r_peak_idx)
        raw_beat = beat.copy()
        
        # Add beat to buffer
        beat_buffer.append((beat, beat_type))
        
        # Keep only last 7 beats
        if len(beat_buffer) > CONTEXT_WINDOW_SIZE:
            beat_buffer = beat_buffer[-CONTEXT_WINDOW_SIZE:]
        
        # Need 7 beats for context-aware inference
        if len(beat_buffer) < CONTEXT_WINDOW_SIZE:
            # Not enough beats yet, return waiting status
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
        
        # ===== V6 PREPROCESSING (matches training exactly) =====
        # 1. Stack 7 beats into context window: (7, 200)
        context_beats = np.stack([b for b, _ in beat_buffer], axis=0)
        
        # 2. Flatten for scaling: (1, 7*200) = (1, 1400)
        #    This matches training: X_train_flat = X_train.reshape(n_train, flat_size)
        flat_size = CONTEXT_WINDOW_SIZE * BEAT_LENGTH_V6  # 7 * 200 = 1400
        context_flat = context_beats.reshape(1, flat_size)
        
        # 3. Normalize using scaler (fitted on training data ONLY)
        #    This matches training: scaler.fit_transform(X_train_flat)
        normalized = scaler.transform(context_flat).astype(np.float32)
        
        # 4. Reshape back to (1, 7, 200) for model input
        #    This matches training: X_train_norm.reshape(-1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH)
        context_input = normalized.reshape(1, CONTEXT_WINDOW_SIZE, BEAT_LENGTH_V6)
        
        # Center beat info (index 3 in window of 7: positions 0,1,2,3,4,5,6)
        center_beat_type = beat_buffer[3][1]
        
    else:
        # V2, V3, V5: Single beat classification (188 samples)
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
        
        raw_beat = beat.copy()
        
        # Normalize using the scaler (fitted only on training data)
        beat_2d = beat.reshape(1, -1)
        normalized = scaler.transform(beat_2d).flatten().astype(np.float32)
        
        # Reshape for the specific model architecture
        input_shape = model_config['input_shape']
        context_input = normalized.reshape(input_shape)
        center_beat_type = beat_type
    
    # ONNX model inference
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: context_input})[0]
    
    # Handle output - PyTorch models output raw logits, apply softmax
    # Output: 0 = Normal, 1 = Abnormal
    if output.shape[1] == 2:
        # Check if output looks like logits (any value outside [0,1] range or values don't sum to 1)
        needs_softmax = (np.min(output) < 0 or np.max(output) > 1 or 
                         abs(np.sum(output[0]) - 1.0) > 0.01)
        if needs_softmax:
            # Apply softmax to convert logits to probabilities
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            proba = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        else:
            proba = output
        prob_abnormal = float(proba[0, 1])
    else:
        # Single output, assume sigmoid was applied
        prob_abnormal = float(output[0, 0])
    
    # Clamp probability to [0, 1] range
    prob_abnormal = max(0.0, min(1.0, prob_abnormal))
    
    predicted_class = 1 if prob_abnormal >= 0.5 else 0
    predicted_label = "ABNORMAL" if predicted_class == 1 else "NORMAL"
    
    # Get ground truth: 'N' is Normal, '?' means no matching annotation (unknown), anything else is Abnormal
    if center_beat_type == '?':
        ground_truth = "UNKNOWN"
        correct = None  # Cannot determine correctness without ground truth
    elif center_beat_type == NORMAL_BEAT_TYPE:
        ground_truth = "NORMAL"
        correct = ground_truth == predicted_label
    else:
        ground_truth = "ABNORMAL"
        correct = ground_truth == predicted_label
    
    # Include R-peak position in beat waveform for accurate marker placement
    if is_context_aware:
        r_peak_pos_in_beat = PRE_SAMPLES_V6  # R-peak is at sample 90 for v6
    else:
        r_peak_pos_in_beat = PRE_SAMPLES  # R-peak is at sample 70 for v2/v3/v5
    
    result = {
        'r_peak': r_peak_idx,
        'beat_type': center_beat_type,
        'ground_truth': ground_truth,
        'predicted': predicted_label,
        'probability': round(prob_abnormal, 4),
        'correct': correct,
        'beat_waveform': raw_beat.tolist(),  # Include raw beat for visualization
        'r_peak_pos_in_beat': r_peak_pos_in_beat,  # Position of R-peak in beat waveform
        'beat_length': BEAT_LENGTH_V6 if is_context_aware else BEAT_LENGTH
    }
    
    if is_context_aware:
        result['context_aware'] = True
        result['buffer_size'] = len(beat_buffer)
    
    return result


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
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <button onclick="scrollHistory(-5)" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">⏪ -5s</button>
                <button onclick="scrollHistory(-1)" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">◀ -1s</button>
                <button id="liveBtn" onclick="goToLive()" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: linear-gradient(45deg, #ffd700, #ffb700); border: none; color: #1a1a2e; font-weight: bold; cursor: pointer;">🔴 Live</button>
                <button id="fwdBtn" onclick="scrollHistory(1)" disabled style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">▶ +1s</button>
                <button id="fwd5Btn" onclick="scrollHistory(5)" disabled style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; cursor: pointer;">⏩ +5s</button>
                <span style="margin: 0 10px; color: #444;">|</span>
                <button onclick="exportECG('png')" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: #00ff88; cursor: pointer;">📷 Export PNG</button>
                <button onclick="exportECG('jpeg')" style="padding: 5px 15px; font-size: 12px; border-radius: 15px; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: #00ff88; cursor: pointer;">📄 Export JPEG</button>
            </div>
            <p style="text-align: center; color: #666; font-size: 11px; margin-top: 8px;">💡 Drag the graph to scroll through history</p>
        </div>
        
        <!-- Beat Snapshot Panel - Shows the current beat segment sent to ONNX model -->
        <div class="beat-snapshot-container" style="background: rgba(0, 0, 0, 0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(0, 255, 136, 0.3);">
            <h3 style="color: #00ff88; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 255, 136, 0.3); padding-bottom: 10px;">💓 Current Beat Snapshot (Input to ONNX Model)</h3>
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
        let filteredSignal = [];  // Filtered signal for R-peak detection
        let annotations = [];  // Used for ground truth comparison only
        let rPeakTolerance = 36;  // Tolerance for matching detected R-peaks to annotations
        let currentIndex = 0;
        let isRunning = false;
        let animationId = null;
        let displayBuffer = [];
        let classifications = [];
        let falseDetections = [];
        let detectedRPeaks = [];  // R-peaks detected by Pan-Tompkins (not from annotations)
        let beatTimes = [];  // Store recent beat times for BPM calculation
        let speedMultiplier = 1;  // 1x = real-time
        let currentBeatWaveform = null;
        let currentRPeakPos = 70;  // R-peak position in beat waveform
        let currentBeatLength = 188;  // Beat length
        
        // Pan-Tompkins detector state (client-side for visualization)
        let lastDetectorIndex = 0;  // Track where we left off in detection
        
        // Graph height tracking - expand but never shrink for better readability
        let maxGraphHeight = 300;  // Track maximum height achieved
        const MIN_GRAPH_HEIGHT = 300;  // Minimum height
        const MAX_GRAPH_HEIGHT = 800;  // Maximum allowed height
        
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
            // Ensure canvas height never shrinks below max achieved
            const currentCSSHeight = parseInt(canvas.style.height) || MIN_GRAPH_HEIGHT;
            if (currentCSSHeight < maxGraphHeight) {
                canvas.style.height = maxGraphHeight + 'px';
            }
            
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
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
            filteredSignal = data.filtered_signal;
            annotations = data.annotations;  // For ground truth comparison only
            rPeakTolerance = data.r_peak_tolerance || 36;
            console.log(`Loaded ${ecgData.length} ECG samples`);
            console.log(`Loaded ${annotations.length} annotations (for ground truth comparison only)`);
            console.log('R-peak detection: Using Pan-Tompkins algorithm');
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
            
            // Find min/max for scaling
            const minVal = Math.min(...buffer);
            const maxVal = Math.max(...buffer);
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
            
            // Draw DETECTED R-peak markers (from Pan-Tompkins algorithm)
            // These are the actual R-peaks the system detected, not from annotations
            detectedRPeaks.forEach(rPeak => {
                if (rPeak > startSample && rPeak <= endSample) {
                    const bufferIdx = rPeak - startSample;
                    if (bufferIdx >= 0 && bufferIdx < buffer.length) {
                        const x = (bufferIdx / DISPLAY_SAMPLES) * width;
                        const y = height - ((buffer[bufferIdx] - minVal) / range) * (height - 40) - 20;
                        
                        // Check classification result for this detected R-peak
                        const classResult = classifications.find(c => c.r_peak === rPeak);
                        
                        // Draw false detection indicator (yellow ring)
                        if (classResult && classResult.correct === false) {
                            ctx.strokeStyle = '#ffd700';
                            ctx.lineWidth = 3;
                            ctx.beginPath();
                            ctx.arc(x, y, 10, 0, Math.PI * 2);
                            ctx.stroke();
                        }
                        
                        // Draw R-peak marker based on prediction
                        // If no matching annotation, show as cyan (unknown ground truth)
                        if (classResult) {
                            if (classResult.beat_type === '?') {
                                // No matching annotation - cyan marker
                                ctx.fillStyle = '#00ccff';
                            } else if (classResult.predicted === 'NORMAL') {
                                ctx.fillStyle = '#00ff88';
                            } else {
                                ctx.fillStyle = '#ff4757';
                            }
                        } else {
                            ctx.fillStyle = '#00ccff';  // Detected but not yet classified
                        }
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
        
        // Look-ahead R-peak detection and classification
        // Detection runs AHEAD of display, classification when beat is complete
        const POST_R_SAMPLES = 110;  // v6: 110 samples after R-peak needed for beat extraction
        const LOOK_AHEAD_SAMPLES = Math.round(SAMPLING_RATE * 0.5);  // 0.5 second look-ahead
        
        // Queue for pending classifications (detected but not yet complete)
        let pendingBeats = [];  // [{r_peak, matched_ann}, ...]
        
        async function checkForBeats() {
            // Skip if already processing
            if (isClassifying) return;
            
            // Look-ahead detection: run detector ahead of current display position
            const detectEnd = Math.min(currentIndex + LOOK_AHEAD_SAMPLES, ecgData.length);
            
            if (lastDetectorIndex >= detectEnd) return;
            
            isClassifying = true;
            
            try {
                // Call server to detect R-peaks in the range
                const detectResponse = await fetch('/api/detect_rpeak', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        start_idx: lastDetectorIndex,
                        end_idx: detectEnd
                    })
                });
                const detectResult = await detectResponse.json();
                lastDetectorIndex = detectEnd;
                
                // Add newly detected R-peaks to pending queue
                for (const rPeak of detectResult.detected_peaks) {
                    if (processedBeats.has(rPeak)) continue;
                    
                    const matchKey = String(rPeak);
                    const matchedAnn = detectResult.matched_annotations[matchKey];
                    
                    pendingBeats.push({
                        r_peak: rPeak,
                        matched_ann: matchedAnn
                    });
                    
                    // Store for visualization (will appear when R-peak scrolls into view)
                    detectedRPeaks.push(rPeak);
                }
                
                // Process pending beats that now have complete windows
                // A beat is complete when currentIndex >= r_peak + POST_R_SAMPLES
                const readyBeats = [];
                const stillPending = [];
                
                for (const beat of pendingBeats) {
                    if (currentIndex >= beat.r_peak + POST_R_SAMPLES) {
                        readyBeats.push(beat);
                    } else {
                        stillPending.push(beat);
                    }
                }
                pendingBeats = stillPending;
                
                // Classify ready beats
                for (const beat of readyBeats) {
                    if (processedBeats.has(beat.r_peak)) continue;
                    processedBeats.add(beat.r_peak);
                    
                    // Limit processed beats set size
                    if (processedBeats.size > 5000) {
                        const toRemove = [...processedBeats].slice(0, 1000);
                        toRemove.forEach(v => processedBeats.delete(v));
                    }
                    
                    const beatType = beat.matched_ann ? beat.matched_ann.beat_type : null;
                    
                    try {
                        const response = await fetch('/api/classify', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                r_peak: beat.r_peak,
                                beat_type: beatType
                            })
                        });
                        const result = await response.json();
                        
                        if (beatType) {
                            console.log('[ECG] Classified R-peak at', beat.r_peak, '(matched ann:', beat.matched_ann.sample_index, ') →', result.predicted);
                        } else {
                            console.log('[ECG] Classified R-peak at', beat.r_peak, '(no matching annotation) →', result.predicted);
                        }
                        
                        addClassification(result);
                        
                        const bpm = calculateBPM(beat.r_peak);
                        if (bpm !== null && bpm > 0 && bpm < 300) {
                            document.getElementById('heartRate').textContent = bpm;
                        }
                    } catch (e) {
                        console.error('Classification error:', e);
                    }
                }
            } catch (e) {
                console.error('R-peak detection error:', e);
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
                if (result.beat_type === '?') {
                    beatTypeEl.textContent = '? (no match)';
                    beatTypeEl.style.color = '#00ccff';  // Cyan for unknown
                } else {
                    beatTypeEl.textContent = result.beat_type;
                    beatTypeEl.style.color = result.beat_type === 'N' ? '#00ff88' : '#ff4757';
                }
                
                const groundTruthEl = document.getElementById('groundTruthDisplay');
                if (result.ground_truth === 'UNKNOWN') {
                    groundTruthEl.textContent = 'UNKNOWN';
                    groundTruthEl.style.color = '#00ccff';  // Cyan for unknown
                } else {
                    groundTruthEl.textContent = result.ground_truth;
                    groundTruthEl.style.color = result.ground_truth === 'NORMAL' ? '#00ff88' : '#ff4757';
                }
                
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
            }
            
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
            lastFrameTime = performance.now();
            animationId = requestAnimationFrame(animate);
        }
        
        function stopSimulation() {
            isRunning = false;
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        }
        
        async function resetSimulation() {
            stopSimulation();
            currentIndex = 0;
            classifications = [];
            falseDetections = [];
            detectedRPeaks = [];  // Reset detected R-peaks
            beatTimes = [];
            currentBeatWaveform = null;
            viewOffset = 0;
            isLive = true;
            
            // Reset high-speed stability tracking
            isClassifying = false;
            classificationQueue = [];
            pendingBeats = [];  // Reset pending beats queue
            processedBeats.clear();
            lastDetectorIndex = 0;  // Reset detector position
            
            // Reset server-side detector
            try {
                await fetch('/api/reset_detector', { method: 'POST' });
                console.log('[ECG] Batch R-peak detector reset');
            } catch (e) {
                console.error('Failed to reset detector:', e);
            }
            
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
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    """Return ECG signal and filtered signal for R-peak detection as JSON.
    
    NOTE: Annotations are included for ground truth comparison only.
    The frontend should use its own R-peak detection, NOT rely on annotation positions.
    """
    return jsonify({
        'signal': ecg_data.tolist(),
        'filtered_signal': filtered_signal.tolist(),
        'annotations': annotations.to_dict('records'),
        'r_peak_tolerance': R_PEAK_TOLERANCE
    })


@app.route('/api/detect_rpeak', methods=['POST'])
def detect_rpeak():
    """
    Detect R-peaks for a range of samples using batch processing.
    
    This endpoint processes ECG samples in batches for efficiency.
    Uses scipy.signal.find_peaks for reliable detection.
    
    Request JSON:
        start_idx: Starting sample index
        end_idx: Ending sample index (exclusive)
    
    Returns:
        detected_peaks: List of detected R-peak indices
        matched_annotations: Dict mapping detected peaks to matching annotations (if any)
    """
    global rpeak_detector
    
    data = request.json
    start_idx = data.get('start_idx', 0)
    end_idx = data.get('end_idx', len(ecg_data))
    
    # Use batch processing for efficiency
    detected_peaks = rpeak_detector.process_batch(start_idx, end_idx, ecg_data)
    
    # Find matching annotations for each detected peak
    matched_annotations = {}
    for r_peak in detected_peaks:
        match = find_matching_annotation(r_peak)
        if match:
            matched_annotations[str(r_peak)] = match
    
    return jsonify({
        'detected_peaks': detected_peaks,
        'matched_annotations': matched_annotations
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a single beat at a detected R-peak position.
    
    The R-peak should be from the Pan-Tompkins detector, NOT from annotations.
    The beat_type is optional - if provided, it's used for ground truth comparison.
    If not provided, we try to find a matching annotation for ground truth.
    """
    data = request.json
    r_peak = data['r_peak']
    
    # Get beat type from request or try to find matching annotation
    if 'beat_type' in data and data['beat_type']:
        beat_type = data['beat_type']
    else:
        # Try to find matching annotation for ground truth
        match = find_matching_annotation(r_peak)
        if match:
            beat_type = match['beat_type']
        else:
            beat_type = '?'  # Unknown - no matching annotation found
    
    result = extract_and_classify_beat(ecg_data, r_peak, beat_type)
    return jsonify(result)


@app.route('/api/model_info')
def get_model_info():
    """Return current model information."""
    return jsonify({
        'name': model_config['name'],
        'onnx_file': model_config['onnx_file'],
        'scaler_file': model_config['scaler_file'],
        'r_peak_detection': 'Batch detector with look-ahead',
        'r_peak_tolerance': R_PEAK_TOLERANCE,
    })


@app.route('/api/reset_detector', methods=['POST'])
def reset_detector():
    """Reset the R-peak detector and beat buffer.
    
    Called when the simulation is reset to start fresh.
    """
    global rpeak_detector, beat_buffer
    
    # Reset the batch detector
    rpeak_detector = BatchRPeakDetector(fs=SAMPLING_RATE)
    
    # Reset the beat buffer for v6 context-aware model
    beat_buffer = []
    
    return jsonify({'status': 'ok', 'message': 'Detector reset successfully'})


def main():
    """Run the real-time frontend."""
    # Parse command-line arguments
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
    print("Using PyTorch ONNX Models + Look-Ahead R-Peak Detection")
    print("=" * 60)
    
    print(f"\nSelected model: {args.model.upper()}")
    if args.model == 'v6':
        print("  Context-Aware CNN1D: Uses 7-beat rolling buffer (3 prev + center + 3 next)")
        print("  Beat extraction: 200 samples (90 before + 110 after R-peak)")
        print("  Normalization: Flatten 7x200 → scale → reshape to (7, 200)")
        print("  First 3 beats will show 'WAITING' status until buffer is full")
    else:
        print(f"  Single-beat classification: 188 samples (70 before + 118 after R-peak)")
    
    print("\nR-Peak Detection: Batch Detector with Look-Ahead")
    print(f"  Tolerance for ground truth matching: {R_PEAK_TOLERANCE} samples (~{R_PEAK_TOLERANCE/SAMPLING_RATE*1000:.0f}ms)")
    print("  NOTE: R-peaks detected by algorithm, NOT from annotation file")
    print("  Annotations used only for ground truth validation when matched")
    print("  Look-ahead: Detection runs 0.5s ahead of display for timely classification")
    
    # All models now use record 119 by default
    # --training-data flag allows falling back to demo data (deprecated)
    use_record_119 = not args.training_data
    use_training_data = args.training_data
    
    print("\nData: Using MIT-BIH record 119 (excluded from training - true validation)")
    
    print("\nLoading data and model...")
    load_data(model_version=args.model, use_training_data=use_training_data, use_record_119=use_record_119)
    
    print(f"\nStarting web server on port {args.port}...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
