"""
Inference Engine Module

Responsible for loading ONNX models and performing beat classification.
This module handles:
- Loading ONNX model once at startup
- Loading scaler (fitted on training data only)
- Preprocessing beats to match training pipeline
- Running inference on sliding windows
- Returning R-peak positions and classifications

The inference engine is completely isolated from ground truth annotations.
It only receives signal data and returns predictions.

Model Configurations:
- v2 (CNN): 188-sample single beats, Conv1D architecture
- v3 (LSTM): 188-sample single beats, Bidirectional LSTM
- v5 (Transformer): 188-sample single beats, Multi-head attention
- v6 (Context-Aware CNN1D): 200-sample beats, 7-beat context window
"""

import os
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("ONNXRuntime not found. Install with: pip install onnxruntime")


class InferenceEngine:
    """
    ONNX Model Inference Engine for ECG Beat Classification
    
    This class manages model loading and inference, providing a clean interface
    for beat classification. It supports:
    - Multiple model versions (v2, v3, v5, v6)
    - Context-aware inference (v6 with 7-beat rolling buffer)
    - Preprocessing matching the training pipeline exactly
    
    Attributes:
        model: ONNX Runtime InferenceSession
        scaler: StandardScaler fitted on training data
        config: Model configuration dict
        beat_buffer: Rolling buffer for context-aware models
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'v2': {
            'name': 'CNN (v2)',
            'onnx_file': 'ecg_cnn_v2_pytorch_final.onnx',
            'scaler_file': 'scaler_v2_pytorch.pkl',
            'input_shape': (1, 1, 188),  # (batch, channels, length)
            'beat_length': 188,
            'pre_r_samples': 70,
            'post_r_samples': 118,
            'context_aware': False,
            'description': '4-layer Conv1D with BatchNorm, Dropout 0.5'
        },
        'v3': {
            'name': 'LSTM (v3)',
            'onnx_file': 'ecg_lstm_v3_pytorch_final.onnx',
            'scaler_file': 'scaler_v3_pytorch.pkl',
            'input_shape': (1, 188, 1),  # (batch, timesteps, features)
            'beat_length': 188,
            'pre_r_samples': 70,
            'post_r_samples': 118,
            'context_aware': False,
            'description': 'Bidirectional LSTM with BatchNorm'
        },
        'v5': {
            'name': 'Transformer (v5)',
            'onnx_file': 'ecg_transformer_v5_pytorch_final.onnx',
            'scaler_file': 'scaler_v5_pytorch.pkl',
            'input_shape': (1, 188, 1),  # (batch, timesteps, features)
            'beat_length': 188,
            'pre_r_samples': 70,
            'post_r_samples': 118,
            'context_aware': False,
            'description': '3-layer Transformer encoder, 4-head attention'
        },
        'v6': {
            'name': 'Context-Aware CNN1D (v6)',
            'onnx_file': 'context_ecg_model.onnx',
            'scaler_file': 'context_ecg_scaler.pkl',
            'input_shape': (1, 7, 200),  # (batch, context_beats, length)
            'beat_length': 200,
            'pre_r_samples': 90,
            'post_r_samples': 110,
            'context_aware': True,
            'context_window_size': 7,  # 3 prev + 1 center + 3 next
            'description': 'Context-aware CNN1D with 7-beat temporal window'
        },
    }
    
    def __init__(self, model_version: str = 'v3', models_dir: Optional[str] = None):
        """
        Initialize the Inference Engine.
        
        Args:
            model_version: Model version to use ('v2', 'v3', 'v5', 'v6')
            models_dir: Directory containing ONNX and scaler files.
                       If None, uses default 'sample/' directory.
        """
        if model_version not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model version: {model_version}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.version = model_version
        self.config = self.MODEL_CONFIGS[model_version]
        self.model: Optional[ort.InferenceSession] = None
        self.scaler = None
        
        # Context-aware models use a rolling buffer
        self.beat_buffer: List[np.ndarray] = []
        
        # Determine models directory
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, '..', 'sample')
        
        self.models_dir = models_dir
        
        # Load model and scaler
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model and scaler from files."""
        # Load ONNX model
        onnx_path = os.path.join(self.models_dir, self.config['onnx_file'])
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        self.model = ort.InferenceSession(onnx_path)
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, self.config['scaler_file'])
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
    
    def reset_buffer(self) -> None:
        """Reset the beat buffer (for context-aware models)."""
        self.beat_buffer = []
    
    def extract_beat(self, signal: np.ndarray, r_peak_idx: int) -> np.ndarray:
        """
        Extract a beat window centered on R-peak.
        
        PREPROCESSING (matches training exactly):
        - v2/v3/v5: 188 samples (70 before + 118 after R-peak)
        - v6: 200 samples (90 before + 110 after R-peak)
        
        Args:
            signal: Full ECG signal array
            r_peak_idx: Index of R-peak in signal
            
        Returns:
            Beat window as numpy array
        """
        beat_length = self.config['beat_length']
        pre_samples = self.config['pre_r_samples']
        post_samples = self.config['post_r_samples']
        
        start_idx = r_peak_idx - pre_samples
        end_idx = r_peak_idx + post_samples
        
        # Handle edge cases with zero padding
        if start_idx < 0:
            pad_before = -start_idx
            beat = np.zeros(beat_length, dtype=np.float32)
            available = signal[:end_idx]
            beat[pad_before:pad_before + len(available)] = available
        elif end_idx > len(signal):
            beat = np.zeros(beat_length, dtype=np.float32)
            available = signal[start_idx:]
            beat[:len(available)] = available
        else:
            beat = signal[start_idx:end_idx].astype(np.float32)
        
        return beat
    
    def classify_beat(self, signal: np.ndarray, r_peak_idx: int) -> Dict:
        """
        Classify a single beat.
        
        For context-aware models (v6), this adds the beat to the rolling buffer
        and only returns a valid classification when the buffer is full.
        
        Args:
            signal: Full ECG signal array
            r_peak_idx: Index of R-peak in signal
            
        Returns:
            Dict with:
                - predicted: 'NORMAL' or 'ABNORMAL' (or 'WAITING' if buffer not full)
                - probability: Abnormal probability (0.0 to 1.0)
                - r_peak: R-peak index
                - beat_waveform: Raw beat samples (for visualization)
                - r_peak_pos_in_beat: Position of R-peak in beat waveform
                - beat_length: Length of beat waveform
                - context_aware: Whether this is a context-aware prediction
                - buffer_size: Current buffer size (for context-aware models)
        """
        # Extract beat
        beat = self.extract_beat(signal, r_peak_idx)
        raw_beat = beat.copy()
        
        is_context_aware = self.config.get('context_aware', False)
        
        if is_context_aware:
            # V6: Add to rolling buffer
            self.beat_buffer.append(beat)
            
            context_size = self.config.get('context_window_size', 7)
            if len(self.beat_buffer) > context_size:
                self.beat_buffer = self.beat_buffer[-context_size:]
            
            # Need full buffer for classification
            if len(self.beat_buffer) < context_size:
                return {
                    'r_peak': r_peak_idx,
                    'predicted': 'WAITING',
                    'probability': 0.0,
                    'beat_waveform': raw_beat.tolist(),
                    'r_peak_pos_in_beat': self.config['pre_r_samples'],
                    'beat_length': self.config['beat_length'],
                    'context_aware': True,
                    'buffer_size': len(self.beat_buffer)
                }
            
            # V6 preprocessing (matches training exactly):
            # 1. Stack 7 beats: (7, 200)
            context_beats = np.stack(self.beat_buffer, axis=0)
            
            # 2. Flatten for scaling: (1, 1400)
            flat_size = context_size * self.config['beat_length']
            context_flat = context_beats.reshape(1, flat_size)
            
            # 3. Normalize with scaler (fitted on training data ONLY)
            normalized = self.scaler.transform(context_flat).astype(np.float32)
            
            # 4. Reshape for model: (1, 7, 200)
            model_input = normalized.reshape(1, context_size, self.config['beat_length'])
            
        else:
            # V2, V3, V5: Single beat classification
            beat_2d = beat.reshape(1, -1)
            normalized = self.scaler.transform(beat_2d).flatten().astype(np.float32)
            
            # Reshape for specific model architecture
            model_input = normalized.reshape(self.config['input_shape'])
        
        # Run ONNX inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        output = self.model.run([output_name], {input_name: model_input})[0]
        
        # Process output - apply softmax if needed
        # Validate output shape before accessing elements
        if len(output.shape) < 2 or output.shape[0] < 1:
            raise ValueError(f"Unexpected model output shape: {output.shape}")
        
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
        
        # Clamp to [0, 1]
        prob_abnormal = max(0.0, min(1.0, prob_abnormal))
        
        predicted = 'ABNORMAL' if prob_abnormal >= 0.5 else 'NORMAL'
        
        return {
            'r_peak': r_peak_idx,
            'predicted': predicted,
            'probability': round(prob_abnormal, 4),
            'beat_waveform': raw_beat.tolist(),
            'r_peak_pos_in_beat': self.config['pre_r_samples'],
            'beat_length': self.config['beat_length'],
            'context_aware': is_context_aware,
            'buffer_size': len(self.beat_buffer) if is_context_aware else 1
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model name, version, description, and configuration
        """
        return {
            'version': self.version,
            'name': self.config['name'],
            'description': self.config['description'],
            'beat_length': self.config['beat_length'],
            'pre_r_samples': self.config['pre_r_samples'],
            'post_r_samples': self.config['post_r_samples'],
            'context_aware': self.config.get('context_aware', False),
            'context_window_size': self.config.get('context_window_size', 1),
            'onnx_file': self.config['onnx_file'],
            'scaler_file': self.config['scaler_file']
        }
    
    @classmethod
    def get_available_models(cls) -> List[Dict]:
        """
        Get list of available model configurations.
        
        Returns:
            List of dicts with version, name, and description
        """
        return [
            {
                'version': v,
                'name': cfg['name'],
                'description': cfg['description'],
                'context_aware': cfg.get('context_aware', False)
            }
            for v, cfg in cls.MODEL_CONFIGS.items()
        ]
