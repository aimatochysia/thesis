import os
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("ONNXRuntime not found. Install with: pip install onnxruntime")


class InferenceEngine:
    
    MODEL_CONFIGS = {
        'v2': {
            'name': 'CNN (v2)',
            'onnx_file': 'ecg_cnn_v2_pytorch_final.onnx',
            'scaler_file': 'scaler_v2_pytorch.pkl',
            'input_shape': (1, 1, 188),
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
            'input_shape': (1, 188, 1),
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
            'input_shape': (1, 188, 1),
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
            'input_shape': (1, 7, 200),
            'beat_length': 200,
            'pre_r_samples': 90,
            'post_r_samples': 110,
            'context_aware': True,
            'context_window_size': 7,
            'description': 'Context-aware CNN1D with 7-beat temporal window'
        },
    }
    
    def __init__(self, model_version: str = 'v3', models_dir: Optional[str] = None):
        if model_version not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model version: {model_version}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")
        
        self.version = model_version
        self.config = self.MODEL_CONFIGS[model_version]
        self.model: Optional[ort.InferenceSession] = None
        self.scaler = None
        
        self.beat_buffer: List[np.ndarray] = []
        
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, '..', 'sample')
        
        self.models_dir = models_dir
        
        self._load_model()
    
    def _load_model(self) -> None:
        onnx_path = os.path.join(self.models_dir, self.config['onnx_file'])
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        self.model = ort.InferenceSession(onnx_path)
        
        scaler_path = os.path.join(self.models_dir, self.config['scaler_file'])
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
    
    def reset_buffer(self) -> None:
        self.beat_buffer = []
    
    def extract_beat(self, signal: np.ndarray, r_peak_idx: int) -> np.ndarray:
        beat_length = self.config['beat_length']
        pre_samples = self.config['pre_r_samples']
        post_samples = self.config['post_r_samples']
        
        start_idx = r_peak_idx - pre_samples
        end_idx = r_peak_idx + post_samples
        
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
        beat = self.extract_beat(signal, r_peak_idx)
        raw_beat = beat.copy()
        
        is_context_aware = self.config.get('context_aware', False)
        
        if is_context_aware:
            self.beat_buffer.append(beat)
            
            context_size = self.config.get('context_window_size', 7)
            if len(self.beat_buffer) > context_size:
                self.beat_buffer = self.beat_buffer[-context_size:]
            
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
            
            context_beats = np.stack(self.beat_buffer, axis=0)
            
            flat_size = context_size * self.config['beat_length']
            context_flat = context_beats.reshape(1, flat_size)
            
            normalized = self.scaler.transform(context_flat).astype(np.float32)
            
            model_input = normalized.reshape(1, context_size, self.config['beat_length'])
            
        else:
            beat_2d = beat.reshape(1, -1)
            normalized = self.scaler.transform(beat_2d).flatten().astype(np.float32)
            
            model_input = normalized.reshape(self.config['input_shape'])
        
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        output = self.model.run([output_name], {input_name: model_input})[0]
        
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
        return [
            {
                'version': v,
                'name': cfg['name'],
                'description': cfg['description'],
                'context_aware': cfg.get('context_aware', False)
            }
            for v, cfg in cls.MODEL_CONFIGS.items()
        ]
