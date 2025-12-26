"""
ECG Real-Time Backend Modules

This package contains the modular backend components for the ECG arrhythmia detection system.
Designed for thesis-quality deployment with clear separation of concerns.

Modules:
- ecg_streamer: ECG signal streaming with simulated real-time playback
- inference_engine: ONNX model inference for beat classification
- evaluation_layer: Performance evaluation against ground truth annotations

Architecture follows the MVC pattern:
- Model: inference_engine (ONNX model loading and prediction)
- View: Frontend HTML/JS (in app.py)
- Controller: Flask API routes (in app.py)
"""

from .ecg_streamer import ECGStreamer
from .inference_engine import InferenceEngine
from .evaluation_layer import EvaluationLayer

__all__ = ['ECGStreamer', 'InferenceEngine', 'EvaluationLayer']
