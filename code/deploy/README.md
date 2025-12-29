# ECG Real-Time Classification Frontend - Deployment Guide

## Overview

This is a web-based real-time ECG monitoring and classification system that uses PyTorch ONNX models (CNN, LSTM, or Transformer) for heartbeat classification. The application supports cross-platform deployment using lightweight ONNX Runtime inference.

## Key Features

- ✅ **Multiple Models**: Choose between CNN (v2), LSTM (v3), or Transformer (v5)
- ✅ **PyTorch ONNX**: Uses PyTorch models exported to ONNX format
- ✅ **Cross-platform**: Works on Windows, Linux, and macOS
- ✅ **Lightweight deployment**: Uses ONNX Runtime (10MB) instead of TensorFlow (500MB+)
- ✅ **Real-time visualization**: Interactive ECG signal display
- ✅ **Live classification**: Classifies heartbeats as Normal or Abnormal
- ✅ **No Data Leakage**: Scalers fit only on training data

## Available Models

| Version | Model Type | Architecture | ONNX File | Scaler File |
|---------|-----------|--------------|-----------|-------------|
| v2 | CNN | 4-layer Conv1D + BatchNorm | `ecg_cnn_v2_pytorch_final.onnx` | `scaler_v2_pytorch.pkl` |
| v3 | LSTM | Bidirectional LSTM | `ecg_lstm_v3_pytorch_final.onnx` | `scaler_v3_pytorch.pkl` |
| v5 | Transformer | 3-layer, 4-head attention | `ecg_transformer_v5_pytorch_final.onnx` | `scaler_v5_pytorch.pkl` |

## Installation

```bash
# Install minimal dependencies
pip install onnxruntime numpy pandas flask joblib scikit-learn

# Navigate to deploy directory
cd code/deploy
```

## Quick Start

### Real-Time Frontend

```bash
# Run with default LSTM model (v3)
python realtime_frontend.py

# Run with specific model
python realtime_frontend.py --model v2    # CNN
python realtime_frontend.py --model v3    # LSTM (default)
python realtime_frontend.py --model v5    # Transformer

# Run on different port
python realtime_frontend.py --model v3 --port 8080
```

Then open http://localhost:5000 in your browser.

### Batch Deployment Pipeline

```bash
python deployment.py --input_csv your_ecg_data.csv \
    --onnx_model sample/ecg_lstm_v3_pytorch_final.onnx \
    --scaler sample/scaler_v3_pytorch.pkl \
    --model_version v3
```

## File Structure

```
code/deploy/
├── realtime_frontend.py          # Real-time web interface (ONNX)
├── deployment.py                  # Batch deployment pipeline (ONNX)
├── classify_heartbeats.py         # Heartbeat classification utility
├── convert_ecg_data.py            # Data conversion utility
├── README.md                      # This file
└── sample/
    ├── 100.csv                    # Sample ECG data
    ├── 100annotations.txt         # Ground truth annotations
    ├── ecg_cnn_v2_pytorch_final.onnx      # v2 CNN model (PyTorch)
    ├── ecg_lstm_v3_pytorch_final.onnx     # v3 LSTM model (PyTorch)
    ├── ecg_transformer_v5_pytorch_final.onnx # v5 Transformer model (PyTorch)
    ├── scaler_v2_pytorch.pkl      # Scaler for v2 (fit on training data only)
    ├── scaler_v3_pytorch.pkl      # Scaler for v3 (fit on training data only)
    └── scaler_v5_pytorch.pkl      # Scaler for v5 (fit on training data only)
```

## Usage Examples

### Starting the Real-Time Frontend

```bash
cd code/deploy
python realtime_frontend.py --model v3
```

Expected output:
```
============================================================
ECG Real-Time Classification Frontend
Using PyTorch ONNX Models
============================================================

Selected model: V3
Loading data and model...
Loading LSTM (v3) model...
Loading ONNX model from: sample/ecg_lstm_v3_pytorch_final.onnx
✓ LSTM (v3) ONNX model loaded successfully
✓ Scaler loaded from: sample/scaler_v3_pytorch.pkl

Loaded 650000 ECG samples
Loaded 2278 annotations

Starting web server on port 5000...
Open your browser and go to: http://localhost:5000

Press Ctrl+C to stop the server
============================================================
```

### Using the Web Interface

1. Open your browser to http://localhost:5000
2. The model name is displayed at the top (e.g., "CNN (v2)")
3. Click "▶ Start" to begin the real-time simulation
4. Watch as the ECG signal scrolls and heartbeats are classified
5. Adjust the speed slider to change simulation speed
6. Click "⏹ Stop" to pause or "🔄 Reset" to restart

## Model Information

### Data Leakage Fix

All models use scalers that were fit **only on training data** (not test data), ensuring realistic performance metrics:
- Split data first (80% train, 10% val, 10% test)
- Fit scaler on training data only
- Transform validation/test using training statistics

### Model Architectures

**v2 - CNN (Convolutional Neural Network)**
- 4 Conv1D layers with BatchNorm
- Dropout 0.5 for regularization
- Input shape: (1, 188)

**v3 - LSTM (Long Short-Term Memory)**
- Bidirectional LSTM (64 units) × 2 layers
- Batch Normalization
- Dropout 0.5
- Input shape: (188, 1)

**v5 - Transformer**
- 3 Transformer encoder layers
- 4-head multi-head attention
- Learnable positional encoding
- Input shape: (188, 1)

## Dependencies

```
onnxruntime>=1.15.0
numpy>=1.20.0
pandas>=1.3.0
flask>=2.0.0
joblib>=1.0.0
scikit-learn>=1.0.0
```

## Troubleshooting

### "No module named 'onnxruntime'"
```bash
pip install onnxruntime
```

### "ONNX model not found"
Ensure the model files exist in the `sample/` directory:
```bash
ls -la code/deploy/sample/*.onnx
```

### Model loading errors
Check that the correct model version is specified:
```bash
python realtime_frontend.py --model v3  # Not "V3" or "lstm"
```

## Performance

| Metric | ONNX Runtime |
|--------|--------------|
| Installation size | ~10 MB |
| Cold start time | <1 second |
| Inference time | ~3-7 ms/beat |
| Memory usage | ~100 MB |

## Changelog

### v3.0 - PyTorch ONNX Models
- Migrated to PyTorch ONNX models (v2/v3/v5)
- Fixed data leakage (scaler fits on training data only)
- Added model selection via command line (--model v2/v3/v5)
- Updated scalers to PyTorch versions
- Display current model in web interface

### v2.0 - ONNX Support
- Added ONNX Runtime support for cross-platform deployment
- Removed hard dependency on TensorFlow/Keras

### v1.0 - Initial Release
- Real-time ECG visualization
- TensorFlow/Keras backend
