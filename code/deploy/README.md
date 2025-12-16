# ECG Real-Time Classification Frontend - Deployment Guide

## Overview

This is a web-based real-time ECG monitoring and classification system that uses an LSTM neural network model for heartbeat classification. The application has been refactored to support cross-platform deployment without requiring heavy TensorFlow/Keras dependencies.

## Key Features

- ✅ **Cross-platform**: Works on Windows, Linux, and macOS
- ✅ **Lightweight deployment**: Uses ONNX Runtime (10MB) instead of TensorFlow (500MB+)
- ✅ **Real-time visualization**: Interactive ECG signal display
- ✅ **Live classification**: Classifies heartbeats as Normal or Abnormal
- ✅ **Performance metrics**: Tracks accuracy, BPM, and classification statistics

## Architecture

The application supports two runtime modes:

1. **ONNX Runtime Mode** (Recommended) - Keras-free inference
   - Requires: `onnxruntime` (~10MB)
   - Fast, lightweight, cross-platform
   - No TensorFlow/Keras needed

2. **TensorFlow/Keras Mode** (Fallback)
   - Requires: `tensorflow` (~500MB)
   - Used when ONNX model is not available

## Installation

### Option 1: ONNX Runtime Mode (Recommended)

```bash
# Install minimal dependencies
pip install onnxruntime numpy pandas flask joblib scikit-learn

# Run the application
cd code/deploy
python realtime_frontend.py
```

**Note**: You need to convert the H5 model to ONNX first. See [ONNX Conversion Guide](README_ONNX_CONVERSION.md).

### Option 2: TensorFlow/Keras Mode (Fallback)

```bash
# Install TensorFlow and dependencies
pip install tensorflow numpy pandas flask joblib scikit-learn

# Run the application
cd code/deploy
python realtime_frontend.py
```

## File Structure

```
code/deploy/
├── realtime_frontend.py          # Main application (ONNX/Keras support)
├── deployment.py                  # Alternative deployment script
├── classify_heartbeats.py         # Heartbeat classification utility
├── convert_ecg_data.py            # Data conversion utility
├── convert_to_onnx_standalone.py # ONNX conversion script
├── README.md                      # This file
├── README_ONNX_CONVERSION.md     # Detailed ONNX conversion guide
└── sample/
    ├── 100.csv                    # Sample ECG data
    ├── 100annotations.txt         # Ground truth annotations
    ├── ecg_lstm_final.h5          # Keras H5 model (v3 LSTM)
    ├── ecg_lstm_final.onnx        # ONNX model (after conversion)
    ├── ecg_lstm_v3_final.keras    # Alternative Keras format
    └── scaler_v3.pkl              # StandardScaler for normalization
```

## Usage

### Starting the Server

```bash
cd code/deploy
python realtime_frontend.py
```

Expected output:
```
============================================================
ECG Real-Time Classification Frontend
============================================================

Loading data...
Loading ONNX model from: /path/to/ecg_lstm_final.onnx
ONNX model loaded successfully (Keras-free inference)
Loaded 650000 ECG samples
Loaded 2278 annotations

Starting web server...
Open your browser and go to: http://localhost:5000

Press Ctrl+C to stop the server
============================================================
```

### Using the Web Interface

1. Open your browser to http://localhost:5000
2. Click "▶ Start" to begin the real-time simulation
3. Watch as the ECG signal scrolls and heartbeats are classified
4. Adjust the speed slider to change simulation speed
5. Click "⏹ Stop" to pause or "🔄 Reset" to restart

## Model Information

### v3 LSTM Model

The application uses the v3 Bidirectional LSTM model:

- **Input**: 188-sample ECG heartbeat segments
- **Architecture**:
  - Bidirectional LSTM (64 units) × 2 layers
  - Batch Normalization
  - Dropout (0.3)
  - Dense layers (64 → 32 → 2)
- **Output**: Binary classification (Normal vs Abnormal)
- **Preprocessing**: StandardScaler normalization (scaler_v3.pkl)

### Model Files

- **ecg_lstm_final.h5**: Original Keras H5 format
- **ecg_lstm_final.onnx**: ONNX format (convert manually)
- **scaler_v3.pkl**: Preprocessing scaler (required)

## Converting to ONNX

To use the lightweight ONNX Runtime mode, you need to convert the H5 model once:

### Quick Conversion

```bash
cd code/deploy
python convert_to_onnx_standalone.py
```

See [README_ONNX_CONVERSION.md](README_ONNX_CONVERSION.md) for detailed instructions and troubleshooting.

## Dependencies

### Minimal (ONNX Mode)
```
onnxruntime>=1.15.0
numpy>=1.20.0
pandas>=1.3.0
flask>=2.0.0
joblib>=1.0.0
scikit-learn>=1.0.0
```

### Full (Keras Mode)
```
tensorflow>=2.13.0
numpy>=1.20.0
pandas>=1.3.0
flask>=2.0.0
joblib>=1.0.0
scikit-learn>=1.0.0
```

## Platform-Specific Notes

### Windows
- Install Python 3.8 or later
- Use `python` instead of `python3`
- ONNX Runtime works out of the box

### Linux
- Install Python 3.8 or later
- May need to install system dependencies: `sudo apt-get install python3-dev`
- ONNX Runtime works out of the box

### macOS
- Install Python 3.8 or later
- ONNX Runtime supports both Intel and Apple Silicon
- Use `python3` command

## Troubleshooting

### "No module named 'onnxruntime'"
```bash
pip install onnxruntime
```

### "ONNX model not found"
The application will automatically fallback to TensorFlow/Keras if ONNX model is not available. To use ONNX mode, convert the model first:
```bash
python convert_to_onnx_standalone.py
```

### "Error: Neither ONNXRuntime nor TensorFlow is available"
Install one of them:
```bash
pip install onnxruntime  # Recommended
# OR
pip install tensorflow
```

### Model loading is slow
- First time loading with TensorFlow can take 5-10 seconds
- ONNX Runtime loads much faster (<1 second)
- Consider converting to ONNX for better performance

## Performance

### ONNX Runtime vs TensorFlow/Keras

| Metric | ONNX Runtime | TensorFlow/Keras |
|--------|--------------|------------------|
| Installation size | ~10 MB | ~500 MB |
| Cold start time | <1 second | 3-5 seconds |
| Inference time | ~3-7 ms/beat | ~5-10 ms/beat |
| Memory usage | ~100 MB | ~500 MB |
| Cross-platform | ✅ Simple | ⚠️ Complex |

## Development

### Adding New Features

The application architecture makes it easy to:
- Add new visualization types
- Implement different classification models
- Customize the UI/UX
- Export classification results

### Testing

Test the application with different scenarios:
```bash
# Test with ONNX model
python realtime_frontend.py

# Test imports
python -c "from realtime_frontend import load_data; print('OK')"

# Check model format
ls -lh sample/*.onnx sample/*.h5
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ecg_realtime_frontend,
  title={ECG Real-Time Classification Frontend},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/aimatochysia/thesis}}
}
```

## License

See the repository LICENSE file for details.

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [README_ONNX_CONVERSION.md](README_ONNX_CONVERSION.md)
3. Open an issue on GitHub

## Changelog

### v2.0 - ONNX Support
- Added ONNX Runtime support for cross-platform deployment
- Removed hard dependency on TensorFlow/Keras
- Added automatic fallback to Keras if ONNX not available
- Created comprehensive conversion documentation
- Improved startup time and reduced memory footprint

### v1.0 - Initial Release
- Real-time ECG visualization
- LSTM-based heartbeat classification
- Web-based interface with Flask
- TensorFlow/Keras backend
