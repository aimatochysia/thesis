# ECG Real-Time Backend Architecture

This document describes the refactored backend architecture for the ECG arrhythmia detection system, designed for thesis-quality deployment.

## Overview

The application has been refactored into a modular frontend/backend architecture with clear separation of concerns:

```
code/deploy/
├── frontend/                    # Node.js Express frontend
│   ├── package.json            # Dependencies
│   ├── src/
│   │   └── server.js           # Express server with API proxy
│   └── public/                 # Static assets
│       ├── index.html          # Main HTML page
│       ├── css/styles.css      # Stylesheet
│       └── js/                 # JavaScript modules
│           ├── api.js          # API client
│           ├── ecgRenderer.js  # ECG canvas renderer
│           ├── beatRenderer.js # Beat snapshot renderer
│           └── app.js          # Main application
│
├── backend/                     # Python modules
│   ├── __init__.py            # Package exports
│   ├── ecg_streamer.py        # Signal loading and streaming
│   ├── inference_engine.py    # ONNX model inference
│   └── evaluation_layer.py    # Performance evaluation
│
├── app.py                       # Flask API server
└── sample/                      # Model files and test data
```

## Module Responsibilities

### 1. ECG Streamer (`ecg_streamer.py`)

**Purpose:** Load and stream ECG signal data with simulated real-time playback.

**Responsibilities:**
- Loading ECG CSV files (MIT-BIH format)
- Parsing annotation files
- Simulating real-time signal streaming at configurable speeds
- Maintaining playback state (current position, speed)

**Key Features:**
- Thread-safe state management
- Configurable playback speed (0.1x to 100x)
- Window-based data retrieval for efficient streaming
- Time-based position updates for deterministic replay

**API:**
```python
streamer = ECGStreamer(signal_path, annotation_path)
streamer.start()                              # Start playback
streamer.stop()                               # Stop playback
streamer.set_speed(2.0)                       # 2x speed
window = streamer.get_window(window_samples)  # Get signal window
position = streamer.get_current_position()    # Get playback state
```

### 2. Inference Engine (`inference_engine.py`)

**Purpose:** Load ONNX models and perform beat classification.

**Responsibilities:**
- Loading ONNX model once at startup
- Loading scaler (fitted on training data only)
- Preprocessing beats to match training pipeline exactly
- Running inference on individual beats or context windows
- Managing rolling beat buffer for context-aware models (v6)

**Key Features:**
- Support for multiple model versions (v2, v3, v5, v6)
- Preprocessing matches training exactly:
  - v2/v3/v5: 188 samples (70 pre + 118 post R-peak), single beat
  - v6: 200 samples (90 pre + 110 post R-peak), 7-beat context window
- Softmax application for probability output

**API:**
```python
engine = InferenceEngine(model_version='v6', models_dir='sample/')
result = engine.classify_beat(signal, r_peak_idx)
# Returns: {predicted, probability, beat_waveform, ...}

model_info = engine.get_model_info()
# Returns: {name, version, beat_length, context_aware, ...}
```

### 3. Evaluation Layer (`evaluation_layer.py`)

**Purpose:** Compare predictions with ground truth and compute metrics.

**Responsibilities:**
- Matching predicted beats with annotated beats
- Computing classification metrics (accuracy, sensitivity, specificity, etc.)
- Tracking false detections for debugging
- Performance statistics over time
- BPM calculation from beat intervals

**Key Features:**
- Complete confusion matrix calculation
- F1 score, precision, recall metrics
- Historical result tracking with configurable max size
- Export capability to pandas DataFrame

**API:**
```python
evaluator = EvaluationLayer()
result = evaluator.add_result(r_peak, beat_type, predicted, probability)
metrics = evaluator.get_metrics()
# Returns: {accuracy, sensitivity, specificity, precision, f1_score, ...}

false_detections = evaluator.get_false_detections(count=50)
```

## REST API Endpoints

### ECG Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ecg/stream` | GET | Get ECG window with samples and timestamps |
| `/ecg/data` | GET | Get full signal and annotations (initial load) |
| `/ecg/annotations` | GET | Get annotations in a sample range |

**GET /ecg/stream Parameters:**
- `window_seconds`: Window duration in seconds (default: 5.0)
- `end_sample`: End sample index (default: current position)

**Response:**
```json
{
  "samples": [940, 942, 945, ...],
  "timestamps": [0.0, 0.00278, 0.00556, ...],
  "start_index": 0,
  "end_index": 1800,
  "sampling_rate": 360
}
```

### Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ecg/infer` | POST | Classify a beat at given R-peak position |

**POST /ecg/infer Body:**
```json
{
  "r_peak": 1234,
  "beat_type": "N"  // Optional: ground truth for evaluation
}
```

**Response:**
```json
{
  "r_peak": 1234,
  "predicted": "NORMAL",
  "probability": 0.0234,
  "ground_truth": "NORMAL",
  "correct": true,
  "beat_waveform": [940, 942, ...],
  "r_peak_pos_in_beat": 90,
  "beat_length": 200,
  "context_aware": true,
  "buffer_size": 7
}
```

### Status and Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ecg/status` | GET | Get current system status |
| `/ecg/control` | POST | Control playback (start/stop/reset/speed) |
| `/ecg/results` | GET | Get classification results and false detections |

**GET /ecg/status Response:**
```json
{
  "simulation": {
    "running": true,
    "current_sample": 5000,
    "current_time": 13.889,
    "progress": 0.035,
    "speed": 1.0
  },
  "model": {
    "version": "v6",
    "name": "Context-Aware CNN1D (v6)",
    "beat_length": 200,
    "context_aware": true,
    "context_window_size": 7
  },
  "metrics": {
    "total": 42,
    "accuracy": 94.5,
    "sensitivity": 92.3,
    "specificity": 96.1,
    "bpm": 72
  },
  "signal": {
    "total_samples": 142000,
    "total_duration": 394.4,
    "sampling_rate": 360
  }
}
```

**POST /ecg/control Body:**
```json
{
  "action": "start"  // or "stop", "reset", "set_speed"
  // For set_speed:
  "speed": 2.0
}
```

## Timing Model

The backend uses **server-side timing** for deterministic replay:

1. When playback starts, the server records start time and start sample
2. On each data request, elapsed time is calculated: `elapsed = now - start_time`
3. Current sample is computed: `sample = start_sample + elapsed * sampling_rate * speed`
4. This ensures consistent behavior regardless of frontend timing

**Benefits:**
- Deterministic replay across sessions
- Reproducible for thesis documentation
- Independent of network latency

## Usage

### Basic Usage

```bash
# Start with default settings (v3 LSTM, record 119, port 5000)
python app.py

# Use v6 Context-Aware model
python app.py --model v6

# Custom port for VPS deployment
python app.py --port 8080 --host 0.0.0.0

# Use different test record
python app.py --record 100
```

### VPS Deployment

For VPS deployment, use a production WSGI server:

```bash
# Using gunicorn (recommended)
gunicorn -w 4 -b 0.0.0.0:8080 app:app

# Using waitress (Windows compatible)
waitress-serve --host 0.0.0.0 --port 8080 app:app
```

## Backward Compatibility

Legacy endpoints are maintained for backward compatibility:

- `/api/data` → `/ecg/data`
- `/api/classify` → `/ecg/infer`
- `/api/model_info` → `/ecg/status` (model section)

The old `realtime_frontend.py` still works and uses the `/api/*` endpoints.

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **No Data Leakage**: Annotations are never exposed to inference engine
3. **Deterministic Replay**: Server controls timing, not frontend
4. **Thesis Quality**: Clean code, comprehensive documentation, proper error handling
5. **VPS Ready**: Designed for persistent backend deployment
