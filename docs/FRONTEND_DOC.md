# ECG Frontend Deployment Guide

## Overview

This document explains how the ECG Real-Time Classification Frontend works with different model versions (v2, v3, v5, v6). Each model version has different preprocessing requirements that are automatically handled by the frontend.

## Model Versions Comparison

| Model | Architecture | Beat Length | Classification Type | Context |
|-------|-------------|-------------|---------------------|---------|
| **v2** | CNN (Conv1D) | 188 samples | **Single beat** | None |
| **v3** | LSTM | 188 samples | **Single beat** | None |
| **v5** | Transformer | 188 samples | **Single beat** | None |
| **v6** | Context-Aware CNN1D | 200 samples | **Context window** | 7 beats |

### Key Differences:

- **v2/v3/v5**: Classify each beat **independently** using 188 samples (70 pre-R + 118 post-R)
- **v6**: Classifies the **center beat** using context from 7 beats (200 samples each)

## Architecture

### V2/V3/V5 Architecture (Single Beat Classification)

```
┌─────────────────────────────────────────────────────────────────┐
│              Frontend - v2/v3/v5 (Single Beat)                  │
├─────────────────────────────────────────────────────────────────┤
│  119.csv + 119annotations.txt  ─────►  ECG Signal + R-peaks     │
│              ↓                                                   │
│  R-peak Detection ─────►  Beat Extraction (188 samples)         │
│              ↓                         70 pre-R + 118 post-R    │
│  Single Beat ─────►  Normalize (188) ─────►  Reshape for model  │
│              ↓                                                   │
│  ONNX Inference ─────►  Classification (per beat)              │
│              ↓                                                   │
│  Real-time Visualization                                        │
└─────────────────────────────────────────────────────────────────┘
```

### V6 Architecture (Context-Aware Classification)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend - v6 (Context-Aware)                │
├─────────────────────────────────────────────────────────────────┤
│  119.csv + 119annotations.txt  ─────►  ECG Signal + R-peaks     │
│              ↓                                                   │
│  R-peak Detection ─────►  Beat Extraction (200 samples)         │
│              ↓                         90 pre-R + 110 post-R    │
│  Rolling Buffer (7 beats) ─────►  Context Window                │
│              ↓                                                   │
│  Flatten (1400) ─────►  Normalize ─────►  Reshape (7, 200)      │
│              ↓                                                   │
│  ONNX Inference ─────►  Classification (center beat)            │
│              ↓                                                   │
│  Real-time Visualization                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Source

### Why Record 119 for All Models?

All models (v2, v3, v5, v6) now use MIT-BIH record 119 by default:

```python
# All models use record 119 by default
use_record_119 = True  # Default for v2, v3, v5, and v6
```

Record 119 is used for ALL models because:
1. **Excluded from v6 training**: Never seen by the v6 model during training or validation
2. **True validation**: Represents completely new patient data for fair comparison
3. **Consistent testing**: All models tested on the same data for accurate comparison
4. **No data leakage**: Guarantees realistic performance estimates
5. **Different distribution**: Tests model generalization across all architectures

## Preprocessing Pipeline (Matches Training Exactly)

### Step 1: Beat Extraction

```python
def extract_beat_v6(signal, r_peak_idx):
    """Extract 200-sample beat centered on R-peak.
    
    Matches training: PRE_R_SAMPLES=90, POST_R_SAMPLES=110
    """
    start_idx = r_peak_idx - 90   # 90 samples before R-peak
    end_idx = r_peak_idx + 110    # 110 samples after R-peak
    
    # Handle edge cases with zero padding
    if start_idx < 0:
        beat = np.zeros(200)
        available = signal[:end_idx]
        beat[-len(available):] = available
    elif end_idx > len(signal):
        beat = np.zeros(200)
        available = signal[start_idx:]
        beat[:len(available)] = available
    else:
        beat = signal[start_idx:end_idx]
    
    return beat  # Shape: (200,)
```

### Step 2: Rolling Beat Buffer

```python
# Global buffer for V6 context-aware model
beat_buffer = []  # List of (beat_waveform, beat_type) tuples

def process_beat(beat, beat_type):
    global beat_buffer
    
    # Add new beat to buffer
    beat_buffer.append((beat, beat_type))
    
    # Keep only last 7 beats
    if len(beat_buffer) > 7:
        beat_buffer = beat_buffer[-7:]
    
    # Need 7 beats for inference
    if len(beat_buffer) < 7:
        return {"status": "WAITING", "buffer_size": len(beat_buffer)}
    
    # Ready for inference
    return run_inference()
```

**Why 7 beats?**
- Matches training: 3 previous + 1 center + 3 subsequent
- First 3 beats show "WAITING" status until buffer fills
- After buffer is full, inference runs on every new beat

### Step 3: Normalization (Critical - Must Match Training)

```python
def prepare_input():
    # Stack 7 beats: (7, 200)
    context_beats = np.stack([b for b, _ in beat_buffer], axis=0)
    
    # Flatten for scaler: (1, 1400)
    flat_size = 7 * 200  # = 1400
    context_flat = context_beats.reshape(1, flat_size)
    
    # Normalize using TRAINING scaler
    # This scaler was fitted on X_train only (no data leakage)
    normalized = scaler.transform(context_flat)
    
    # Reshape for model input: (1, 7, 200)
    model_input = normalized.reshape(1, 7, 200).astype(np.float32)
    
    return model_input
```

**Why this exact process?**
1. **Same flattening order**: Row-major (C-order) reshaping matches training
2. **Same scaler**: Loaded from `context_ecg_scaler.pkl`
3. **Same reshape**: Final shape (1, 7, 200) matches model input

### Step 4: ONNX Inference

```python
def run_inference():
    model_input = prepare_input()
    
    # ONNX session inference
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: model_input})[0]
    
    # Output is logits: [logit_normal, logit_abnormal]
    # Apply softmax to get probabilities
    exp_output = np.exp(output - np.max(output))
    proba = exp_output / np.sum(exp_output)
    
    prob_abnormal = proba[0, 1]
    predicted_class = 1 if prob_abnormal >= 0.5 else 0
    
    return {
        "predicted": "ABNORMAL" if predicted_class == 1 else "NORMAL",
        "probability": prob_abnormal,
        "ground_truth": get_ground_truth(beat_buffer[3][1])  # Center beat
    }
```

### Step 5: Ground Truth Comparison

```python
def get_ground_truth(beat_type):
    """N = Normal, anything else = Abnormal"""
    return "NORMAL" if beat_type == 'N' else "ABNORMAL"
```

The center beat (index 3 in the 7-beat window) is used for ground truth because:
- The model predicts the center beat's classification
- Surrounding beats provide context only

## Frontend Features

### Speed Control

```python
# Speed presets (multiplier of real-time)
speeds = [0.1, 0.5, 1, 5, 10]
# 1x = 360 samples/second (real-time MIT-BIH)
# 10x = 3600 samples/second (10x faster playback)
```

### BPM Calculation

```python
def calculateBPM(currentBeatSample):
    beatTimes.append(currentBeatSample)
    
    # Keep last 10 beats for smoothing
    if len(beatTimes) > 10:
        beatTimes.pop(0)
    
    # Average interval (filter outliers: 30-200 BPM range)
    intervals = []
    for i in range(1, len(beatTimes)):
        interval = (beatTimes[i] - beatTimes[i-1]) / 360  # seconds
        if 0.3 < interval < 2.0:  # 30-200 BPM
            intervals.append(interval)
    
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        return round(60 / avg_interval)
    return None
```

**Why filter to 30-200 BPM?**
- Physiologically reasonable range
- Filters out missed beats or double detections
- Smooths display to avoid glitching

### History Navigation

```python
# Navigation controls
scrollHistory(-5)  # Go back 5 seconds
scrollHistory(-1)  # Go back 1 second
goToLive()         # Return to live view
scrollHistory(+1)  # Go forward 1 second (if viewing history)
scrollHistory(+5)  # Go forward 5 seconds
```

### False Detection Logging

```python
if result.ground_truth != result.predicted:
    falseDetections.append({
        "time": result.r_peak / 360,  # Convert to seconds
        "expected": result.ground_truth,
        "got": result.predicted,
        "r_peak": result.r_peak
    })
    updateFalseDetectionList()  # Update UI
```

Clickable false detections allow navigation to that specific time in the signal.

## File Structure

```
code/deploy/
├── realtime_frontend.py     # Main frontend application
├── sample/
│   ├── context_ecg_model.onnx    # V6 ONNX model
│   ├── context_ecg_scaler.pkl    # V6 scaler (trained on training data only)
│   ├── 119.csv                   # ECG signal for V6 testing
│   └── 119annotations.txt        # Annotations for ground truth
└── V6_FRONTEND_DEPLOYMENT.md     # This documentation
```

## Usage

### Command Line

```bash
# Run with V6 model (uses record 119 by default)
python realtime_frontend.py --model v6

# Run with specific port
python realtime_frontend.py --model v6 --port 8080

# Force record 119 for other models
python realtime_frontend.py --model v3 --record-119
```

### Web Interface

1. Open browser to `http://localhost:5000`
2. Click "Start" to begin simulation
3. Use speed controls (0.1x - 10x) to adjust playback
4. Watch real-time classification
5. Click false detections to navigate to those beats
6. Use history navigation to review past beats

## R-Peak Position in Beat Snapshot

```python
# V6 model: R-peak at sample 90 (PRE_R_SAMPLES)
currentRPeakPos = 90  # For visualization marker

# V2/V3/V5 models: R-peak at sample 70
currentRPeakPos = 70
```

The R-peak marker in the beat snapshot is positioned correctly based on the model's preprocessing.

## Expected Performance on Record 119

Based on the test set evaluation (which included record 119-like unseen patients):

| Metric | Expected Value |
|--------|----------------|
| Accuracy | ~69% |
| Recall (Abnormal) | ~55% |
| AUC-ROC | ~0.80 |

**Why lower than V2/V3/V5?**
- V6 uses record-wise split (no patient leakage)
- V2/V3/V5 use the original dataset (ecg.csv) which may have leakage
- V6 metrics are more realistic for real-world deployment

## Troubleshooting

### "WAITING" Status

If the display shows "WAITING" for the first 3 beats:
- This is expected behavior
- The 7-beat buffer needs to fill before inference
- Wait for 7 R-peaks to be detected

### Model Loading Errors

```python
# Check model file exists
ls sample/context_ecg_model.onnx
ls sample/context_ecg_scaler.pkl
```

### Prediction Mismatch

If predictions don't match ground truth:
1. **Expected**: ~55% recall on abnormal beats
2. **Record 119 distribution**: Check the normal/abnormal ratio
3. **Distribution shift**: Record 119 may have different characteristics than training data

## Integration with Thesis

This frontend demonstrates:
1. **Real-time ECG classification**: Simulated live monitoring
2. **Context-aware model**: Uses temporal patterns across beats
3. **No data leakage**: Record 119 never used in training
4. **Proper preprocessing**: Matches training exactly
5. **Clinical utility**: Ground truth comparison and false detection logging
