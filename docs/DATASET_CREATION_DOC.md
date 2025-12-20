# V6 Dataset Creation Process

## Overview

This document describes the complete process for creating a context-aware ECG beat dataset from the MIT-BIH Arrhythmia Database. The dataset is specifically designed for subject-independent (record-wise) training and evaluation to ensure no patient leakage between training, validation, and test sets.

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BEAT_LENGTH` | 200 samples | Chosen to capture the complete PQRST complex of an ECG beat. At 360 Hz sampling rate, this represents ~556ms which is sufficient to capture both the P-wave onset and T-wave offset for most heart rates. |
| `PRE_R_SAMPLES` | 90 samples | Provides ~250ms before the R-peak to capture the P-wave and PR interval. The P-wave typically starts 120-200ms before the R-peak. |
| `POST_R_SAMPLES` | 110 samples | Provides ~306ms after the R-peak to capture the ST segment and T-wave. The T-wave typically ends 200-300ms after the R-peak. |
| `CONTEXT_WINDOW_SIZE` | 7 beats | Uses 3 previous beats + 1 center beat + 3 subsequent beats. This provides temporal context that helps the model learn patterns that span multiple heartbeats. |
| `SAMPLING_RATE` | 360 Hz | Standard MIT-BIH database sampling rate. |
| `EXCLUDED_RECORDS` | ['119'] | Reserved strictly for live testing/validation. Never used during training or validation. |

## Why These Numbers?

### Beat Length (200 samples)

The beat length of 200 samples was chosen because:

1. **Complete PQRST capture**: At 360 Hz, 200 samples = 556ms. A typical ECG beat has:
   - P-wave: 80-120ms duration
   - QRS complex: 80-120ms duration
   - T-wave: 120-200ms duration
   - Total: ~300-440ms minimum

2. **Buffer for variability**: The extra samples provide margin for:
   - Heart rate variability (faster/slower beats)
   - Individual anatomical differences
   - Slight misalignment in R-peak detection

3. **CNN compatibility**: 200 is divisible by common pooling factors (2, 4, 5, 8, 10), making it efficient for MaxPool layers.

### Context Window Size (7 beats)

The 7-beat context window (3 previous + 1 center + 3 subsequent) was chosen because:

1. **Arrhythmia patterns**: Many arrhythmias are characterized by patterns across multiple beats:
   - Premature Ventricular Contractions (PVCs) often occur in bigeminy (every other beat) or trigeminy (every third beat)
   - Atrial fibrillation shows irregularly irregular rhythms across multiple beats
   - Heart blocks affect the relationship between consecutive beats

2. **Computational efficiency**: 7 beats balance context richness with computational cost:
   - Too few beats (3-5): Miss longer-term patterns
   - Too many beats (9+): Increased computational cost with diminishing returns
   - 7 beats: Captures most clinically relevant multi-beat patterns

3. **Symmetric padding**: 3 beats before + 3 beats after provides balanced temporal context.

### Pre-R and Post-R Sample Distribution (90/110)

The asymmetric split (90 before, 110 after R-peak) was chosen because:

1. **R-peak position**: The R-peak is the most reliable fiducial point for beat detection, but it's not centered in the beat morphology.

2. **Post-R importance**: The ST segment and T-wave (after R-peak) are critical for detecting:
   - ST elevation/depression (ischemia, infarction)
   - T-wave abnormalities (electrolyte imbalances, ischemia)
   - These require more samples after the R-peak.

3. **Pre-R components**: The P-wave and PR interval (before R-peak) are important but typically shorter duration than post-R components.

## Processing Pipeline

### Step 1: Data Loading

```python
# Load signal from CSV (MLII lead)
signal_df = pd.read_csv(signal_file)
signal = signal_df['MLII'].values
```

The MLII (Modified Lead II) is used because it provides excellent visualization of P-waves and has consistent polarity across patients.

### Step 2: R-Peak Detection (Pan-Tompkins Algorithm)

The Pan-Tompkins algorithm is used for R-peak detection because:

1. **Proven reliability**: Industry standard for QRS detection since 1985
2. **Noise tolerance**: The bandpass filter (5-15 Hz) effectively removes:
   - Baseline wander (<1 Hz)
   - Muscle noise (>35 Hz)
   - Power line interference (50/60 Hz)
3. **Adaptive thresholding**: Handles varying signal amplitudes

```python
def detect_r_peaks_pan_tompkins(signal, fs=360):
    # Bandpass filter (5-15 Hz)
    filtered = bandpass_filter(signal, 5.0, 15.0, fs)
    # Differentiate
    diff = np.diff(filtered)
    # Square
    squared = diff ** 2
    # Moving window integration (150ms window)
    integrated = np.convolve(squared, np.ones(window_size)/window_size)
    # Find peaks with adaptive threshold
    peaks = find_peaks(integrated, height=threshold, distance=min_distance)
    return peaks
```

### Step 3: Beat Extraction

```python
def extract_beat(signal, r_peak_idx):
    start_idx = r_peak_idx - PRE_R_SAMPLES  # 90 samples before
    end_idx = r_peak_idx + POST_R_SAMPLES   # 110 samples after
    
    # Handle edge cases with zero padding
    if start_idx < 0 or end_idx > len(signal):
        beat = np.zeros(BEAT_LENGTH)
        # Fill available samples
    else:
        beat = signal[start_idx:end_idx]
    
    return beat
```

### Step 4: Labeling

Binary classification with clear clinical rationale:

| Label | Annotation | Clinical Meaning |
|-------|------------|------------------|
| 0 (Normal) | 'N' | Normal sinus rhythm beat |
| 1 (Abnormal) | Any other | Includes PVCs, PACs, blocks, etc. |

This binary classification approach simplifies the problem while remaining clinically relevant. Most clinical applications need to first distinguish normal from abnormal before further classification.

### Step 5: Context Window Construction

```python
def create_context_windows(beats_data, window_size=7):
    half_window = window_size // 2  # = 3
    
    for center_idx in range(half_window, len(beats_data) - half_window):
        # Get beats: [center-3, center-2, center-1, center, center+1, center+2, center+3]
        window_beats = beats_data[center_idx - half_window : center_idx + half_window + 1]
        
        # Stack into (7, 200) array
        context_window = np.stack([beat for beat, _, _ in window_beats])
        
        # Use CENTER beat's label for the entire window
        label = beats_data[center_idx][1]
        
        yield context_window, label
```

**Why center beat label?** The model predicts the classification of the center beat using context from surrounding beats. This mimics how cardiologists analyze ECGs—they consider the rhythm context when classifying individual beats.

### Step 6: Record-Wise Split (Critical for No Patient Leakage)

```python
# Split RECORDS, not beats
n_records = len(all_records)
n_train = int(0.7 * n_records)
n_val = int(0.15 * n_records)
n_test = n_records - n_train - n_val

train_records = all_records[:n_train]
val_records = all_records[n_train:n_train + n_val]
test_records = all_records[n_train + n_val:]
```

**Why record-wise split?**

1. **Patient independence**: Each MIT-BIH record is from a different patient. Random beat-wise splitting would leak patient-specific patterns into test set.

2. **Realistic evaluation**: In real deployment, the model sees completely new patients. Record-wise split simulates this.

3. **Clinical validity**: This approach is required for FDA and regulatory submissions.

## Output Files

| File | Description |
|------|-------------|
| `X_train.npy` | Training features: (75343, 7, 200) |
| `y_train.npy` | Training labels: (75343,) |
| `X_val.npy` | Validation features: (14745, 7, 200) |
| `y_val.npy` | Validation labels: (14745,) |
| `X_test.npy` | Test features: (17814, 7, 200) |
| `y_test.npy` | Test labels: (17814,) |
| `record_split.json` | Which records in train/val/test |
| `dataset_config.json` | All configuration parameters |

## Class Distribution (Why Imbalanced is OK)

```
Train: Normal=53802, Abnormal=21541
Val:   Normal=5650,  Abnormal=9095
Test:  Normal=13863, Abnormal=3951
```

The imbalanced distribution is acceptable because:

1. **Reflects reality**: ECG data is naturally imbalanced—most heartbeats are normal.

2. **Class weighting**: During training, class weights compensate for imbalance:
   ```python
   class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
   criterion = CrossEntropyLoss(weight=class_weights)
   ```

3. **Evaluation metrics**: Use AUC-ROC and recall (not just accuracy) to evaluate minority class performance.

## Usage

```bash
# Run on Kaggle
# 1. Upload notebook to Kaggle
# 2. Add MIT-BIH Arrhythmia Database as input
# 3. Run all cells
# 4. Download output files
```

## Next Steps

After dataset creation, proceed to:
1. Run `mitbih_context_cnn1d_training.ipynb` for model training
2. Download ONNX and PKL files for deployment
3. Test with record 119 (excluded from training)
