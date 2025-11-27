# ECG Arrhythmia Classification

This repository contains multiple approaches for ECG (Electrocardiogram) binary classification to detect arrhythmias (normal vs abnormal heartbeats).

## Dataset

- **Input**: 188 columns of ECG time series data representing a single heartbeat
- **Output**: Binary classification (0 = normal, 1 = abnormal/arrhythmia)
- **Baseline**: ECG signal baseline is around 958, with peaks (up and down) during heartbeats

## Version History

### v0 - Initial Baseline Model

**File**: `ecg-train-last.ipynb`, `ecg-train-last.py`

The original baseline implementation using a simple CNN architecture:

- **Data**: Single ECG dataset from Kaggle
- **Architecture**: Basic 2-layer Conv1D network
  - Conv1D(64) → MaxPooling → Dropout(0.3)
  - Conv1D(128) → MaxPooling → Dropout(0.3)
  - Flatten → Dense(128) → Dense(output)
- **Optimizer**: Adam with default learning rate
- **Features**: Basic data preprocessing, confusion matrix evaluation

### v1 - Extended Dataset

**File**: `ecg-train-last-mod1.ipynb`

Extended version with combined datasets:

- **Data**: Combined two ECG datasets (ecg.csv + ecg3.csv) for more training samples
- **Architecture**: Same as v0 (2-layer Conv1D)
- **Improvements**: 
  - Data concatenation from multiple sources
  - Column alignment fixes for dataset compatibility
  - Increased training data volume

### v2 - Best CNN Model (Optimized)

**File**: `ecg-train-best-cnn.ipynb`

Significantly improved CNN architecture with modern deep learning techniques:

#### Justification for Improvements

| Aspect | v0/v1 | v2 (Improved) | Why It Matters |
|--------|-------|---------------|----------------|
| **Depth** | 2 Conv layers | 4 Conv blocks (8 Conv layers) | Deeper networks can learn more complex hierarchical features from ECG signals |
| **Normalization** | None | BatchNormalization after each Conv | Stabilizes training, allows higher learning rates, reduces internal covariate shift |
| **Pooling** | Flatten | GlobalAveragePooling1D | Reduces parameters, prevents overfitting, more robust to spatial translations |
| **Filters** | 64→128 | 32→64→128→256 | Gradual increase allows learning from simple to complex patterns |
| **Padding** | Valid (default) | Same | Preserves spatial dimensions, important for capturing edge features in ECG |
| **Callbacks** | ModelCheckpoint only | EarlyStopping + ReduceLROnPlateau + ModelCheckpoint | Prevents overfitting, adaptive learning rate for better convergence |
| **Class Weights** | None | compute_class_weight('balanced') | Handles imbalanced datasets common in medical data |
| **Train/Test Split** | Random | Stratified | Ensures both classes are proportionally represented |

#### Architecture Details
```
Block 1: Conv1D(32)×2 + BatchNorm + MaxPool + Dropout(0.2)
Block 2: Conv1D(64)×2 + BatchNorm + MaxPool + Dropout(0.2)
Block 3: Conv1D(128)×2 + BatchNorm + MaxPool + Dropout(0.3)
Block 4: Conv1D(256) + BatchNorm + GlobalAvgPool
Dense: 128 → 64 → Output (with BatchNorm and Dropout)
```

### v3 - LSTM (Recurrent Neural Network)

**File**: `ecg-train-lstm.ipynb`

#### Why LSTM for ECG?

ECG signals are **temporal sequences** where the order of data points matters. LSTM (Long Short-Term Memory) networks are specifically designed for sequential data:

| Feature | CNN (v2) | LSTM (v3) |
|---------|----------|-----------|
| **Temporal Modeling** | Local patterns via convolution | Long-range temporal dependencies |
| **Memory** | No memory of past inputs | Maintains hidden state across sequence |
| **Direction** | Forward only | Bidirectional (forward + backward) |
| **Best For** | Local feature extraction | Sequence patterns, rhythm analysis |

#### Architecture
- **Bidirectional LSTM**: Processes sequence in both directions to capture context from past and future
- Bidirectional(LSTM(64)) → BatchNorm → Dropout(0.3)
- Bidirectional(LSTM(32)) → BatchNorm → Dropout(0.3)
- Dense(64) → Dense(32) → Output
- **Preprocessing**: StandardScaler normalization (important for RNNs)

### v4 - Ensemble Machine Learning

**File**: `ecg-train-ensemble.ipynb`

#### Why Traditional ML Ensemble?

Traditional ML methods can be highly effective with proper **feature engineering**:

| Aspect | Deep Learning (v2, v3, v5) | Ensemble ML (v4) |
|--------|---------------------------|------------------|
| **Features** | Learned automatically | Hand-crafted domain features |
| **Interpretability** | Black box | Feature importance visualization |
| **Training Time** | GPU required, slower | CPU, faster |
| **Data Requirements** | Needs more data | Works well with smaller datasets |

#### Feature Engineering
Extracted 17+ statistical features from each ECG signal:
- **Basic Statistics**: mean, std, max, min, range, median
- **Higher-Order**: skewness, kurtosis
- **Signal Properties**: RMS, variance, max_diff, mean_abs_diff
- **Percentiles**: p25, p75, IQR
- **Temporal**: zero_crossings (around baseline)
- **Energy**: sum of squared values

#### Models
1. **Random Forest**: 200 trees, max_depth=20, class_weight='balanced'
2. **XGBoost/GradientBoosting**: 200 estimators, learning_rate=0.1
3. **Voting Ensemble**: Soft voting combining both models

### v5 - Transformer with Self-Attention

**File**: `ecg-train-transformer.ipynb`

#### Why Transformers for ECG?

Transformers use **self-attention** to capture relationships between any two points in the sequence, regardless of distance:

| Feature | CNN | LSTM | Transformer |
|---------|-----|------|-------------|
| **Long-Range Dependencies** | Limited by kernel size | Gradual decay | Direct attention to any position |
| **Parallelization** | High | Low (sequential) | High |
| **Interpretability** | Low | Low | Attention weights visualizable |
| **Global Context** | Multiple layers needed | Memory bottleneck | Single layer captures all |

#### Why It Matters for ECG
- **P-wave to QRS relationship**: Transformer can directly attend to P-wave features when analyzing QRS complex
- **Rhythm abnormalities**: Can detect irregular patterns spanning the entire heartbeat
- **Multi-scale patterns**: Self-attention captures both local and global features simultaneously

#### Architecture
- Custom `PositionalEncoding` layer (learnable embeddings)
- 3 Transformer encoder blocks with Multi-Head Attention (4 heads)
- GlobalAveragePooling1D
- MLP head: Dense(128) → Dense(64) → Output

```python
class PositionalEncoding(tf.keras.layers.Layer):
    """Learnable positional encoding for ECG sequences."""
    def __init__(self, sequence_length, d_model):
        self.position_embedding = trainable_weight(shape=(sequence_length, d_model))
    
    def call(self, x):
        return x + self.position_embedding
```

## Comparison Summary

| Version | Model Type | Key Innovation | Best For |
|---------|------------|----------------|----------|
| v0 | Basic CNN | Baseline | Quick prototyping |
| v1 | Basic CNN | Extended data | More training samples |
| v2 | Deep CNN | BatchNorm, GlobalAvgPool, Callbacks | General purpose, best CNN |
| v3 | Bi-LSTM | Temporal sequence modeling | Rhythm-based detection |
| v4 | RF + XGBoost | Feature engineering, interpretability | Explainable predictions |
| v5 | Transformer | Self-attention, long-range dependencies | Complex pattern detection |

## Evaluation Metrics

All models include:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, Recall, F1-score per class
- **ROC-AUC Score**: Area under the ROC curve

## Usage

Each notebook can be run independently. Adjust the data path based on your environment:

```python
# For Kaggle
df = pd.read_csv('/kaggle/input/ecg-dataset/ecg.csv')

# For local
df = pd.read_csv('../../dataset_aritmia_NEW.csv')
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas, numpy, matplotlib
- XGBoost (optional, for v4)

---

## Training Pipeline and Deployment

This repository includes a complete, reproducible pipeline for training and deployment, with a focus on:
1. **Misalignment-robust training** for the v2 CNN teacher model
2. **Knowledge distillation** into a tiny student model suitable for low-end deployment
3. **Streaming segmentation pipeline** for real-time beat detection and classification

### Directory Structure

```
code/
├── data/                    # Data preprocessing utilities
│   ├── __init__.py
│   └── preprocessing.py     # Beat segmentation, normalization, patient-wise splits
├── training/                # Training scripts
│   ├── __init__.py
│   ├── train_teacher_v2_robust.py    # Robust teacher training with augmentations
│   └── train_student_distill.py      # Knowledge distillation for tiny student
├── eval/                    # Evaluation scripts
│   ├── __init__.py
│   └── evaluate_robustness.py        # Robustness curves and metrics
├── deployment/              # Deployment utilities
│   ├── deploy.py            # Streaming ECG pipeline
│   └── export_tflite.py     # TFLite INT8 conversion
├── deploy/                  # Legacy deployment script
│   └── deployment.py
└── v0-v5/                   # Original model notebooks
outputs/
├── models/                  # Saved model files
└── plots/                   # Generated plots and metrics
```

### Data Preprocessing

The preprocessing module (`code/data/preprocessing.py`) provides:

- **RR-adaptive windowing**: pre_frac=0.35, post_frac=0.65 with clamps:
  - Pre: 0.08–0.35s
  - Post: 0.16–0.60s
- **Resampling** to 188 samples
- **Normalization**: baseline_shift_scale (baseline ~950, scale=100)
- **Patient-wise splits** to prevent data leakage
- Support for CSV and PhysioNet/MIT-BIH records

```python
from code.data import ECGDataLoader, load_csv_data, patient_wise_split

# Load pre-segmented beats
loader = ECGDataLoader(normalize=True, norm_mode="baseline_shift_scale")
X, y = loader.load_csv_beats(["ecg.csv", "ecg3.csv"])

# Prepare for training
data = loader.prepare_for_training(X, y, test_size=0.2, val_size=0.1)
```

### Training the Robust Teacher Model

Train the v2 CNN with misalignment-robust augmentations:

```bash
python code/training/train_teacher_v2_robust.py \
    --data_path ecg.csv \
    --data_path2 ecg3.csv \
    --output_dir outputs/models \
    --epochs 200 \
    --batch_size 32 \
    --consistency_weight 0.1
```

**Augmentations applied:**
- Random temporal shift (±10–40 ms)
- Mild time-warp (95–105%)
- Small amplitude scaling (95–105%) and noise

**Output:**
- `outputs/models/teacher_v2_robust.h5` - Trained model
- `outputs/plots/robustness_curve_teacher.png` - Performance vs shift
- `outputs/plots/robustness_metrics_teacher.csv` - Metrics table

### Training the Distilled Student Model

Train a compact student model (~<100k params) using knowledge distillation:

```bash
python code/training/train_student_distill.py \
    --teacher_path outputs/models/teacher_v2_robust.h5 \
    --data_path ecg.csv \
    --output_dir outputs/models \
    --temperature 3.0 \
    --alpha 0.7 \
    --use_consistency
```

**Student architecture:**
- Depthwise separable convolutions (3-4 blocks)
- BatchNorm + ReLU
- GlobalAvgPool
- Small dense head (~40k params)

**Distillation loss:**
- KL divergence to teacher's soft outputs (temperature T=3)
- Cross-entropy with hard labels
- Weight: α=0.7 (distillation) / 0.3 (hard labels)

**Output:**
- `outputs/models/student_distilled.h5` - Distilled student
- `outputs/models/baseline_tiny.h5` - Non-distilled baseline
- `outputs/plots/model_comparison.csv` - F1, AUC, accuracy, params, speed
- `outputs/plots/robustness_comparison.png` - Teacher vs Student vs Baseline

### Streaming Deployment

Run the deployment pipeline on continuous ECG recordings:

```bash
python code/deployment/deploy.py \
    --input_csv long_recording.csv \
    --keras_h5 outputs/models/student_distilled.h5 \
    --output_csv outputs/per_beat_predictions.csv \
    --plots_dir outputs/plots \
    --fs 360
```

**Features:**
- Pan–Tompkins-like R-peak detection with adaptive thresholding
- Refractory period and search-back
- RR-adaptive beat segmentation
- Resampling to 188 samples
- Per-beat classification

**Output:**
- `outputs/per_beat_predictions.csv` - Timestamps, window bounds, prob, label
- `outputs/plots/continuous_with_beats.png` - Signal with R-peaks and windows
- `outputs/plots/beats_grid.png` - Grid of beats with predictions

### Robustness Evaluation

Generate robustness curves and comparison tables:

```bash
python code/eval/evaluate_robustness.py \
    --model_path outputs/models/teacher_v2_robust.h5,outputs/models/student_distilled.h5 \
    --model_names "Teacher,Student" \
    --data_path ecg.csv \
    --output_dir outputs/plots
```

**Output:**
- Robustness curves (accuracy/AUC/F1 vs temporal shift)
- Confusion matrices at different shifts
- ROC curves
- Model comparison table

### TFLite INT8 Export (Optional)

Convert models to TFLite for edge deployment:

```bash
python code/deployment/export_tflite.py \
    --model_path outputs/models/student_distilled.h5 \
    --data_path ecg.csv \
    --quantize int8 \
    --compare
```

**Output:**
- `outputs/models/student_distilled_int8.tflite` - Quantized model
- Size and latency comparison with original Keras model

### Expected Results

| Metric | Teacher | Student (Distilled) | Baseline (Tiny) |
|--------|---------|---------------------|-----------------|
| Parameters | ~500k | ~40k | ~40k |
| Accuracy (0ms) | ~0.98 | ~0.97 | ~0.95 |
| AUC | ~0.99 | ~0.98 | ~0.96 |
| F1 (Abnormal) | ~0.97 | ~0.96 | ~0.93 |
| Inference Time | ~10ms | ~3ms | ~3ms |
| Max Acc Drop (±40ms) | ~2% | ~3% | ~5% |

The distilled student achieves comparable performance to the teacher while being significantly smaller and faster, with better robustness than a non-distilled baseline.
