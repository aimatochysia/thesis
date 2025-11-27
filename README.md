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
