# V6 Model Training Process

## Overview

This document explains the context-aware CNN1D model training process, including architecture decisions, normalization, anti-overfitting techniques, and why the training stopped at specific epochs.

## Model Architecture

### Input Shape: (7, 200)

The model takes 7 beats of 200 samples each. In PyTorch Conv1d convention, this is:
- **Channels**: 7 (each beat is treated as a channel)
- **Length**: 200 (time samples per beat)

### Architecture Details

```python
class ContextAwareCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv1D Block 1: Extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding=1),   # 7 → 32 channels
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 200 → 100 samples
        )
        
        # Conv1D Block 2: Extract mid-level features
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # 32 → 64 channels
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 100 → 50 samples
        )
        
        # Conv1D Block 3: Extract high-level features
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3), # 64 → 128 channels
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 50 → 25 samples
        )
        
        # Global Average Pooling: Aggregate temporal features
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 25 → 1
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 2 classes: Normal, Abnormal
        )
```

### Why These Kernel Sizes?

| Layer | Kernel Size | Receptive Field | Rationale |
|-------|-------------|-----------------|-----------|
| Conv1 | 3 | ~8ms | Captures sharp QRS complex features |
| Conv2 | 5 | ~14ms | Captures P-wave and T-wave morphology |
| Conv3 | 7 | ~19ms | Captures inter-beat relationships |

Increasing kernel sizes allow progressively larger temporal patterns to be captured at higher abstraction levels.

### Why 32 → 64 → 128 Filters?

This progressive doubling is a common pattern in CNNs:
1. **Lower layers**: Fewer filters capture basic patterns (edges, peaks)
2. **Higher layers**: More filters capture complex combinations
3. **Computational balance**: Pooling reduces spatial dimensions as channels increase

## Normalization (Critical for No Data Leakage)

### The Correct Way

```python
# Step 1: Flatten for scaling
X_train_flat = X_train.reshape(n_train, 7*200)  # (samples, 1400)
X_val_flat = X_val.reshape(n_val, 7*200)
X_test_flat = X_test.reshape(n_test, 7*200)

# Step 2: Fit scaler on TRAINING DATA ONLY
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_flat)   # fit + transform
X_val_norm = scaler.transform(X_val_flat)           # transform only
X_test_norm = scaler.transform(X_test_flat)         # transform only

# Step 3: Reshape back for CNN
X_train_norm = X_train_norm.reshape(n_train, 7, 200)
```

### Why Flatten Before Scaling?

1. **Consistent statistics**: StandardScaler computes mean/std per feature. Flattening ensures consistent statistics across the entire context window.

2. **Deployment compatibility**: The saved scaler expects flattened input (1400 features), matching how the frontend will preprocess data.

### Resulting Statistics

```
Train - Mean: 0.000000, Std: 1.000000  (perfectly normalized)
Val   - Mean: 0.052375, Std: 0.992120  (slightly different distribution)
Test  - Mean: -0.092125, Std: 1.103607 (different patient population)
```

The validation and test sets have slightly different statistics because:
1. They contain different patients with different ECG characteristics
2. **This is expected and correct** - it simulates real-world deployment
3. If they matched exactly (mean≈0, std≈1), it would indicate data leakage

## Training Configuration

### Loss Function: Cross-Entropy with Class Weights

```python
class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
# Output: tensor([0.7002, 1.7488])

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Why class weights?**
- Training data: 53802 Normal vs 21541 Abnormal (71.4% vs 28.6%)
- Without weights, model would favor predicting Normal
- Weight of 1.7488 for Abnormal class compensates for this imbalance

### Optimizer: AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**Why AdamW?**
- Adam with decoupled weight decay (better than L2 regularization)
- Adaptive learning rates per parameter
- Weight decay of 1e-4 prevents overfitting

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',        # Maximize val AUC
    factor=0.5,        # Halve LR when stuck
    patience=5,        # Wait 5 epochs before reducing
    min_lr=1e-6        # Don't go below this
)
```

**Why based on validation AUC?**
- AUC is robust to class imbalance
- Better indicator of model quality than accuracy for binary classification

## Anti-Overfitting Techniques

### 1. Dropout (0.5)

Applied before final classification layer:
```python
nn.Dropout(0.5)
```
Randomly zeroes 50% of features during training, preventing co-adaptation.

### 2. Batch Normalization

Applied after each Conv layer:
```python
nn.BatchNorm1d(32)
```
Normalizes activations, allowing higher learning rates and providing regularization effect.

### 3. Weight Decay (L2 Regularization)

```python
weight_decay=1e-4
```
Penalizes large weights, encouraging simpler solutions.

### 4. Early Stopping

```python
PATIENCE = 15
if val_auc > best_val_auc:
    best_val_auc = val_auc
    patience_counter = 0
    save_model()
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        break  # Stop training
```

### 5. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents exploding gradients, stabilizing training.

## Training Results Analysis

### Why Training Stopped at Epoch 16

```
Epoch [ 1/100] Val AUC: 0.8147 ← Best model saved
Epoch [ 2/100] Val AUC: 0.6995 ← Dropped
...
Epoch [15/100] Val AUC: 0.6788
Epoch [16/100] Early stopping triggered (patience=15 exhausted)
```

**Analysis:**
1. The model achieved best validation AUC (0.8147) at epoch 1
2. Subsequent epochs showed increasing training accuracy but decreasing validation AUC
3. This is classic **overfitting** - the model memorized training patterns instead of learning generalizable features

**Why overfitting happened:**
1. **Record-wise split creates distribution shift**: Different patients have different ECG characteristics
2. **Limited patient diversity**: Only 47 records total, with some having very different characteristics
3. **Model complexity vs data**: 77,314 parameters for patterns that may not generalize across patients

**Why epoch 1 was best:**
- Early stopping correctly identified that the model generalized best with minimal training
- More training led to patient-specific pattern memorization

### Final Metrics (Test Set)

```
Accuracy:  0.6891 (68.91%)
Precision: 0.4007 (40.07%)
Recall:    0.5501 (55.01%)
F1 Score:  0.4636 (46.36%)
AUC-ROC:   0.8060 (80.60%)
```

**Interpretation:**
- **AUC-ROC 0.8060**: Good discriminative ability - the model can distinguish between classes
- **Accuracy 0.69**: Lower than AUC because of class imbalance in test set
- **Recall 0.55**: Model catches 55% of abnormal beats (could be improved)
- **Precision 0.40**: When model predicts abnormal, 40% are actually abnormal

**Why these metrics differ from validation?**
- Test set has different patient population
- Test set: 13863 Normal (78%) vs 3951 Abnormal (22%)
- The model was optimized on validation set, test set is completely unseen

## Comparison with Non-Record-Wise Split

If we had used random beat-wise splitting instead:

| Split Method | Test AUC | Reason |
|--------------|----------|--------|
| Beat-wise (wrong) | ~0.95+ | Patient patterns in both train and test |
| Record-wise (correct) | ~0.80 | Truly unseen patients in test |

The lower record-wise AUC is **more realistic** for real-world deployment.

## ONNX Export

```python
torch.onnx.export(
    model,
    dummy_input,            # (1, 7, 200)
    'context_ecg_model.onnx',
    opset_version=13,       # Wide compatibility
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Output Files

| File | Description |
|------|-------------|
| `context_ecg_model.onnx` | ONNX model for cross-platform inference |
| `context_ecg_scaler.pkl` | StandardScaler fitted on training data |
| `model_config.json` | Configuration and metrics |
| `context_ecg_model.pth` | PyTorch checkpoint for fine-tuning |

## How the AI Model Works

### Inference Pipeline

```
Input: 7 beats × 200 samples (raw ECG values)
    ↓
Flatten: 1400 values
    ↓
Normalize: (x - mean) / std (using training scaler)
    ↓
Reshape: (1, 7, 200)
    ↓
Conv1D layers: Extract features
    ↓
Global Average Pooling: Aggregate
    ↓
Dense layers: Classify
    ↓
Softmax: Probabilities [P(Normal), P(Abnormal)]
    ↓
Output: Class with higher probability
```

### What the Model Learns

1. **Conv1 (kernel=3)**: Sharp transitions in QRS complex
2. **Conv2 (kernel=5)**: P-wave and T-wave shapes
3. **Conv3 (kernel=7)**: Relationships between parts of different beats
4. **Global Pool**: Which features are consistently present across the window
5. **Dense**: Which feature combinations indicate abnormality

## Deployment Considerations

1. **Always use the same scaler**: `context_ecg_scaler.pkl`
2. **Maintain 7-beat buffer**: Wait until 7 consecutive beats are available
3. **Center beat prediction**: The output classifies the 4th beat (center of the window)
4. **Threshold**: Default 0.5, can be adjusted based on sensitivity/specificity needs
