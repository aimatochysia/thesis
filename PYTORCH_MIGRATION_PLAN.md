# PyTorch Migration Plan

## Issue Identified

**Critical Data Leakage**: All three notebooks (v2, v3, v5) fit the StandardScaler on the entire dataset before splitting into train/val/test sets. This causes the validation and test sets to "leak" information into the training process, resulting in artificially inflated accuracy (99-100%).

```python
# CURRENT (WRONG):
X_normalized = scaler.fit_transform(X)  # Fits on ALL data
X_train, X_test = train_test_split(X_normalized, ...)  # Then splits

# CORRECT:
X_train, X_test = train_test_split(X, ...)  # Split first
X_train = scaler.fit_transform(X_train)  # Fit only on training
X_test = scaler.transform(X_test)  # Transform using training stats
```

## User Request

1. Convert training from TensorFlow/Keras to PyTorch
2. Export models to ONNX format
3. Fix overfitting issues

## Migration Scope

### Files to Convert
- `code/v2/ecg-train-last-v2.ipynb` - CNN model
- `code/v3/ecg-train-last-v3.ipynb` - LSTM model  
- `code/v5/ecg-train-last-v5.ipynb` - Transformer model

### Changes Required (Per Notebook)

#### 1. Imports (~10-15 lines changed)
```python
# OLD (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# NEW (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

#### 2. Data Loading (~20-30 lines changed)
- Create PyTorch Dataset class
- Implement DataLoader for batching
- **FIX DATA LEAKAGE**: Fit scaler only on training data

```python
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

#### 3. Model Architecture (~50-100 lines changed)
- Convert Sequential/Functional API to `nn.Module`
- Rewrite forward pass
- Different initialization strategies

**v2 - CNN Model**:
```python
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)  # Increased from 0.3
        # ... rest of architecture
```

**v3 - LSTM Model**:
```python
class ECG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
```

**v5 - Transformer Model**:
```python
class ECG_Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, num_classes=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=0.5),
            num_layers
        )
```

#### 4. Training Loop (~100-150 lines changed)
- Replace `model.fit()` with custom training loop
- Implement manual epoch iteration
- Add validation during training
- Implement early stopping manually

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), correct / total
```

#### 5. Callbacks/Regularization (~50 lines)
- Manual early stopping
- Learning rate scheduling
- **Stronger regularization** to fix overfitting:
  - Increase dropout from 0.3 to 0.5-0.6
  - Add L2 regularization (weight_decay)
  - Reduce model capacity if needed
  - Add label smoothing

```python
# Early stopping
patience = 10  # Reduced from 15
best_val_loss = float('inf')
counter = 0

# Optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

#### 6. ONNX Export (~20-30 lines)
- Direct PyTorch to ONNX export (simpler than TensorFlow)
- No tf2onnx needed

```python
# Export to ONNX
dummy_input = torch.randn(1, 188, 1)
torch.onnx.export(
    model,
    dummy_input,
    'ecg_lstm_v3_pytorch.onnx',
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Estimated Effort

| Notebook | Lines Changed | Time Estimate |
|----------|---------------|---------------|
| v2 (CNN) | ~400-500 | 3-4 hours |
| v3 (LSTM) | ~400-500 | 3-4 hours |
| v5 (Transformer) | ~500-600 | 4-5 hours |
| Testing/Validation | - | 2-3 hours |
| **Total** | **~1500 lines** | **12-16 hours** |

## Benefits of PyTorch Approach

1. **Cleaner ONNX export**: Direct export, no tf2onnx compatibility issues
2. **More control**: Explicit training loops easier to debug
3. **Better for research**: More flexibility for experimentation
4. **Fixes data leakage**: Proper preprocessing workflow
5. **Stronger regularization**: Will address overfitting

## Risks

1. **Large scope**: Complete rewrite of 3 notebooks
2. **Testing required**: Need to verify models train properly
3. **Documentation**: Need to update all guides
4. **Compatibility**: realtime_frontend.py works with ONNX (no changes needed)

## Alternative: Quick Fix (TensorFlow)

If PyTorch migration is too ambitious, we can:
1. Fix data leakage in current TensorFlow notebooks (~1 hour)
2. Increase regularization (~30 minutes)
3. Keep existing ONNX export workflow

This would achieve 80% of the benefit with 10% of the effort.

## Recommendation

Given the scope, I recommend:
1. **Phase 1**: Fix data leakage in existing TensorFlow notebooks (immediate)
2. **Phase 2**: Add stronger regularization (immediate)
3. **Phase 3**: Migrate to PyTorch (if still desired after seeing Phase 1-2 results)

This allows validating that the overfitting is indeed due to data leakage before committing to a full rewrite.
