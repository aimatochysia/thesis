# Data Leakage Fix - Demo for v3 LSTM

## Problem Identified

All three notebooks (v2, v3, v5) have the same critical data leakage issue:

```python
# CURRENT CODE (WRONG):
# Step 1: Normalize ALL data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # ❌ Fits on entire dataset

# Step 2: Then split
X_train, X_val, y_train, y_val = train_test_split(X_normalized, ...)
```

**Why this is wrong:**
- The scaler learns statistics (mean, std) from the ENTIRE dataset including validation/test data
- When validation/test data is normalized using these statistics, it's been "contaminated" with information it shouldn't have
- This causes the model to appear to perform better than it actually would on truly unseen data
- Result: 99-100% accuracy (artificially inflated)

## The Fix

```python
# CORRECTED CODE:
# Step 1: Split FIRST
X_train, X_temp, y_train, y_temp = train_test_split(X, y, ...)  # ✅ Split raw data first

# Step 2: Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)  # ✅ Fit only on training

# Step 3: Transform other sets using training statistics
X_val_normalized = scaler.transform(X_val)    # ✅ Transform, don't fit
X_test_normalized = scaler.transform(X_test)  # ✅ Transform, don't fit
```

## Code Changes Required

### In STEP 4 (Data Preprocessing), replace:

**BEFORE:**
```python
# =============================================================================
# STEP 4: Data Preprocessing - NORMALIZATION
# =============================================================================
# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # ❌ WRONG: Fits on all data

# ... later in STEP 5 ...
X_train, X_temp, y_train, y_temp = train_test_split(
    X_normalized, y_categorical,  # ❌ Using pre-normalized data
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
```

**AFTER:**
```python
# =============================================================================
# STEP 4: Data Preprocessing - Separate Features and Labels
# =============================================================================
X = df.drop('label', axis=1).values
y = df['label'].values

# Convert labels to categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# =============================================================================
# STEP 5: Train-Validation-Test Split (80%-10%-10%)
# =============================================================================
# IMPORTANT: Split BEFORE normalization to prevent data leakage

RANDOM_STATE = 42

# First split: 80% training, 20% temporary (for validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical,  # ✅ Split raw data FIRST
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Get encoded labels for temp set (for stratification)
y_temp_encoded = np.argmax(y_temp, axis=1)

# Second split: 50% of temp = 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp_encoded
)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Validation set: {X_val.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

# =============================================================================
# STEP 5.1: NORMALIZATION - Fit ONLY on training data
# =============================================================================
# This is the CRITICAL FIX for data leakage

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)  # ✅ Fit only on training
X_val_normalized = scaler.transform(X_val)          # ✅ Transform using training stats
X_test_normalized = scaler.transform(X_test)        # ✅ Transform using training stats

print(f'\\nNormalization applied (training stats only):')
print(f'Training mean: {scaler.mean_[:5]}')
print(f'Training std: {scaler.scale_[:5]}')

# Reshape for LSTM
X_train = X_train_normalized.reshape(-1, 188, 1)
X_val = X_val_normalized.reshape(-1, 188, 1)
X_test = X_test_normalized.reshape(-1, 188, 1)
```

## Expected Results After Fix

**Before Fix (with data leakage):**
- Training accuracy: 99.63%
- Validation accuracy: 99.80%
- Test accuracy: 100%
- ⚠️ Suspiciously perfect - indicates data leakage

**After Fix (without data leakage):**
- Training accuracy: ~85-92%
- Validation accuracy: ~83-90%
- Test accuracy: ~82-89%
- ✅ More realistic - shows actual model performance

## Additional Regularization

To further prevent overfitting, also increase dropout:

```python
# In model architecture, change:
Dropout(0.3)  # OLD
# to:
Dropout(0.5)  # NEW - more aggressive regularization
```

And reduce early stopping patience:

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,  # Changed from 15 to 10
    restore_best_weights=True,
    verbose=1
)
```

## How to Apply This Fix

1. Open the notebook: `code/v3/ecg-train-last-v3.ipynb`
2. Locate STEP 4 (Data Preprocessing)
3. Replace the code as shown above
4. Rerun all cells
5. Observe more realistic accuracy numbers

The same fix applies to v2 and v5 notebooks - they all have identical data leakage issues.
