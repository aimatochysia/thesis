# Step-by-Step Training and Deployment Guide

This guide provides detailed instructions for training, distilling, and deploying ECG arrhythmia classification models.

## Overview

The complete workflow consists of 4 main steps:

1. **Data Preparation** - Gather and prepare ECG beat data
2. **Teacher Training** - Train robust teacher model with augmentations
3. **Student Distillation** - Train compact student using knowledge distillation
4. **Deployment** - Deploy model for real-time ECG classification

---

## Step 0: Data Preparation

### Required Data Format

Your ECG data should be in CSV format with:
- **188 columns** of ECG signal values (single heartbeat)
- **1 label column** (last column): 0 = normal, 1 = abnormal
- **No header row**
- **Baseline around 950** (typical ECG ADC value)

Example row:
```
955,954,956,955,...,931,932,0
```

### Option A: Use Pre-segmented Data (Recommended for Training)

If you have pre-segmented beat data (like the MIT-BIH arrhythmia dataset converted to beats):

1. Download ECG datasets from Kaggle:
   - [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
   - Or any dataset with 188-sample beats

2. Ensure format matches: 188 features + 1 label column

### Option B: Process Raw ECG Recordings

If you have continuous ECG recordings:

1. Use `deploy.py` to segment beats:
   ```bash
   python code/deployment/deploy.py \
       --input_csv your_recording.csv \
       --output_csv segmented_beats.csv \
       --fs 360
   ```

2. Manually label the segmented beats or use existing annotations

### Data Sources

| Source | Description | Link |
|--------|-------------|------|
| MIT-BIH | Standard arrhythmia dataset | PhysioNet |
| PTB | Diagnostic ECG database | PhysioNet |
| Kaggle ECG | Pre-processed beats | kaggle.com |

---

## Step 1: Train Robust Teacher Model

### Using Kaggle (Recommended)

1. **Upload notebook**: Upload `code/training/train_teacher_v2_robust.ipynb` to Kaggle

2. **Add dataset**: 
   - Click "Add data" 
   - Add your ECG dataset (ecg.csv)

3. **Configure paths**: Update the configuration cell:
   ```python
   DATA_PATH = '/kaggle/input/your-dataset/ecg.csv'
   OUTPUT_DIR = '/kaggle/working'
   ```

4. **Run all cells**: Click "Run All"

5. **Download outputs**:
   - `teacher_v2_robust.h5` - The trained model
   - `training_history_teacher.png` - Training curves
   - `robustness_curves_teacher.png` - Robustness analysis

### Using Local Machine

```bash
cd /path/to/thesis

python code/training/train_teacher_v2_robust.py \
    --data_path ecg.csv \
    --output_dir outputs/models \
    --epochs 200 \
    --batch_size 32
```

### Expected Results

After training, you should see:
- **Accuracy**: ~97-99% on test set
- **AUC**: ~0.98-0.99
- **Robustness**: <3% accuracy drop at ±40ms shift

---

## Step 2: Train Student Model with Distillation

### Using Kaggle (Recommended)

1. **Upload notebook**: Upload `code/training/train_student_distill.ipynb`

2. **Add data**:
   - Add your ECG dataset
   - Add the teacher model from Step 1 (upload `teacher_v2_robust.h5` as a dataset)

3. **Configure paths**:
   ```python
   DATA_PATH = '/kaggle/input/your-dataset/ecg.csv'
   TEACHER_MODEL_PATH = '/kaggle/input/teacher-model/teacher_v2_robust.h5'
   OUTPUT_DIR = '/kaggle/working'
   ```

4. **Run all cells**

5. **Download outputs**:
   - `student_distilled.h5` - **USE THIS FOR DEPLOYMENT**
   - `baseline_tiny.h5` - Non-distilled comparison
   - `model_comparison.csv` - Performance metrics

### Using Local Machine

```bash
python code/training/train_student_distill.py \
    --teacher_path outputs/models/teacher_v2_robust.h5 \
    --data_path ecg.csv \
    --output_dir outputs/models \
    --temperature 3.0 \
    --alpha 0.7
```

### Expected Results

| Metric | Teacher | Student (Distilled) |
|--------|---------|---------------------|
| Parameters | ~250k | ~20k |
| Accuracy | ~98% | ~97% |
| Size | ~1 MB | ~100 KB |
| Inference | ~10ms | ~3ms |

---

## Step 3: Test and Evaluate

### Run Robustness Evaluation

```bash
python code/eval/evaluate_robustness.py \
    --model_path outputs/models/student_distilled.h5 \
    --data_path ecg.csv \
    --output_dir outputs/plots
```

This generates:
- Robustness curves (accuracy vs temporal shift)
- Confusion matrices
- ROC curves
- Summary tables

### Compare Multiple Models

```bash
python code/eval/evaluate_robustness.py \
    --model_path outputs/models/teacher_v2_robust.h5,outputs/models/student_distilled.h5 \
    --model_names "Teacher,Student" \
    --data_path ecg.csv \
    --output_dir outputs/plots
```

---

## Step 4: Deployment

### Option A: Test on Long ECG Recording

Use `deploy.py` to process continuous ECG and get per-beat predictions:

```bash
python code/deployment/deploy.py \
    --input_csv long_recording.csv \
    --keras_h5 outputs/models/student_distilled.h5 \
    --output_csv outputs/predictions.csv \
    --plots_dir outputs/plots \
    --fs 360
```

**Outputs:**
- `predictions.csv` - Per-beat timestamps, probabilities, labels
- `continuous_with_beats.png` - ECG with detected R-peaks
- `beats_grid.png` - Grid of beats with predictions

### Option B: Convert to TFLite for Mobile

```bash
python code/deployment/export_tflite.py \
    --model_path outputs/models/student_distilled.h5 \
    --data_path ecg.csv \
    --quantize int8 \
    --compare
```

**Outputs:**
- `student_distilled_int8.tflite` - Quantized model (~25KB)
- Size and speed comparison

### Option C: Web/Mobile Integration

For frontend/mobile deployment:

1. **TensorFlow.js (Web)**:
   ```javascript
   // Convert to TensorFlow.js format
   tensorflowjs_converter \
       --input_format=keras \
       student_distilled.h5 \
       tfjs_model/
   ```

2. **TensorFlow Lite (Mobile)**:
   - Use the INT8 quantized model
   - Integrate with TFLite interpreter in your app
   - Process live ECG: segment → normalize → predict

---

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
│  Raw ECG → Beat Segmentation → Labeled Dataset (188 cols)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: TRAIN TEACHER MODEL                     │
│  train_teacher_v2_robust.ipynb → teacher_v2_robust.h5       │
│  (~250k params, robust to misalignment)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: DISTILL TO STUDENT                      │
│  train_student_distill.ipynb → student_distilled.h5         │
│  (~20k params, 12x smaller, comparable accuracy)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: EVALUATE                                │
│  evaluate_robustness.py → plots, metrics, tables            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 4: DEPLOY                                  │
│  Option A: deploy.py (test on recordings)                    │
│  Option B: export_tflite.py (mobile)                         │
│  Option C: TensorFlow.js (web)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

1. **GPU not detected on Kaggle**
   - Enable GPU: Settings → Accelerator → GPU

2. **Out of memory**
   - Reduce batch size: `BATCH_SIZE = 16`
   - Use smaller epochs initially

3. **Model not converging**
   - Check data normalization (baseline ~950)
   - Ensure labels are 0/1

4. **TFLite conversion fails**
   - Ensure TensorFlow ≥2.10
   - Try dynamic range quantization first

### Getting Help

- Check existing notebooks in `code/v0` to `code/v5` for reference implementations
- Review the README.md for additional documentation

---

## File Reference

| File | Purpose |
|------|---------|
| `train_teacher_v2_robust.ipynb` | Train robust teacher (Step 1) |
| `train_student_distill.ipynb` | Distill student model (Step 2) |
| `evaluate_robustness.py` | Generate evaluation plots |
| `deploy.py` | Process continuous ECG |
| `export_tflite.py` | Convert to TFLite |
