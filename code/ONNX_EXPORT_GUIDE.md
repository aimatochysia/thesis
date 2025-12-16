# ONNX Export in Training Notebooks

## Overview

All training notebooks (v2, v3, v5) have been enhanced to automatically export models to ONNX format after training completes. This enables lightweight, cross-platform deployment without requiring TensorFlow/Keras.

## What Was Added

Each training notebook now includes a new **STEP 11: Export to ONNX Format** section that:

1. **Automatically converts** the trained Keras model to ONNX format
2. **Saves the ONNX model** alongside the Keras model
3. **Verifies the conversion** by testing inference
4. **Provides deployment instructions** for using the ONNX model

## Modified Notebooks

### v2 - CNN Model
- **Notebook**: `code/v2/ecg-train-last-v2.ipynb`
- **Keras Output**: `ecg_cnn_v2_final.keras`
- **ONNX Output**: `ecg_cnn_v2_final.onnx`
- **Scaler**: `scaler_v2.pkl`

### v3 - LSTM Model
- **Notebook**: `code/v3/ecg-train-last-v3.ipynb`
- **Keras Output**: `ecg_lstm_v3_final.keras`
- **ONNX Output**: `ecg_lstm_v3_final.onnx`
- **Scaler**: `scaler_v3.pkl`

### v5 - Transformer Model
- **Notebook**: `code/v5/ecg-train-last-v5.ipynb`
- **Keras Output**: `ecg_transformer_v5_final.keras`
- **ONNX Output**: `ecg_transformer_v5_final.onnx`
- **Scaler**: `scaler_v5.pkl`

## How It Works

When you run a training notebook, after STEP 10 (Save Model), STEP 11 will:

```python
# 1. Import ONNX conversion libraries
import tf2onnx
import onnx

# 2. Convert the trained model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path='model_name.onnx'
)

# 3. Verify the ONNX model works
import onnxruntime as ort
session = ort.InferenceSession('model_name.onnx')
# Test with sample data...

# 4. Print deployment instructions
```

## Prerequisites for ONNX Export

The ONNX export requires additional packages:

```bash
pip install tf2onnx onnx onnxruntime
```

If these packages are not installed, the notebook will:
- ✅ Still save the Keras model successfully
- ⚠️ Skip ONNX export with a warning
- 💡 Show instructions for manual conversion later

## Using the ONNX Models

### Lightweight Deployment

Once you have the ONNX model, you can deploy with minimal dependencies:

```bash
# Install only lightweight packages (no TensorFlow!)
pip install onnxruntime numpy pandas joblib scikit-learn

# Use in your application
import onnxruntime as ort
import numpy as np
import joblib

# Load scaler and model
scaler = joblib.load('scaler_v3.pkl')
session = ort.InferenceSession('ecg_lstm_v3_final.onnx')

# Preprocess and predict
beat_normalized = scaler.transform(beat.reshape(1, -1))
beat_input = beat_normalized.reshape(1, 188, 1).astype(np.float32)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
prediction = session.run([output_name], {input_name: beat_input})
```

## Benefits

| Aspect | Keras/TensorFlow | ONNX Runtime |
|--------|------------------|--------------|
| Installation Size | ~500 MB | ~10 MB |
| Dependencies | tensorflow, keras | onnxruntime only |
| Load Time | 3-5 seconds | <1 second |
| Inference Speed | 5-10 ms | 3-7 ms |
| Cross-Platform | Complex | Simple |
| Production Ready | ⚠️ Heavy | ✅ Optimal |

## Troubleshooting

### "ModuleNotFoundError: No module named 'tf2onnx'"

Install the required packages:
```bash
pip install tf2onnx onnx onnxruntime
```

### ONNX Export Fails During Training

The notebook will continue and save the Keras model. You can convert manually later:

```python
import tensorflow as tf
import tf2onnx

# Load the saved Keras model
model = tf.keras.models.load_model('ecg_lstm_v3_final.keras')

# Convert to ONNX
spec = (tf.TensorSpec((None, 188, 1), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path='ecg_lstm_v3_final.onnx'
)
```

Or use the standalone conversion script:
```bash
python code/deploy/convert_to_onnx_standalone.py
```

### Version Compatibility Issues

If you encounter compatibility issues between TensorFlow, tf2onnx, and numpy:

```bash
# Recommended versions
pip install tensorflow==2.16.2 "numpy<2.0" tf2onnx==1.16.1 onnx==1.17.0
```

## Integration with Deployment

The exported ONNX models are compatible with:

1. **realtime_frontend.py** - Automatically detects and uses ONNX models
2. **deployment.py** - Can be updated to use ONNX Runtime
3. **Edge devices** - ARM, mobile, embedded systems
4. **Cloud services** - AWS, Azure, GCP all support ONNX Runtime

## Best Practices

1. **Always train with the ONNX export enabled** - Ensures you have both formats
2. **Test both models** - Verify that ONNX and Keras give same results
3. **Version control both formats** - Keep .keras for retraining, .onnx for deployment
4. **Document model versions** - Track which training run produced which ONNX file

## Example Workflow

```bash
# 1. Train model (generates both .keras and .onnx)
jupyter notebook code/v3/ecg-train-last-v3.ipynb

# 2. Copy ONNX model to deployment directory
cp ecg_lstm_v3_final.onnx code/deploy/sample/
cp scaler_v3.pkl code/deploy/sample/

# 3. Deploy with lightweight dependencies
cd code/deploy
pip install onnxruntime numpy pandas flask joblib scikit-learn
python realtime_frontend.py
```

## Verification

After training completes, verify both models exist:

```bash
ls -lh ecg_lstm_v3_final.*
# Should show:
# ecg_lstm_v3_final.keras  (~350 KB)
# ecg_lstm_v3_final.onnx   (~1.1 MB)
```

Test the ONNX model:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('ecg_lstm_v3_final.onnx')
test_input = np.random.randn(1, 188, 1).astype(np.float32)
result = session.run(None, {session.get_inputs()[0].name: test_input})
print(f"ONNX inference successful! Output shape: {result[0].shape}")
```

## Support

For questions or issues:
1. Check the [main ONNX conversion guide](../deploy/README_ONNX_CONVERSION.md)
2. Review training notebook outputs for specific error messages
3. Ensure all prerequisites are installed
4. Try manual conversion if automatic export fails

---

**Note**: The ONNX export is a post-training step. If it fails, your model training is still successful and you have the Keras model saved. ONNX conversion can be done later.
