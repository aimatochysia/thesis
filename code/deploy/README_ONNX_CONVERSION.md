# Converting LSTM Model to ONNX for Cross-Platform Deployment

## Why ONNX?

ONNX (Open Neural Network Exchange) allows you to run the trained LSTM model without needing TensorFlow/Keras installed. This provides:
- Smaller deployment footprint
- Better cross-platform compatibility (Windows, Linux, macOS)
- Faster inference in some cases
- No heavy TensorFlow dependency

## Prerequisites

To convert the model, you need Python with TensorFlow installed (only for conversion, not for deployment).

```bash
pip install tensorflow==2.13.0 tf2onnx onnx onnxruntime
```

Note: The conversion only needs to be done once. After that, you only need `onnxruntime` for deployment.

## Conversion Method 1: Using SavedModel (Recommended)

```python
import tensorflow as tf
import subprocess
import os

# Load the H5 model
model = tf.keras.models.load_model('code/deploy/sample/ecg_lstm_final.h5', compile=False)

# Save as SavedModel first
saved_model_dir = 'code/deploy/sample/ecg_lstm_saved_model'
tf.saved_model.save(model, saved_model_dir)

# Convert SavedModel to ONNX using command line
subprocess.run([
    'python', '-m', 'tf2onnx.convert',
    '--saved-model', saved_model_dir,
    '--output', 'code/deploy/sample/ecg_lstm_final.onnx',
    '--opset', '13'
])
```

## Conversion Method 2: Direct Conversion Script

Save this as `convert_to_onnx.py`:

```python
#!/usr/bin/env python3
"""
One-time conversion script to convert H5 model to ONNX.
Run this on a machine with TensorFlow installed.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import tensorflow as tf
import numpy as np

def convert_model():
    # Paths
    h5_path = 'sample/ecg_lstm_final.h5'
    onnx_path = 'sample/ecg_lstm_final.onnx'
    saved_model_dir = 'sample/ecg_lstm_saved_model'
    
    print(f"Loading model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("Model loaded. Saving as SavedModel...")
    tf.saved_model.save(model, saved_model_dir)
    
    print(f"Converting to ONNX format...")
    import subprocess
    result = subprocess.run([
        'python', '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', onnx_path,
        '--opset', '13'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Successfully converted to {onnx_path}")
        
        # Verify
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        print(f"✓ ONNX model verified. Input shape: {session.get_inputs()[0].shape}")
        
        # Test inference
        test_input = np.random.randn(1, 188, 1).astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: test_input})
        print(f"✓ Test inference successful. Output shape: {output[0].shape}")
    else:
        print(f"✗ Conversion failed:")
        print(result.stderr)
        return False
    
    return True

if __name__ == '__main__':
    convert_model()
```

Then run:
```bash
cd code/deploy
python convert_to_onnx.py
```

## Conversion Method 3: Using Docker (If local conversion fails)

If you're having trouble with library versions, use Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install tensorflow==2.13.0 tf2onnx onnx onnxruntime numpy

# Copy model file
COPY sample/ecg_lstm_final.h5 /app/

# Conversion script
COPY convert_to_onnx.py /app/

# Run conversion
CMD ["python", "convert_to_onnx.py"]
```

Build and run:
```bash
docker build -t model-converter .
docker run -v $(pwd)/sample:/app/sample model-converter
```

## After Conversion

Once you have the ONNX model (`ecg_lstm_final.onnx`), you can run the application with only:

```bash
pip install onnxruntime numpy pandas flask joblib
python realtime_frontend.py
```

No TensorFlow needed! The application will automatically detect and use the ONNX model.

## Troubleshooting

### "Module 'numpy' has no attribute 'bool'"
This is a version compatibility issue. Try:
```bash
pip install "numpy<1.24" "protobuf<4"
```

### "AttributeError: 'Sequential' object has no attribute 'output_names'"
Try using an older version of tf2onnx:
```bash
pip install tf2onnx==1.13.0
```

### Conversion takes too long or fails
Try converting on a different machine or use the Docker method above.

## Verifying the Converted Model

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('sample/ecg_lstm_final.onnx')

# Check input/output
print("Input:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].name, session.get_outputs()[0].shape)

# Test with random data
test_input = np.random.randn(1, 188, 1).astype(np.float32)
result = session.run(None, {session.get_inputs()[0].name: test_input})
print("Prediction:", result[0])
```

## Performance Comparison

| Metric | TensorFlow/Keras | ONNX Runtime |
|--------|------------------|--------------|
| Installation Size | ~500 MB | ~10 MB |
| Cold start time | 3-5 seconds | <1 second |
| Inference time | ~5-10 ms | ~3-7 ms |
| Cross-platform | Complex | Simple |

## Support

If you continue to have issues with conversion, you can:
1. Use the provided H5 model with TensorFlow/Keras (fallback mode)
2. Request a pre-converted ONNX model from the maintainer
3. Use online conversion services like convertmodel.com
