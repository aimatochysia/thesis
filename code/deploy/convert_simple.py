"""
Simple conversion script that saves as SavedModel and converts to ONNX.
"""

import os
import sys
import subprocess
import numpy as np

# Use TensorFlow with eager execution disabled to avoid compatibility issues
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = '0'

import tensorflow as tf

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    h5_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    saved_model_dir = os.path.join(sample_dir, 'ecg_lstm_saved_model')
    onnx_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    
    print(f"Loading model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("\nModel summary:")
    model.summary()
    
    # Save as SavedModel
    print(f"\nSaving as SavedModel to: {saved_model_dir}")
    tf.saved_model.save(model, saved_model_dir)
    print("SavedModel saved successfully!")
    
    # Convert to ONNX using command line
    print(f"\nConverting to ONNX...")
    cmd = [
        'python3', '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', onnx_path,
        '--opset', '13'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"\nONNX model saved to: {onnx_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to ONNX: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"Output name: {output_name}, shape: {session.get_outputs()[0].shape}")
    
    # Test inference
    sample_input = np.random.randn(1, 188, 1).astype(np.float32)
    result = session.run([output_name], {input_name: sample_input})
    print(f"Test inference successful! Output shape: {result[0].shape}")
    print(f"Output: {result[0]}")
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print(f"ONNX model: {onnx_path}")
    print("="*60)

if __name__ == "__main__":
    main()
