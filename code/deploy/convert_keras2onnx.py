"""
Convert Keras LSTM model to ONNX using keras2onnx.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import keras2onnx
import onnx

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    h5_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    onnx_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    
    print(f"Loading model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("\nModel summary:")
    model.summary()
    
    # Convert to ONNX using keras2onnx
    print(f"\nConverting to ONNX...")
    onnx_model = keras2onnx.convert_keras(model, model.name)
    
    # Save ONNX model
    print(f"Saving ONNX model to: {onnx_path}")
    onnx.save_model(onnx_model, onnx_path)
    
    print("ONNX model saved successfully!")
    
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
