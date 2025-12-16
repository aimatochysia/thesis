"""
Convert Keras LSTM model to ONNX format for cross-platform inference without Keras dependency.

This script converts the ecg_lstm_final.h5 model to ONNX format so it can be used
with onnxruntime instead of TensorFlow/Keras, enabling better cross-platform compatibility.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx


def convert_h5_to_onnx(h5_path, onnx_path):
    """
    Convert a Keras h5 model to ONNX format.
    
    Args:
        h5_path: Path to the input .h5 model file
        onnx_path: Path where the output .onnx model will be saved
    """
    print(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("Model summary:")
    model.summary()
    
    # Get input shape from model
    input_shape = model.input_shape
    print(f"Input shape: {input_shape}")
    
    # Convert to ONNX
    print(f"Converting to ONNX format...")
    
    # Create input signature
    spec = (tf.TensorSpec((None, 188, 1), tf.float32, name="input"),)
    
    # Convert using tf2onnx
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=onnx_path
    )
    
    print(f"Successfully converted model to ONNX format: {onnx_path}")
    print(f"ONNX model saved successfully!")
    
    return onnx_path


def verify_onnx_model(onnx_path, sample_input):
    """
    Verify that the ONNX model can be loaded and used for inference.
    
    Args:
        onnx_path: Path to the ONNX model
        sample_input: Sample input data for testing
    """
    import onnxruntime as ort
    
    print(f"\nVerifying ONNX model: {onnx_path}")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output details
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    
    # Run inference with sample input
    result = session.run([output_name], {input_name: sample_input})
    
    print(f"Test inference result shape: {result[0].shape}")
    print(f"Test inference result: {result[0]}")
    
    return result


def main():
    """Main conversion script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # Input and output paths
    h5_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    onnx_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    
    if not os.path.exists(h5_path):
        print(f"Error: Model file not found: {h5_path}")
        sys.exit(1)
    
    # Convert model
    convert_h5_to_onnx(h5_path, onnx_path)
    
    # Verify with sample input (batch_size=1, sequence_length=188, features=1)
    sample_input = np.random.randn(1, 188, 1).astype(np.float32)
    verify_onnx_model(onnx_path, sample_input)
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    print("="*60)


if __name__ == "__main__":
    main()
