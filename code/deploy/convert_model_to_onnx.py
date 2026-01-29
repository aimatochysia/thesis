import os
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx


def convert_h5_to_onnx(h5_path, onnx_path):
    print(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("Model summary:")
    model.summary()
    
    input_shape = model.input_shape
    print(f"Input shape: {input_shape}")
    
    print(f"Converting to ONNX format...")
    
    spec = (tf.TensorSpec((None, 188, 1), tf.float32, name="input"),)
    
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
    import onnxruntime as ort
    
    print(f"\nVerifying ONNX model: {onnx_path}")
    
    session = ort.InferenceSession(onnx_path)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    
    result = session.run([output_name], {input_name: sample_input})
    
    print(f"Test inference result shape: {result[0].shape}")
    print(f"Test inference result: {result[0]}")
    
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    h5_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    onnx_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    
    if not os.path.exists(h5_path):
        print(f"Error: Model file not found: {h5_path}")
        sys.exit(1)
    
    convert_h5_to_onnx(h5_path, onnx_path)
    
    sample_input = np.random.randn(1, 188, 1).astype(np.float32)
    verify_onnx_model(onnx_path, sample_input)
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    print("="*60)


if __name__ == "__main__":
    main()
