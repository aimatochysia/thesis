#!/usr/bin/env python3
"""
Standalone ONNX Conversion Script

This script converts the Keras H5 LSTM model to ONNX format for deployment without Keras.
Run this once on a machine with TensorFlow installed. The resulting ONNX file can then
be used on any platform with just onnxruntime (no TensorFlow needed).

Usage:
    python convert_to_onnx_standalone.py

Requirements for conversion (one-time):
    pip install tensorflow onnx onnxruntime

Requirements for deployment (after conversion):
    pip install onnxruntime  (much lighter!)
"""

import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_and_install_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import tensorflow
    except ImportError:
        missing.append('tensorflow')
    
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    try:
        import onnxruntime
    except ImportError:
        missing.append('onnxruntime')
    
    if missing:
        print("Missing required packages for conversion:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def convert_h5_to_onnx():
    """Convert H5 model to ONNX format."""
    import tensorflow as tf
    import numpy as np
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # Paths
    h5_path = os.path.join(sample_dir, 'ecg_lstm_final.h5')
    onnx_path = os.path.join(sample_dir, 'ecg_lstm_final.onnx')
    saved_model_dir = os.path.join(sample_dir, 'temp_saved_model')
    
    # Check if H5 file exists
    if not os.path.exists(h5_path):
        print(f"Error: Model file not found: {h5_path}")
        return False
    
    print("="*60)
    print("ONNX Model Conversion Tool")
    print("="*60)
    print(f"\nInput:  {h5_path}")
    print(f"Output: {onnx_path}")
    print()
    
    try:
        # Load H5 model
        print("[1/4] Loading Keras H5 model...")
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"      ✓ Model loaded successfully")
        print(f"      Input shape: {model.input_shape}")
        print(f"      Output shape: {model.output_shape}")
        
        # Save as SavedModel (intermediate format)
        print("\n[2/4] Converting to SavedModel format...")
        if os.path.exists(saved_model_dir):
            import shutil
            shutil.rmtree(saved_model_dir)
        tf.saved_model.save(model, saved_model_dir)
        print(f"      ✓ SavedModel created at: {saved_model_dir}")
        
        # Convert to ONNX using tf2onnx command line tool
        print("\n[3/4] Converting SavedModel to ONNX...")
        import subprocess
        
        cmd = [
            sys.executable, '-m', 'tf2onnx.convert',
            '--saved-model', saved_model_dir,
            '--output', onnx_path,
            '--opset', '13',
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"      ✗ Conversion failed!")
            print(f"\nError output:")
            print(result.stderr)
            return False
        
        print(f"      ✓ ONNX model created successfully")
        
        # Verify ONNX model
        print("\n[4/4] Verifying ONNX model...")
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"      ✓ ONNX model is valid")
        print(f"      Input:  {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        print(f"      Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
        # Test inference
        print("\n      Running test inference...")
        test_input = np.random.randn(1, 188, 1).astype(np.float32)
        test_output = session.run([output_info.name], {input_info.name: test_input})
        print(f"      ✓ Test inference successful")
        print(f"      Test output shape: {test_output[0].shape}")
        print(f"      Test output: {test_output[0]}")
        
        # Cleanup temporary SavedModel
        print("\n      Cleaning up temporary files...")
        if os.path.exists(saved_model_dir):
            import shutil
            shutil.rmtree(saved_model_dir)
        print(f"      ✓ Cleanup complete")
        
        # Success message
        print("\n" + "="*60)
        print("✓ CONVERSION SUCCESSFUL!")
        print("="*60)
        print(f"\nONNX model saved to: {onnx_path}")
        print(f"File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")
        print("\nYou can now use this model with onnxruntime only:")
        print("  pip install onnxruntime numpy pandas flask joblib")
        print("  python realtime_frontend.py")
        print("\nNo TensorFlow/Keras required for deployment!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print()
    
    # Check dependencies
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Perform conversion
    success = convert_h5_to_onnx()
    
    if not success:
        print("\n" + "="*60)
        print("CONVERSION FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
        print("See README_ONNX_CONVERSION.md for troubleshooting tips.")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
