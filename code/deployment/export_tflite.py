#!/usr/bin/env python
"""
TFLite INT8 Conversion Script

Convert Keras models to TensorFlow Lite format with INT8 quantization
for efficient deployment on low-end devices.

Usage:
    python export_tflite.py --model_path model.h5 --output_path model.tflite \
        --data_path ecg.csv --quantize int8

Features:
- Full integer quantization (INT8)
- Dynamic range quantization option
- Representative dataset calibration
- Size and latency metrics reporting
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from typing import Optional, Callable, Generator

import tensorflow as tf


def representative_dataset_generator(
    X_samples: np.ndarray,
    n_samples: int = 100,
) -> Generator:
    """
    Generator for representative dataset used in quantization calibration.

    Args:
        X_samples: Sample data for calibration
        n_samples: Number of samples to use

    Yields:
        List of input arrays
    """
    indices = np.random.choice(len(X_samples), min(n_samples, len(X_samples)), replace=False)

    for idx in indices:
        sample = X_samples[idx:idx+1].astype(np.float32)
        yield [sample]


def convert_to_tflite(
    model_path: str,
    output_path: str,
    quantization: str = "none",
    representative_data: Optional[np.ndarray] = None,
) -> dict:
    """
    Convert Keras model to TFLite format.

    Args:
        model_path: Path to Keras model (.h5)
        output_path: Path for output TFLite file
        quantization: Quantization mode ('none', 'dynamic', 'int8')
        representative_data: Data for INT8 quantization calibration

    Returns:
        Dict with conversion info (size, etc.)
    """
    # Load Keras model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Configure quantization
    if quantization == "dynamic":
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization == "int8":
        print("Applying INT8 quantization...")
        if representative_data is None:
            raise ValueError("INT8 quantization requires representative data")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_generator(
            representative_data, n_samples=100
        )

        # For full integer quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    elif quantization == "float16":
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Get sizes
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(output_path)

    info = {
        "original_size_bytes": original_size,
        "original_size_mb": original_size / (1024 * 1024),
        "tflite_size_bytes": tflite_size,
        "tflite_size_mb": tflite_size / (1024 * 1024),
        "compression_ratio": original_size / tflite_size,
        "quantization": quantization,
    }

    print(f"\nConversion complete:")
    print(f"  Original size: {info['original_size_mb']:.2f} MB")
    print(f"  TFLite size: {info['tflite_size_mb']:.2f} MB")
    print(f"  Compression ratio: {info['compression_ratio']:.2f}x")

    return info


def measure_tflite_inference(
    tflite_path: str,
    X_test: np.ndarray,
    n_runs: int = 100,
) -> dict:
    """
    Measure TFLite model inference time.

    Args:
        tflite_path: Path to TFLite model
        X_test: Test data
        n_runs: Number of runs for timing

    Returns:
        Dict with timing info
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if quantized
    is_quantized = input_details[0]['dtype'] == np.int8

    # Prepare input
    input_shape = input_details[0]['shape']

    # Warm up
    for _ in range(10):
        sample = X_test[:1].astype(np.float32)
        if is_quantized:
            input_scale, input_zero_point = input_details[0]['quantization']
            sample = (sample / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()

    # Measure
    times = []
    for _ in range(n_runs):
        sample = X_test[:1].astype(np.float32)
        if is_quantized:
            sample = (sample / input_scale + input_zero_point).astype(np.int8)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    return {
        "mean_inference_ms": np.mean(times),
        "std_inference_ms": np.std(times),
        "min_inference_ms": np.min(times),
        "max_inference_ms": np.max(times),
        "is_quantized": is_quantized,
    }


def compare_models(
    keras_path: str,
    tflite_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Compare Keras and TFLite model accuracy and speed.

    Args:
        keras_path: Path to Keras model
        tflite_path: Path to TFLite model
        X_test: Test data
        y_test: Test labels

    Returns:
        Dict with comparison metrics
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Keras inference
    print("\nMeasuring Keras model...")
    keras_model = tf.keras.models.load_model(keras_path)

    keras_times = []
    for _ in range(100):
        start = time.perf_counter()
        keras_model.predict(X_test[:1], verbose=0)
        end = time.perf_counter()
        keras_times.append((end - start) * 1000)

    keras_preds = np.argmax(keras_model.predict(X_test, verbose=0), axis=1)
    keras_acc = accuracy_score(y_test, keras_preds)
    keras_f1 = f1_score(y_test, keras_preds, pos_label=1)

    # TFLite inference
    print("Measuring TFLite model...")
    tflite_metrics = measure_tflite_inference(tflite_path, X_test)

    # TFLite accuracy
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_quantized = input_details[0]['dtype'] == np.int8

    tflite_preds = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        if is_quantized:
            input_scale, input_zero_point = input_details[0]['quantization']
            sample = (sample / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if is_quantized:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        tflite_preds.append(np.argmax(output))

    tflite_preds = np.array(tflite_preds)
    tflite_acc = accuracy_score(y_test, tflite_preds)
    tflite_f1 = f1_score(y_test, tflite_preds, pos_label=1)

    # Compare
    comparison = {
        "keras_accuracy": keras_acc,
        "keras_f1": keras_f1,
        "keras_inference_ms": np.mean(keras_times),
        "keras_size_mb": os.path.getsize(keras_path) / (1024 * 1024),
        "tflite_accuracy": tflite_acc,
        "tflite_f1": tflite_f1,
        "tflite_inference_ms": tflite_metrics["mean_inference_ms"],
        "tflite_size_mb": os.path.getsize(tflite_path) / (1024 * 1024),
        "accuracy_diff": keras_acc - tflite_acc,
        "speedup": np.mean(keras_times) / tflite_metrics["mean_inference_ms"],
        "size_reduction": os.path.getsize(keras_path) / os.path.getsize(tflite_path),
    }

    print("\nComparison Results:")
    print("=" * 50)
    print(f"{'Metric':<25} {'Keras':>12} {'TFLite':>12}")
    print("-" * 50)
    print(f"{'Accuracy':<25} {keras_acc:>12.4f} {tflite_acc:>12.4f}")
    print(f"{'F1 Score':<25} {keras_f1:>12.4f} {tflite_f1:>12.4f}")
    print(f"{'Inference Time (ms)':<25} {np.mean(keras_times):>12.2f} {tflite_metrics['mean_inference_ms']:>12.2f}")
    print(f"{'Model Size (MB)':<25} {comparison['keras_size_mb']:>12.2f} {comparison['tflite_size_mb']:>12.2f}")
    print("-" * 50)
    print(f"{'Accuracy Difference':<25} {comparison['accuracy_diff']:>12.4f}")
    print(f"{'Speedup':<25} {comparison['speedup']:>12.2f}x")
    print(f"{'Size Reduction':<25} {comparison['size_reduction']:>12.2f}x")

    return comparison


def load_calibration_data(data_path: str, n_samples: int = 200) -> np.ndarray:
    """Load sample data for quantization calibration."""
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]

    return X


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Keras model (.h5)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path for output TFLite file (default: same as model with .tflite)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to data CSV for calibration (required for int8)")
    parser.add_argument("--quantize", type=str, default="int8",
                        choices=["none", "dynamic", "int8", "float16"],
                        help="Quantization mode")
    parser.add_argument("--compare", action="store_true",
                        help="Compare Keras and TFLite models")
    args = parser.parse_args()

    # Set output path
    if args.output_path is None:
        base = os.path.splitext(args.model_path)[0]
        suffix = f"_{args.quantize}" if args.quantize != "none" else ""
        args.output_path = f"{base}{suffix}.tflite"

    # Load calibration data if needed
    representative_data = None
    if args.quantize == "int8":
        if args.data_path is None:
            raise ValueError("--data_path required for INT8 quantization")
        representative_data = load_calibration_data(args.data_path)

    # Convert
    info = convert_to_tflite(
        args.model_path,
        args.output_path,
        quantization=args.quantize,
        representative_data=representative_data,
    )

    # Save info
    info_path = args.output_path.replace('.tflite', '_info.csv')
    pd.DataFrame([info]).to_csv(info_path, index=False)
    print(f"Conversion info saved to {info_path}")

    # Compare if requested
    if args.compare and args.data_path:
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(args.data_path, header=None)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int32)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        comparison = compare_models(
            args.model_path,
            args.output_path,
            X_test,
            y_test
        )

        comparison_path = args.output_path.replace('.tflite', '_comparison.csv')
        pd.DataFrame([comparison]).to_csv(comparison_path, index=False)
        print(f"Comparison results saved to {comparison_path}")


if __name__ == "__main__":
    main()
