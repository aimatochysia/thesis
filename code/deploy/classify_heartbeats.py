"""
ECG Heartbeat Classification Deployment Script

This script implements a simulated deployment pipeline that:
1. Reads the usable CSV dataset (MLII signal)
2. Extracts 188-length heartbeat segments using R-peak annotations
3. Normalizes each heartbeat using the pre-trained scaler
4. Classifies each beat as NORMAL or ABNORMAL using the trained LSTM model

BEAT EXTRACTION RULES:
- Uses annotation information to segment the MLII signal into individual heartbeats
- Each heartbeat is exactly 188 samples long
- Beats that cannot meet the 188-length requirement are padded

MODEL:
- Loads the pre-trained LSTM model (ecg_lstm_v3_final.keras)
- Uses StandardScaler normalization (scaler_v3.pkl)

OUTPUT:
- Classification results per beat (NORMAL or ABNORMAL)
- Results saved to CSV

Usage:
    python classify_heartbeats.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    print("Error: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)


# Constants
BEAT_LENGTH = 188  # Required beat length for the model
PRE_SAMPLES = 70   # Samples before R-peak
POST_SAMPLES = 118 # Samples after R-peak (total = 70 + 118 = 188)

# Beat type mapping for classification interpretation
# Normal beats: N (Normal)
# Abnormal beats: A (Atrial premature), V (Ventricular premature), etc.
NORMAL_BEAT_TYPES = {'N'}
ABNORMAL_BEAT_TYPES = {'A', 'V', 'F', 'S', 'Q', '!', 'E', 'J', 'L', 'R'}


def load_mlii_signal(signal_csv_path: str) -> np.ndarray:
    """
    Load the MLII signal from CSV file.
    
    Args:
        signal_csv_path: Path to the MLII signal CSV
        
    Returns:
        NumPy array of MLII values
    """
    df = pd.read_csv(signal_csv_path, index_col=0)
    return df['MLII'].values.astype(np.float32)


def load_annotations(annotations_csv_path: str) -> pd.DataFrame:
    """
    Load parsed annotations from CSV file.
    
    Args:
        annotations_csv_path: Path to the parsed annotations CSV
        
    Returns:
        DataFrame with sample_index and beat_type columns
    """
    return pd.read_csv(annotations_csv_path)


def extract_beat(signal: np.ndarray, r_peak_idx: int, 
                 pre_samples: int = PRE_SAMPLES, 
                 post_samples: int = POST_SAMPLES) -> np.ndarray:
    """
    Extract a heartbeat segment centered around the R-peak.
    
    Args:
        signal: Full MLII signal array
        r_peak_idx: Index of the R-peak
        pre_samples: Number of samples before R-peak
        post_samples: Number of samples after R-peak
        
    Returns:
        NumPy array of beat segment (188 samples), padded if necessary
    """
    total_length = pre_samples + post_samples
    
    start_idx = r_peak_idx - pre_samples
    end_idx = r_peak_idx + post_samples
    
    # Handle edge cases with padding
    if start_idx < 0:
        # Pad at the beginning
        pad_before = -start_idx
        beat = np.zeros(total_length, dtype=np.float32)
        available_signal = signal[:end_idx]
        beat[pad_before:pad_before + len(available_signal)] = available_signal
    elif end_idx > len(signal):
        # Pad at the end
        pad_after = end_idx - len(signal)
        beat = np.zeros(total_length, dtype=np.float32)
        available_signal = signal[start_idx:]
        beat[:len(available_signal)] = available_signal
    else:
        # No padding needed
        beat = signal[start_idx:end_idx]
    
    return beat.astype(np.float32)


def normalize_beat(beat: np.ndarray, scaler) -> np.ndarray:
    """
    Normalize a beat using the pre-trained StandardScaler.
    
    The scaler was fitted during model training and transforms
    each feature (time point) to have mean=0 and std=1.
    
    Args:
        beat: Raw beat array (188 samples)
        scaler: Fitted StandardScaler object
        
    Returns:
        Normalized beat array
    """
    # Reshape for scaler: (1, 188) -> scaler expects 2D array
    beat_2d = beat.reshape(1, -1)
    normalized = scaler.transform(beat_2d)
    return normalized.flatten().astype(np.float32)


def classify_beat(model, normalized_beat: np.ndarray) -> tuple:
    """
    Classify a single normalized beat using the LSTM model.
    
    Args:
        model: Loaded Keras model
        normalized_beat: Normalized beat array (188 samples)
        
    Returns:
        Tuple of (predicted_class, probability)
        - predicted_class: 0 (NORMAL) or 1 (ABNORMAL)
        - probability: Probability of abnormal class
    """
    # Reshape for LSTM: (1, 188, 1)
    beat_input = normalized_beat.reshape(1, BEAT_LENGTH, 1)
    
    # Get prediction probabilities
    proba = model.predict(beat_input, verbose=0)
    
    # For binary classification with softmax output
    if proba.shape[1] == 2:
        prob_abnormal = float(proba[0, 1])
        predicted_class = 1 if prob_abnormal >= 0.5 else 0
    else:
        # Single output with sigmoid
        prob_abnormal = float(proba[0, 0])
        predicted_class = 1 if prob_abnormal >= 0.5 else 0
    
    return predicted_class, prob_abnormal


def get_ground_truth_label(beat_type: str) -> int:
    """
    Get ground truth label based on annotation beat type.
    
    Args:
        beat_type: Beat type from annotation (e.g., 'N', 'A', 'V')
        
    Returns:
        0 for NORMAL, 1 for ABNORMAL
    """
    if beat_type in NORMAL_BEAT_TYPES:
        return 0
    elif beat_type in ABNORMAL_BEAT_TYPES:
        return 1
    else:
        # Unknown types treated as special cases
        return -1  # Unknown


def run_classification_pipeline(
    signal_path: str,
    annotations_path: str,
    model_path: str,
    scaler_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Run the complete classification pipeline.
    
    Args:
        signal_path: Path to MLII signal CSV
        annotations_path: Path to parsed annotations CSV
        model_path: Path to trained Keras model
        scaler_path: Path to fitted scaler
        output_path: Path to save results CSV
        
    Returns:
        DataFrame with classification results
    """
    print("=" * 70)
    print("ECG Heartbeat Classification Pipeline")
    print("=" * 70)
    
    # Load signal
    print(f"\n1. Loading MLII signal from: {signal_path}")
    signal = load_mlii_signal(signal_path)
    print(f"   Signal length: {len(signal)} samples")
    
    # Load annotations
    print(f"\n2. Loading annotations from: {annotations_path}")
    annotations = load_annotations(annotations_path)
    print(f"   Total annotated beats: {len(annotations)}")
    
    # Load model
    print(f"\n3. Loading model from: {model_path}")
    model = load_model(model_path)
    print("   Model loaded successfully")
    
    # Load scaler
    print(f"\n4. Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("   Scaler loaded successfully")
    
    # Process each beat
    print("\n5. Processing beats...")
    results = []
    
    for idx, row in annotations.iterrows():
        r_peak_idx = row['sample_index']
        beat_type = row['beat_type']
        
        # Skip non-beat annotations (like rhythm annotations marked with '+')
        if beat_type == '+':
            continue
        
        # Extract beat segment
        beat = extract_beat(signal, r_peak_idx)
        
        # Normalize
        normalized_beat = normalize_beat(beat, scaler)
        
        # Classify
        predicted_class, prob_abnormal = classify_beat(model, normalized_beat)
        
        # Get ground truth (if available from annotation type)
        ground_truth = get_ground_truth_label(beat_type)
        
        # Map class to label
        predicted_label = "ABNORMAL" if predicted_class == 1 else "NORMAL"
        ground_truth_label = "ABNORMAL" if ground_truth == 1 else ("NORMAL" if ground_truth == 0 else "UNKNOWN")
        
        results.append({
            'beat_index': idx,
            'r_peak_sample': r_peak_idx,
            'annotation_type': beat_type,
            'ground_truth': ground_truth_label,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'probability_abnormal': round(prob_abnormal, 4),
            'probability_normal': round(1 - prob_abnormal, 4)
        })
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(annotations)} beats...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    print(f"\n6. Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Classification Summary")
    print("=" * 70)
    
    total_beats = len(results_df)
    normal_count = len(results_df[results_df['predicted_label'] == 'NORMAL'])
    abnormal_count = len(results_df[results_df['predicted_label'] == 'ABNORMAL'])
    
    print(f"\nTotal beats classified: {total_beats}")
    print(f"  - NORMAL:   {normal_count} ({normal_count/total_beats*100:.1f}%)")
    print(f"  - ABNORMAL: {abnormal_count} ({abnormal_count/total_beats*100:.1f}%)")
    
    # Accuracy calculation (for beats with known ground truth)
    known_gt = results_df[results_df['ground_truth'] != 'UNKNOWN']
    if len(known_gt) > 0:
        correct = len(known_gt[known_gt['ground_truth'] == known_gt['predicted_label']])
        accuracy = correct / len(known_gt) * 100
        print(f"\nAccuracy (on known ground truth): {accuracy:.2f}%")
        print(f"  - Beats with known labels: {len(known_gt)}")
        print(f"  - Correctly classified: {correct}")
    
    # Show sample predictions
    print("\nSample Predictions (first 10 beats):")
    print("-" * 70)
    for _, row in results_df.head(10).iterrows():
        print(f"  Beat {row['beat_index']:4d} | R-peak: {row['r_peak_sample']:6d} | "
              f"Annotation: {row['annotation_type']:2s} | "
              f"Predicted: {row['predicted_label']:8s} (prob={row['probability_abnormal']:.3f})")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    
    return results_df


def main():
    """Main function to run the classification pipeline."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    # Input files
    signal_path = os.path.join(sample_dir, 'mlii_signal.csv')
    annotations_path = os.path.join(sample_dir, 'parsed_annotations.csv')
    model_path = os.path.join(sample_dir, 'ecg_lstm_v3_final.keras')
    scaler_path = os.path.join(sample_dir, 'scaler_v3.pkl')
    
    # Output file
    output_path = os.path.join(sample_dir, 'classification_results.csv')
    
    # Check if required files exist
    required_files = [
        (signal_path, "MLII signal CSV"),
        (annotations_path, "Parsed annotations CSV"),
        (model_path, "Trained model"),
        (scaler_path, "Fitted scaler")
    ]
    
    missing_files = []
    for path, desc in required_files:
        if not os.path.exists(path):
            missing_files.append(f"  - {desc}: {path}")
    
    if missing_files:
        # Try to create the signal and annotation files first
        if not os.path.exists(signal_path) or not os.path.exists(annotations_path):
            print("Signal/Annotation files not found. Running data conversion first...")
            # Import and run conversion
            from convert_ecg_data import main as convert_main
            convert_main()
            print()
    
    # Re-check for model and scaler
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found: {scaler_path}")
        sys.exit(1)
    
    # Run pipeline
    results = run_classification_pipeline(
        signal_path=signal_path,
        annotations_path=annotations_path,
        model_path=model_path,
        scaler_path=scaler_path,
        output_path=output_path
    )
    
    return results


if __name__ == "__main__":
    main()
