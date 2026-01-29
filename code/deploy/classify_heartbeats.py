import os
import sys
import numpy as np
import pandas as pd
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    print("Error: TensorFlow is required. Install with: pip install tensorflow")
    sys.exit(1)


BEAT_LENGTH = 188
PRE_SAMPLES = 70
POST_SAMPLES = 118

NORMAL_BEAT_TYPES = {'N'}
ABNORMAL_BEAT_TYPES = {'A', 'V', 'F', 'S', 'Q', '!', 'E', 'J', 'L', 'R'}


def load_mlii_signal(signal_csv_path: str) -> np.ndarray:
    df = pd.read_csv(signal_csv_path, index_col=0)
    return df['MLII'].values.astype(np.float32)


def load_annotations(annotations_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(annotations_csv_path)


def extract_beat(signal: np.ndarray, r_peak_idx: int, 
                 pre_samples: int = PRE_SAMPLES, 
                 post_samples: int = POST_SAMPLES) -> np.ndarray:
    total_length = pre_samples + post_samples
    
    start_idx = r_peak_idx - pre_samples
    end_idx = r_peak_idx + post_samples
    
    if start_idx < 0:
        pad_before = -start_idx
        beat = np.zeros(total_length, dtype=np.float32)
        available_signal = signal[:end_idx]
        beat[pad_before:pad_before + len(available_signal)] = available_signal
    elif end_idx > len(signal):
        pad_after = end_idx - len(signal)
        beat = np.zeros(total_length, dtype=np.float32)
        available_signal = signal[start_idx:]
        beat[:len(available_signal)] = available_signal
    else:
        beat = signal[start_idx:end_idx]
    
    return beat.astype(np.float32)


def normalize_beat(beat: np.ndarray, scaler) -> np.ndarray:
    beat_2d = beat.reshape(1, -1)
    normalized = scaler.transform(beat_2d)
    return normalized.flatten().astype(np.float32)


def classify_beat(model, normalized_beat: np.ndarray) -> tuple:
    beat_input = normalized_beat.reshape(1, BEAT_LENGTH, 1)
    
    proba = model.predict(beat_input, verbose=0)
    
    if proba.ndim != 2 or proba.shape[0] != 1:
        raise ValueError(f"Unexpected model output shape: {proba.shape}. Expected (1, n_classes).")
    
    if proba.shape[1] == 2:
        prob_abnormal = float(proba[0, 1])
        predicted_class = 1 if prob_abnormal >= 0.5 else 0
    elif proba.shape[1] == 1:
        prob_abnormal = float(proba[0, 0])
        predicted_class = 1 if prob_abnormal >= 0.5 else 0
    else:
        raise ValueError(f"Unexpected model output shape: {proba.shape}. "
                         f"Expected 1 or 2 output units for binary classification.")
    
    return predicted_class, prob_abnormal


def get_ground_truth_label(beat_type: str) -> int:
    if beat_type in NORMAL_BEAT_TYPES:
        return 0
    elif beat_type in ABNORMAL_BEAT_TYPES:
        return 1
    else:
        return -1


def run_classification_pipeline(
    signal_path: str,
    annotations_path: str,
    model_path: str,
    scaler_path: str,
    output_path: str
) -> pd.DataFrame:
    print("=" * 70)
    print("ECG Heartbeat Classification Pipeline")
    print("=" * 70)
    
    print(f"\n1. Loading MLII signal from: {signal_path}")
    signal = load_mlii_signal(signal_path)
    print(f"   Signal length: {len(signal)} samples")
    
    print(f"\n2. Loading annotations from: {annotations_path}")
    annotations = load_annotations(annotations_path)
    print(f"   Total annotated beats: {len(annotations)}")
    
    print(f"\n3. Loading model from: {model_path}")
    model = load_model(model_path)
    print("   Model loaded successfully")
    
    print(f"\n4. Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("   Scaler loaded successfully")
    
    print("\n5. Processing beats...")
    results = []
    
    for idx, row in annotations.iterrows():
        r_peak_idx = row['sample_index']
        beat_type = row['beat_type']
        
        if beat_type == '+':
            continue
        
        beat = extract_beat(signal, r_peak_idx)
        
        normalized_beat = normalize_beat(beat, scaler)
        
        predicted_class, prob_abnormal = classify_beat(model, normalized_beat)
        
        ground_truth = get_ground_truth_label(beat_type)
        
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
        
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(annotations)} beats...")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n6. Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("Classification Summary")
    print("=" * 70)
    
    total_beats = len(results_df)
    normal_count = len(results_df[results_df['predicted_label'] == 'NORMAL'])
    abnormal_count = len(results_df[results_df['predicted_label'] == 'ABNORMAL'])
    
    print(f"\nTotal beats classified: {total_beats}")
    print(f"  - NORMAL:   {normal_count} ({normal_count/total_beats*100:.1f}%)")
    print(f"  - ABNORMAL: {abnormal_count} ({abnormal_count/total_beats*100:.1f}%)")
    
    known_gt = results_df[results_df['ground_truth'] != 'UNKNOWN']
    if len(known_gt) > 0:
        correct = len(known_gt[known_gt['ground_truth'] == known_gt['predicted_label']])
        accuracy = correct / len(known_gt) * 100
        print(f"\nAccuracy (on known ground truth): {accuracy:.2f}%")
        print(f"  - Beats with known labels: {len(known_gt)}")
        print(f"  - Correctly classified: {correct}")
    
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, 'sample')
    
    signal_path = os.path.join(sample_dir, 'mlii_signal.csv')
    annotations_path = os.path.join(sample_dir, 'parsed_annotations.csv')
    model_path = os.path.join(sample_dir, 'ecg_lstm_v3_final.keras')
    scaler_path = os.path.join(sample_dir, 'scaler_v3.pkl')
    
    output_path = os.path.join(sample_dir, 'classification_results.csv')
    
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
        if not os.path.exists(signal_path) or not os.path.exists(annotations_path):
            print("Signal/Annotation files not found.")
            print("Please run the data conversion script first:")
            print("  python convert_ecg_data.py")
            print()
            print("Attempting to run conversion automatically...")
            try:
                from convert_ecg_data import main as convert_main
                convert_main()
                print()
            except ImportError as e:
                print(f"Error: Could not import convert_ecg_data: {e}")
                sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found: {scaler_path}")
        sys.exit(1)
    
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
