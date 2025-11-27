"""
ECG Data Preprocessing Module

Implements consistent beat segmentation pipeline for deployment:
- RR-adaptive windowing (pre_frac=0.35, post_frac=0.65) with clamps
- Resampling to 188 samples
- Normalization (baseline_shift_scale)

Supports CSV and PhysioNet/MIT-BIH record loading, patient-wise splits.
"""

import os
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.signal import resample


# Default preprocessing parameters
DEFAULT_CONFIG = {
    # RR-adaptive window fractions
    "pre_frac": 0.35,
    "post_frac": 0.65,
    # Window clamps in seconds
    "pre_min_sec": 0.08,
    "pre_max_sec": 0.35,
    "post_min_sec": 0.16,
    "post_max_sec": 0.60,
    # Model input length
    "model_input_len": 188,
    # Normalization
    "target_baseline": 950.0,
    "global_scale": 100.0,
    # Sampling rate
    "fs": 360,
}


def load_csv_data(
    csv_path: str,
    label_column: str = "label",
    feature_prefix: str = "f",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG beat data from a CSV file.

    Args:
        csv_path: Path to CSV file
        label_column: Name of the label column
        feature_prefix: Prefix for feature columns (e.g., 'f' for f0, f1, ...)

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    df = pd.read_csv(csv_path)
    
    # Handle case where CSV has no header (188 features + label)
    if df.columns[0] == '0' or df.shape[1] == 189:
        # No header, assume last column is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif label_column in df.columns:
        X = df.drop(columns=[label_column]).values
        y = df[label_column].values
    else:
        # Try to find feature columns
        feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
        if feature_cols:
            X = df[feature_cols].values
            y = df[label_column].values if label_column in df.columns else np.zeros(len(df))
        else:
            # Assume all columns except last are features
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
    
    return X.astype(np.float32), y.astype(np.int32)


def load_physionet_record(
    record_path: str,
    annotation_ext: str = "atr",
) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
    """
    Load a PhysioNet/MIT-BIH record using wfdb.

    Args:
        record_path: Path to the record (without extension)
        annotation_ext: Annotation extension (default: 'atr')

    Returns:
        Tuple of (signal, labels, r_peak_indices, sampling_rate)
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required for PhysioNet records. Install with: pip install wfdb")
    
    # Read record
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # Use first channel
    fs = record.fs
    
    # Read annotations
    ann = wfdb.rdann(record_path, annotation_ext)
    r_peaks = ann.sample
    labels = ann.symbol
    
    # Map annotation symbols to binary (N = normal, others = abnormal)
    normal_symbols = {'N', 'L', 'R', 'e', 'j'}
    binary_labels = np.array([0 if s in normal_symbols else 1 for s in labels])
    
    return signal.astype(np.float32), binary_labels.astype(np.int32), r_peaks.tolist(), fs


def rr_adaptive_window(
    r_index: int,
    rr_history: List[int],
    signal_len: int,
    fs: int,
    config: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Compute RR-adaptive window bounds for beat segmentation.

    Args:
        r_index: Index of the R-peak
        rr_history: List of recent RR intervals (in samples)
        signal_len: Total signal length
        fs: Sampling rate
        config: Configuration dict (uses DEFAULT_CONFIG if None)

    Returns:
        Tuple of (start_index, end_index) for the window
    """
    cfg = config or DEFAULT_CONFIG
    
    # Get median RR interval
    if rr_history:
        rr = int(np.median(rr_history))
    else:
        rr = int(0.8 * fs)  # Default ~0.8 s
    
    # Compute window sizes
    pre = int(round(cfg["pre_frac"] * rr))
    post = int(round(cfg["post_frac"] * rr))
    
    # Apply clamps
    pre_min = int(round(cfg["pre_min_sec"] * fs))
    pre_max = int(round(cfg["pre_max_sec"] * fs))
    post_min = int(round(cfg["post_min_sec"] * fs))
    post_max = int(round(cfg["post_max_sec"] * fs))
    
    pre = max(pre_min, min(pre_max, pre))
    post = max(post_min, min(post_max, post))
    
    # Compute bounds with clipping
    start = max(0, r_index - pre)
    end = min(signal_len, r_index + post)
    
    return start, end


def resample_beat(
    beat: np.ndarray,
    target_len: int = 188,
) -> np.ndarray:
    """
    Resample a beat to target length using linear interpolation.

    Args:
        beat: Input beat signal
        target_len: Target length (default: 188)

    Returns:
        Resampled beat
    """
    if len(beat) == target_len:
        return beat.astype(np.float32)
    
    # Use scipy's resample for quality
    resampled = resample(beat, target_len)
    return resampled.astype(np.float32)


def normalize_beat(
    beat: np.ndarray,
    mode: str = "baseline_shift_scale",
    target_baseline: float = 950.0,
    global_scale: float = 100.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize beat signal.

    Args:
        beat: Input beat signal
        mode: Normalization mode
            - "baseline_shift_scale": (x - target_baseline) / global_scale
            - "per_beat_standardize": (x - mean) / std
            - "none": No normalization
        target_baseline: Target baseline value (default: 950.0)
        global_scale: Scale divisor (default: 100.0)
        eps: Small value to prevent division by zero

    Returns:
        Normalized beat
    """
    if mode == "baseline_shift_scale":
        return (beat - target_baseline) / global_scale
    elif mode == "per_beat_standardize":
        m = float(np.mean(beat))
        s = float(np.std(beat))
        s = s if s > eps else 1.0
        return (beat - m) / s
    else:
        return beat.astype(np.float32)


def beat_segmentation(
    signal: np.ndarray,
    r_peaks: List[int],
    fs: int,
    config: Optional[Dict] = None,
    normalize: bool = True,
    norm_mode: str = "baseline_shift_scale",
) -> List[np.ndarray]:
    """
    Segment continuous ECG signal into individual beats.

    Args:
        signal: Continuous ECG signal
        r_peaks: List of R-peak indices
        fs: Sampling rate
        config: Configuration dict
        normalize: Whether to normalize beats
        norm_mode: Normalization mode

    Returns:
        List of segmented, resampled, and optionally normalized beats
    """
    cfg = config or DEFAULT_CONFIG
    target_len = cfg.get("model_input_len", 188)
    
    beats = []
    rr_history = []
    
    for i, r_idx in enumerate(r_peaks):
        # Update RR history
        if i > 0:
            rr = r_idx - r_peaks[i - 1]
            rr_history.append(rr)
            if len(rr_history) > 8:
                rr_history.pop(0)
        
        # Get window bounds
        start, end = rr_adaptive_window(r_idx, rr_history, len(signal), fs, cfg)
        
        # Extract and resample beat
        beat = signal[start:end]
        beat = resample_beat(beat, target_len)
        
        # Normalize if requested
        if normalize:
            beat = normalize_beat(
                beat,
                mode=norm_mode,
                target_baseline=cfg.get("target_baseline", 950.0),
                global_scale=cfg.get("global_scale", 100.0),
            )
        
        beats.append(beat)
    
    return beats


def patient_wise_split(
    patient_ids: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by patient IDs to ensure no data leakage.

    Args:
        patient_ids: Array of patient IDs for each sample
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training)
        random_state: Random seed

    Returns:
        Tuple of (train_mask, val_mask, test_mask) boolean arrays
    """
    np.random.seed(random_state)
    
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # Shuffle patients
    shuffled_idx = np.random.permutation(n_patients)
    shuffled_patients = unique_patients[shuffled_idx]
    
    # Split patients
    n_test = max(1, int(n_patients * test_size))
    n_val = max(1, int(n_patients * val_size))
    
    test_patients = set(shuffled_patients[:n_test])
    val_patients = set(shuffled_patients[n_test:n_test + n_val])
    train_patients = set(shuffled_patients[n_test + n_val:])
    
    # Create masks
    train_mask = np.array([p in train_patients for p in patient_ids])
    val_mask = np.array([p in val_patients for p in patient_ids])
    test_mask = np.array([p in test_patients for p in patient_ids])
    
    return train_mask, val_mask, test_mask


class ECGDataLoader:
    """
    Data loader for ECG beat classification.
    Supports loading from CSV files and PhysioNet records.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        normalize: bool = True,
        norm_mode: str = "baseline_shift_scale",
    ):
        """
        Initialize data loader.

        Args:
            config: Configuration dict
            normalize: Whether to normalize beats
            norm_mode: Normalization mode
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.normalize = normalize
        self.norm_mode = norm_mode
    
    def load_csv_beats(
        self,
        csv_paths: Union[str, List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-segmented beats from CSV files.

        Args:
            csv_paths: Path or list of paths to CSV files

        Returns:
            Tuple of (X, y) arrays
        """
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        all_X = []
        all_y = []
        
        for path in csv_paths:
            X, y = load_csv_data(path)
            all_X.append(X)
            all_y.append(y)
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Normalize if requested
        if self.normalize:
            X_norm = np.zeros_like(X)
            for i in range(len(X)):
                X_norm[i] = normalize_beat(
                    X[i],
                    mode=self.norm_mode,
                    target_baseline=self.config.get("target_baseline", 950.0),
                    global_scale=self.config.get("global_scale", 100.0),
                )
            X = X_norm
        
        return X, y
    
    def load_continuous_signal(
        self,
        csv_path: str,
        ecg_column: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load a continuous ECG signal from CSV.

        Args:
            csv_path: Path to CSV file
            ecg_column: Column name for ECG values

        Returns:
            Signal array
        """
        df = pd.read_csv(csv_path)
        
        if ecg_column and ecg_column in df.columns:
            signal = df[ecg_column].values
        else:
            # Pick first numeric column
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    signal = df[col].values
                    break
            else:
                raise ValueError("No numeric column found in CSV")
        
        return signal.astype(np.float32)
    
    def prepare_for_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        patient_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for training with train/val/test splits.

        Args:
            X: Feature array
            y: Label array
            test_size: Test set fraction
            val_size: Validation set fraction
            random_state: Random seed
            patient_ids: Optional patient IDs for patient-wise split

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if patient_ids is not None:
            # Patient-wise split
            train_mask, val_mask, test_mask = patient_wise_split(
                patient_ids, test_size, val_size, random_state
            )
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
        else:
            # Random split with stratification
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            val_frac = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
            )
        
        # Reshape for Conv1D: (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }
