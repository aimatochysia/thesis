"""
ECG Data Processor

This script processes ECG data following the methodology from ecg2hrv.ipynb:
1. Loads raw ECG data (or generates synthetic MIT-BIH-like data)
2. Applies preprocessing: detrending, bandpass filtering
3. Detects R-peaks using differentiation, squaring, and smoothing
4. Segments individual heartbeats around R-peaks
5. Resamples each beat to 188 samples (matching the model input)
6. Outputs CSV with timeframes and beat amplitudes

Output format:
- Each row represents one heartbeat segment
- Column 1: timeframe (time in seconds when the R-peak occurred)
- Columns 2-189: amplitude values for the heartbeat (188 samples)

Usage:
    python ecg_data_processor.py --num_records 5 --duration_minutes 10 --output output.csv
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy.signal import butter, sosfilt
import argparse

# Default parameters matching ecg2hrv.ipynb
DEFAULT_FS = 360  # Sampling frequency (Hz)
DEFAULT_DURATION_MINUTES = 10  # Duration per record
DEFAULT_NUM_RECORDS = 5  # Number of records to process
DEFAULT_OUTPUT_LEN = 188  # Target samples per beat

# Preprocessing parameters from ecg2hrv.ipynb
BANDPASS_LOW = 0.4  # Hz - removes low frequency noise
BANDPASS_HIGH = 40  # Hz - removes high frequency noise
FILTER_ORDER = 5  # Butterworth filter order


def butter_bandpass_sos(lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """Create Butterworth bandpass filter coefficients."""
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Simple detrending by removing linear trend.
    Approximates the effect of nk.signal_detrend().
    """
    n = len(signal)
    x = np.arange(n)
    # Fit linear regression
    coeffs = np.polyfit(x, signal, 1)
    trend = np.polyval(coeffs, x)
    return signal - trend


def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float = BANDPASS_LOW,
                   highcut: float = BANDPASS_HIGH, order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply bandpass filter to ECG signal.
    """
    sos = butter_bandpass_sos(lowcut, highcut, fs, order)
    filtered = sosfilt(sos, signal)
    return filtered


def preprocess_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Full preprocessing pipeline following ecg2hrv.ipynb:
    1. Detrending
    2. Bandpass filtering (0.4-40 Hz)
    """
    # Detrend the signal
    detrended = detrend_signal(signal)
    # Apply bandpass filter
    filtered = bandpass_filter(detrended, fs)
    return filtered


def smooth_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving window smoothing (convolution).
    """
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, window, mode='same')
    return smoothed


def detect_r_peaks(ecg_cleaned: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect R-peaks using the method from ecg2hrv.ipynb:
    1. Differentiate
    2. Square
    3. Smooth (moving window integration)
    4. Threshold-based peak detection
    """
    # Differentiation
    ecg_diff = np.gradient(ecg_cleaned)
    
    # Squaring
    ecg_squared = ecg_diff ** 2
    
    # Smoothing (Moving-Window Integration)
    window_size = int(0.1 * fs)  # 100 ms window
    ecg_smoothed = smooth_signal(ecg_squared, window_size)
    
    # Threshold-based peak detection
    threshold = np.median(ecg_smoothed) + 2.5 * np.std(ecg_smoothed)
    
    # Find peaks above threshold with minimum distance (refractory period)
    min_distance = int(0.2 * fs)  # 200 ms refractory period
    
    peaks = []
    i = 0
    while i < len(ecg_smoothed):
        if ecg_smoothed[i] > threshold:
            # Find local maximum in a small window
            start = i
            end = min(i + min_distance, len(ecg_smoothed))
            local_max_idx = start + np.argmax(ecg_smoothed[start:end])
            peaks.append(local_max_idx)
            i = local_max_idx + min_distance
        else:
            i += 1
    
    return np.array(peaks)


def resample_beat(beat: np.ndarray, target_len: int = DEFAULT_OUTPUT_LEN) -> np.ndarray:
    """
    Resample beat to target length using linear interpolation.
    """
    if len(beat) == target_len:
        return beat.astype(np.float32)
    xi = np.linspace(0, 1, num=len(beat), endpoint=True)
    xo = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(xo, xi, beat).astype(np.float32)


def segment_beats(ecg_cleaned: np.ndarray, r_peaks: np.ndarray, fs: int,
                  pre_sec: float = 0.25, post_sec: float = 0.45) -> List[Dict]:
    """
    Segment individual heartbeats around R-peaks.
    
    Args:
        ecg_cleaned: Preprocessed ECG signal
        r_peaks: Array of R-peak indices
        fs: Sampling frequency
        pre_sec: Seconds before R-peak to include
        post_sec: Seconds after R-peak to include
    
    Returns:
        List of dicts with 'timeframe' and 'beat' keys
    """
    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)
    
    beats = []
    for r_idx in r_peaks:
        start = r_idx - pre_samples
        end = r_idx + post_samples
        
        # Skip if segment goes out of bounds
        if start < 0 or end > len(ecg_cleaned):
            continue
        
        beat_segment = ecg_cleaned[start:end]
        timeframe = r_idx / fs  # Time in seconds
        
        beats.append({
            'timeframe': timeframe,
            'beat': beat_segment
        })
    
    return beats


def generate_synthetic_ecg(duration_sec: int, fs: int, heart_rate_bpm: float = 75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG signal with known R-peak locations.
    This simulates MIT-BIH-style data for testing.
    
    Returns:
        Tuple of (ecg_signal, annotation_indices)
    """
    num_samples = int(duration_sec * fs)
    t = np.arange(num_samples) / fs
    
    # Calculate beat interval
    beat_interval = 60.0 / heart_rate_bpm  # seconds per beat
    
    # Add some heart rate variability
    rr_variability = 0.05  # 5% variability
    
    # Generate ECG signal
    ecg = np.zeros(num_samples)
    annotations = []
    
    # Baseline around 950 (as mentioned in README)
    baseline = 950
    
    current_time = 0.5  # Start after a small delay
    while current_time < duration_sec - 1:
        # Add some RR variability
        current_rr = beat_interval * (1 + np.random.uniform(-rr_variability, rr_variability))
        
        r_idx = int(current_time * fs)
        if r_idx >= num_samples:
            break
        
        annotations.append(r_idx)
        
        # Create a single heartbeat (simplified PQRST complex)
        # P-wave
        p_start = max(0, r_idx - int(0.2 * fs))
        p_end = max(0, r_idx - int(0.12 * fs))
        for i in range(p_start, min(p_end, num_samples)):
            offset = i - p_start
            width = p_end - p_start
            if width > 0:
                ecg[i] += 15 * np.sin(np.pi * offset / width)
        
        # Q-wave
        q_start = max(0, r_idx - int(0.04 * fs))
        for i in range(q_start, min(r_idx, num_samples)):
            offset = i - q_start
            width = r_idx - q_start
            if width > 0:
                ecg[i] -= 20 * (offset / width)
        
        # R-peak
        if r_idx < num_samples:
            ecg[r_idx] += 250  # Strong R-peak
        
        # S-wave
        s_end = min(num_samples, r_idx + int(0.04 * fs))
        for i in range(r_idx + 1, s_end):
            offset = i - r_idx
            width = s_end - r_idx
            if width > 0:
                ecg[i] -= 40 * (1 - offset / width)
        
        # T-wave
        t_start = r_idx + int(0.1 * fs)
        t_end = min(num_samples, r_idx + int(0.35 * fs))
        for i in range(t_start, t_end):
            offset = i - t_start
            width = t_end - t_start
            if width > 0:
                ecg[i] += 30 * np.sin(np.pi * offset / width)
        
        current_time += current_rr
    
    # Add baseline and noise
    ecg += baseline
    ecg += np.random.normal(0, 2, num_samples)  # Small noise
    
    return ecg, np.array(annotations)


def process_ecg_record(ecg_raw: np.ndarray, fs: int = DEFAULT_FS,
                       output_len: int = DEFAULT_OUTPUT_LEN) -> pd.DataFrame:
    """
    Process a single ECG record and return a DataFrame with heartbeat segments.
    
    Args:
        ecg_raw: Raw ECG signal
        fs: Sampling frequency
        output_len: Number of samples per output beat
    
    Returns:
        DataFrame with columns: timeframe, amp_1, amp_2, ..., amp_188
    """
    # Preprocess
    ecg_cleaned = preprocess_ecg(ecg_raw, fs)
    
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_cleaned, fs)
    
    # Segment beats
    beats = segment_beats(ecg_cleaned, r_peaks, fs)
    
    # Build DataFrame
    rows = []
    for beat_data in beats:
        timeframe = beat_data['timeframe']
        beat = beat_data['beat']
        
        # Resample to target length
        resampled = resample_beat(beat, output_len)
        
        # Create row with timeframe and amplitude values
        row = {'timeframe': timeframe}
        for i, amp in enumerate(resampled):
            row[f'amp_{i+1}'] = amp
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_and_process_records(num_records: int = DEFAULT_NUM_RECORDS,
                                 duration_minutes: int = DEFAULT_DURATION_MINUTES,
                                 fs: int = DEFAULT_FS,
                                 output_len: int = DEFAULT_OUTPUT_LEN) -> pd.DataFrame:
    """
    Generate multiple synthetic ECG records and process them.
    
    Args:
        num_records: Number of records to generate
        duration_minutes: Duration of each record in minutes
        fs: Sampling frequency
        output_len: Number of samples per output beat
    
    Returns:
        Combined DataFrame with all heartbeat segments
    """
    duration_sec = duration_minutes * 60
    all_dfs = []
    
    # MIT-BIH record numbers (using some common ones for naming)
    record_ids = [100, 101, 102, 103, 104][:num_records]
    
    for i, record_id in enumerate(record_ids):
        print(f"Processing record {record_id} ({i+1}/{num_records})...")
        
        # Vary heart rate slightly for each record
        heart_rate = 70 + np.random.uniform(-10, 15)
        
        # Generate synthetic ECG
        ecg_raw, annotations = generate_synthetic_ecg(duration_sec, fs, heart_rate)
        
        print(f"  Generated {len(annotations)} reference annotations")
        
        # Process the record
        df = process_ecg_record(ecg_raw, fs, output_len)
        
        # Add record identifier
        df['record_id'] = record_id
        
        print(f"  Extracted {len(df)} heartbeat segments")
        
        all_dfs.append(df)
    
    # Combine all records
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="ECG Data Processor")
    parser.add_argument("--num_records", type=int, default=DEFAULT_NUM_RECORDS,
                        help=f"Number of ECG records to process (default: {DEFAULT_NUM_RECORDS})")
    parser.add_argument("--duration_minutes", type=int, default=DEFAULT_DURATION_MINUTES,
                        help=f"Duration of each record in minutes (default: {DEFAULT_DURATION_MINUTES})")
    parser.add_argument("--fs", type=int, default=DEFAULT_FS,
                        help=f"Sampling frequency in Hz (default: {DEFAULT_FS})")
    parser.add_argument("--output", type=str, default="processed_ecg_data.csv",
                        help="Output CSV file path (default: processed_ecg_data.csv)")
    parser.add_argument("--output_len", type=int, default=DEFAULT_OUTPUT_LEN,
                        help=f"Number of samples per beat (default: {DEFAULT_OUTPUT_LEN})")
    
    args = parser.parse_args()
    
    print(f"ECG Data Processor")
    print(f"==================")
    print(f"Number of records: {args.num_records}")
    print(f"Duration per record: {args.duration_minutes} minutes")
    print(f"Sampling frequency: {args.fs} Hz")
    print(f"Output file: {args.output}")
    print()
    
    # Generate and process records
    df = generate_and_process_records(
        num_records=args.num_records,
        duration_minutes=args.duration_minutes,
        fs=args.fs,
        output_len=args.output_len
    )
    
    # Reorder columns: timeframe first, then amplitude columns
    amp_cols = [f'amp_{i+1}' for i in range(args.output_len)]
    cols = ['timeframe'] + amp_cols
    if 'record_id' in df.columns:
        cols = ['record_id'] + cols
    
    df = df[cols]
    
    # Save to CSV
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total heartbeat segments: {len(df)}")
    print(f"Output saved to: {args.output}")
    
    # Print sample of the output
    print(f"\nSample of output (first 5 rows, first few columns):")
    print(df.iloc[:5, :6])


if __name__ == "__main__":
    main()
