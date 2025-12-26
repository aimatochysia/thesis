"""
ECG Streamer Module

Responsible for loading and streaming ECG signal data with simulated real-time playback.
This module handles:
- Loading ECG CSV files (MIT-BIH format)
- Parsing annotation files
- Simulating real-time signal streaming at configurable speeds
- Maintaining playback state (current position, speed)

The streamer does NOT perform any inference or classification - it purely handles
signal delivery. Annotations are loaded for ground truth evaluation but are NOT
exposed during inference to maintain separation of concerns.

MIT-BIH Format:
- Signal files: XXX.csv with columns including 'MLII' (lead signal)
- Annotation files: XXXannotations.txt with sample_index, beat_type
- Sampling rate: 360 Hz (standard MIT-BIH)
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ECGStreamer:
    """
    ECG Signal Streamer with Simulated Real-Time Playback
    
    This class manages ECG signal streaming, providing a clean interface for
    both batch and real-time data access. It supports:
    - Loading from MIT-BIH CSV format
    - Configurable playback speed (0.1x to 10x)
    - Window-based data retrieval
    - Thread-safe state management
    
    Attributes:
        signal (np.ndarray): Full ECG signal array
        annotations (pd.DataFrame): Beat annotations with sample_index and beat_type
        sampling_rate (int): Signal sampling rate in Hz
        current_sample (int): Current playback position
        speed_multiplier (float): Playback speed (1.0 = real-time)
        is_running (bool): Whether streaming is active
    """
    
    # Standard MIT-BIH sampling rate
    SAMPLING_RATE = 360  # Hz
    
    # Default window size for streaming (in seconds)
    DEFAULT_WINDOW_SECONDS = 0.5
    
    def __init__(self, signal_path: Optional[str] = None, annotation_path: Optional[str] = None):
        """
        Initialize the ECG Streamer.
        
        Args:
            signal_path: Path to ECG signal CSV file. If None, must call load_data() later.
            annotation_path: Path to annotations text file. If None, must call load_data() later.
        """
        self.signal: Optional[np.ndarray] = None
        self.annotations: Optional[pd.DataFrame] = None
        self.sampling_rate: int = self.SAMPLING_RATE
        
        # Playback state
        self.current_sample: int = 0
        self.speed_multiplier: float = 1.0
        self.is_running: bool = False
        self._start_time: float = 0.0
        self._start_sample: int = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load data if paths provided
        if signal_path and annotation_path:
            self.load_data(signal_path, annotation_path)
    
    def load_data(self, signal_path: str, annotation_path: str) -> None:
        """
        Load ECG signal and annotations from files.
        
        Args:
            signal_path: Path to ECG signal CSV file (MIT-BIH format)
            annotation_path: Path to annotations text file
            
        Raises:
            FileNotFoundError: If either file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        # Load signal
        df = pd.read_csv(signal_path)
        df.columns = df.columns.str.strip().str.strip("'")
        
        if 'MLII' not in df.columns:
            raise ValueError(f"Signal file missing 'MLII' column. Available: {list(df.columns)}")
        
        self.signal = df['MLII'].values.astype(np.float32)
        
        # Load annotations
        annotations_list = []
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    sample_idx = int(parts[1])
                    beat_type = parts[2]
                    time_str = parts[0]
                    annotations_list.append({
                        'sample_index': sample_idx,
                        'beat_type': beat_type,
                        'time': time_str
                    })
                except (ValueError, IndexError):
                    continue
        
        self.annotations = pd.DataFrame(annotations_list)
        
        # Reset playback state
        self.reset()
    
    def reset(self) -> None:
        """Reset playback to the beginning."""
        with self._lock:
            self.current_sample = 0
            self.is_running = False
            self._start_time = 0.0
            self._start_sample = 0
    
    def start(self) -> None:
        """Start real-time playback simulation."""
        with self._lock:
            self.is_running = True
            self._start_time = time.time()
            self._start_sample = self.current_sample
    
    def stop(self) -> None:
        """Stop real-time playback simulation."""
        with self._lock:
            if self.is_running:
                # Update current sample before stopping
                self._update_position()
            self.is_running = False
    
    def set_speed(self, speed: float) -> None:
        """
        Set playback speed multiplier.
        
        Args:
            speed: Speed multiplier (0.1 = 10x slower, 10.0 = 10x faster)
        """
        speed = max(0.1, min(100.0, speed))  # Clamp to reasonable range
        
        with self._lock:
            if self.is_running:
                # Update position before changing speed
                self._update_position()
                self._start_time = time.time()
                self._start_sample = self.current_sample
            self.speed_multiplier = speed
    
    def _update_position(self) -> None:
        """Update current sample based on elapsed time (internal, call with lock held)."""
        if not self.is_running or self.signal is None:
            return
        
        elapsed_time = time.time() - self._start_time
        samples_elapsed = int(elapsed_time * self.sampling_rate * self.speed_multiplier)
        new_sample = self._start_sample + samples_elapsed
        
        # Clamp to signal bounds
        self.current_sample = min(new_sample, len(self.signal))
    
    def get_current_position(self) -> Dict:
        """
        Get current playback position.
        
        Returns:
            Dict with:
                - absolute_index: Current sample index
                - time_seconds: Current time in seconds
                - total_samples: Total samples in signal
                - total_seconds: Total signal duration
                - progress: Playback progress (0.0 to 1.0)
        """
        with self._lock:
            if self.is_running:
                self._update_position()
            
            total = len(self.signal) if self.signal is not None else 0
            return {
                'absolute_index': self.current_sample,
                'time_seconds': self.current_sample / self.sampling_rate,
                'total_samples': total,
                'total_seconds': total / self.sampling_rate if total > 0 else 0,
                'progress': self.current_sample / total if total > 0 else 0
            }
    
    def get_window(self, window_samples: Optional[int] = None, end_sample: Optional[int] = None) -> Dict:
        """
        Get a window of ECG samples.
        
        Args:
            window_samples: Number of samples to return. If None, uses default (0.5s worth)
            end_sample: End position of window. If None, uses current position
            
        Returns:
            Dict with:
                - samples: List of signal values
                - timestamps: List of timestamps (in seconds)
                - start_index: First sample index (absolute)
                - end_index: Last sample index (absolute)
                - sampling_rate: Signal sampling rate
        """
        if self.signal is None:
            return {
                'samples': [],
                'timestamps': [],
                'start_index': 0,
                'end_index': 0,
                'sampling_rate': self.sampling_rate
            }
        
        with self._lock:
            if self.is_running:
                self._update_position()
            
            if window_samples is None:
                window_samples = int(self.DEFAULT_WINDOW_SECONDS * self.sampling_rate)
            
            if end_sample is None:
                end_sample = self.current_sample
            
            start_sample = max(0, end_sample - window_samples)
            end_sample = min(end_sample, len(self.signal))
            
            samples = self.signal[start_sample:end_sample].tolist()
            timestamps = [i / self.sampling_rate for i in range(start_sample, end_sample)]
            
            return {
                'samples': samples,
                'timestamps': timestamps,
                'start_index': start_sample,
                'end_index': end_sample,
                'sampling_rate': self.sampling_rate
            }
    
    def get_annotations_in_range(self, start_sample: int, end_sample: int) -> List[Dict]:
        """
        Get annotations within a sample range.
        
        Note: This method is for EVALUATION ONLY, not inference.
        The inference engine should not have access to annotations during prediction.
        
        Args:
            start_sample: Start of range (inclusive)
            end_sample: End of range (exclusive)
            
        Returns:
            List of annotation dicts with sample_index, beat_type, time
        """
        if self.annotations is None:
            return []
        
        mask = (self.annotations['sample_index'] >= start_sample) & \
               (self.annotations['sample_index'] < end_sample)
        
        return self.annotations[mask].to_dict('records')
    
    def get_full_signal(self) -> np.ndarray:
        """
        Get the complete ECG signal array.
        
        Returns:
            Full signal as numpy array
        """
        return self.signal if self.signal is not None else np.array([])
    
    def get_all_annotations(self) -> pd.DataFrame:
        """
        Get all annotations as DataFrame.
        
        Returns:
            DataFrame with columns: sample_index, beat_type, time
        """
        return self.annotations if self.annotations is not None else pd.DataFrame()
    
    @property
    def total_samples(self) -> int:
        """Total number of samples in signal."""
        return len(self.signal) if self.signal is not None else 0
    
    @property
    def total_duration(self) -> float:
        """Total signal duration in seconds."""
        return self.total_samples / self.sampling_rate
    
    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded."""
        return self.signal is not None and self.annotations is not None
