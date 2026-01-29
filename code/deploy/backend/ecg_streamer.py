import os
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ECGStreamer:
    
    SAMPLING_RATE = 360
    
    DEFAULT_WINDOW_SECONDS = 0.5
    
    def __init__(self, signal_path: Optional[str] = None, annotation_path: Optional[str] = None):
        self.signal: Optional[np.ndarray] = None
        self.annotations: Optional[pd.DataFrame] = None
        self.sampling_rate: int = self.SAMPLING_RATE
        
        self.current_sample: int = 0
        self.speed_multiplier: float = 1.0
        self.is_running: bool = False
        self._start_time: float = 0.0
        self._start_sample: int = 0
        
        self._lock = threading.Lock()
        
        if signal_path and annotation_path:
            self.load_data(signal_path, annotation_path)
    
    def load_data(self, signal_path: str, annotation_path: str) -> None:
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        df = pd.read_csv(signal_path)
        df.columns = df.columns.str.strip().str.strip("'")
        
        if 'MLII' not in df.columns:
            raise ValueError(f"Signal file missing 'MLII' column. Available: {list(df.columns)}")
        
        self.signal = df['MLII'].values.astype(np.float32)
        
        annotations_list = []
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
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
        
        self.reset()
    
    def reset(self) -> None:
        with self._lock:
            self.current_sample = 0
            self.is_running = False
            self._start_time = 0.0
            self._start_sample = 0
    
    def start(self) -> None:
        with self._lock:
            self.is_running = True
            self._start_time = time.time()
            self._start_sample = self.current_sample
    
    def stop(self) -> None:
        with self._lock:
            if self.is_running:
                self._update_position()
            self.is_running = False
    
    def set_speed(self, speed: float) -> None:
        speed = max(0.1, min(100.0, speed))
        
        with self._lock:
            if self.is_running:
                self._update_position()
                self._start_time = time.time()
                self._start_sample = self.current_sample
            self.speed_multiplier = speed
    
    def _update_position(self) -> None:
        if not self.is_running or self.signal is None:
            return
        
        elapsed_time = time.time() - self._start_time
        samples_elapsed = int(elapsed_time * self.sampling_rate * self.speed_multiplier)
        new_sample = self._start_sample + samples_elapsed
        
        self.current_sample = min(new_sample, len(self.signal))
    
    def get_current_position(self) -> Dict:
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
        if self.annotations is None:
            return []
        
        mask = (self.annotations['sample_index'] >= start_sample) & \
               (self.annotations['sample_index'] < end_sample)
        
        return self.annotations[mask].to_dict('records')
    
    def get_full_signal(self) -> np.ndarray:
        return self.signal if self.signal is not None else np.array([])
    
    def get_all_annotations(self) -> pd.DataFrame:
        return self.annotations if self.annotations is not None else pd.DataFrame()
    
    @property
    def total_samples(self) -> int:
        return len(self.signal) if self.signal is not None else 0
    
    @property
    def total_duration(self) -> float:
        return self.total_samples / self.sampling_rate
    
    @property
    def is_loaded(self) -> bool:
        return self.signal is not None and self.annotations is not None
