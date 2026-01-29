import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ClassificationResult:
    r_peak: int
    beat_type: str
    ground_truth: str
    predicted: str
    probability: float
    correct: bool
    time_seconds: float
    beat_waveform: Optional[List[float]] = None
    r_peak_pos_in_beat: int = 70
    beat_length: int = 188


class EvaluationLayer:
    
    NORMAL_BEAT_TYPE = 'N'
    
    SAMPLING_RATE = 360
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.results: List[ClassificationResult] = []
        self.false_detections: List[ClassificationResult] = []
        
        self.recent_beat_times: deque = deque(maxlen=10)
    
    def reset(self) -> None:
        self.results = []
        self.false_detections = []
        self.recent_beat_times.clear()
    
    def get_ground_truth(self, beat_type: str) -> str:
        return 'NORMAL' if beat_type == self.NORMAL_BEAT_TYPE else 'ABNORMAL'
    
    def add_result(self, r_peak: int, beat_type: str, predicted: str, probability: float,
                   beat_waveform: Optional[List[float]] = None,
                   r_peak_pos_in_beat: int = 70,
                   beat_length: int = 188) -> ClassificationResult:
        if predicted == 'WAITING':
            return None
        
        ground_truth = self.get_ground_truth(beat_type)
        correct = (ground_truth == predicted)
        time_seconds = r_peak / self.SAMPLING_RATE
        
        result = ClassificationResult(
            r_peak=r_peak,
            beat_type=beat_type,
            ground_truth=ground_truth,
            predicted=predicted,
            probability=probability,
            correct=correct,
            time_seconds=time_seconds,
            beat_waveform=beat_waveform,
            r_peak_pos_in_beat=r_peak_pos_in_beat,
            beat_length=beat_length
        )
        
        self.results.append(result)
        if len(self.results) > self.max_history:
            self.results.pop(0)
        
        if not correct:
            self.false_detections.append(result)
        
        self.recent_beat_times.append(r_peak)
        
        return result
    
    def calculate_bpm(self) -> Optional[int]:
        if len(self.recent_beat_times) < 2:
            return None
        
        times = list(self.recent_beat_times)
        intervals = []
        
        for i in range(1, len(times)):
            interval = (times[i] - times[i-1]) / self.SAMPLING_RATE
            if 0.3 < interval < 2.0:
                intervals.append(interval)
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        return int(60 / avg_interval)
    
    def get_metrics(self) -> Dict:
        if not self.results:
            return {
                'total': 0,
                'normal_count': 0,
                'abnormal_count': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'bpm': None
            }
        
        total = len(self.results)
        normal_count = sum(1 for r in self.results if r.predicted == 'NORMAL')
        abnormal_count = sum(1 for r in self.results if r.predicted == 'ABNORMAL')
        correct = sum(1 for r in self.results if r.correct)
        incorrect = total - correct
        
        tp = sum(1 for r in self.results if r.predicted == 'ABNORMAL' and r.ground_truth == 'ABNORMAL')
        tn = sum(1 for r in self.results if r.predicted == 'NORMAL' and r.ground_truth == 'NORMAL')
        fp = sum(1 for r in self.results if r.predicted == 'ABNORMAL' and r.ground_truth == 'NORMAL')
        fn = sum(1 for r in self.results if r.predicted == 'NORMAL' and r.ground_truth == 'ABNORMAL')
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1_score = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
        
        return {
            'total': total,
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(accuracy, 2),
            'sensitivity': round(sensitivity, 2),
            'specificity': round(specificity, 2),
            'precision': round(precision, 2),
            'f1_score': round(f1_score, 2),
            'bpm': self.calculate_bpm(),
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        }
    
    def get_recent_results(self, count: int = 50) -> List[Dict]:
        recent = self.results[-count:][::-1]
        return [
            {
                'r_peak': r.r_peak,
                'beat_type': r.beat_type,
                'ground_truth': r.ground_truth,
                'predicted': r.predicted,
                'probability': r.probability,
                'correct': r.correct,
                'time': round(r.time_seconds, 3)
            }
            for r in recent
        ]
    
    def get_false_detections(self, count: int = 50) -> List[Dict]:
        recent = self.false_detections[-count:][::-1]
        return [
            {
                'r_peak': r.r_peak,
                'beat_type': r.beat_type,
                'ground_truth': r.ground_truth,
                'predicted': r.predicted,
                'probability': r.probability,
                'time': round(r.time_seconds, 3)
            }
            for r in recent
        ]
    
    def export_results(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'r_peak': r.r_peak,
                'time_seconds': r.time_seconds,
                'beat_type': r.beat_type,
                'ground_truth': r.ground_truth,
                'predicted': r.predicted,
                'probability': r.probability,
                'correct': r.correct
            }
            for r in self.results
        ])
