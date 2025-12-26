"""
Evaluation Layer Module

Responsible for matching predictions with ground truth annotations
and computing performance metrics. This module handles:
- Matching predicted beats with annotated beats
- Computing classification metrics (accuracy, sensitivity, specificity, etc.)
- Tracking false detections
- Performance statistics over time

This layer is SEPARATE from inference - it only compares predictions
that have already been made against ground truth annotations.

Beat Classification Ground Truth:
- 'N' = Normal (class 0)
- Any other annotation = Abnormal (class 1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ClassificationResult:
    """Single beat classification result with ground truth."""
    r_peak: int
    beat_type: str  # Original annotation (N, V, A, etc.)
    ground_truth: str  # 'NORMAL' or 'ABNORMAL'
    predicted: str  # 'NORMAL' or 'ABNORMAL'
    probability: float
    correct: bool
    time_seconds: float
    beat_waveform: Optional[List[float]] = None
    r_peak_pos_in_beat: int = 70
    beat_length: int = 188


class EvaluationLayer:
    """
    Performance Evaluation Layer for ECG Classification
    
    This class manages ground truth comparisons and performance metrics.
    It maintains a history of all classifications and computes various
    statistics for thesis reporting.
    
    Attributes:
        results: List of all classification results
        false_detections: List of incorrect predictions
        metrics: Current performance metrics
    """
    
    # Normal beat type (MIT-BIH annotation)
    NORMAL_BEAT_TYPE = 'N'
    
    # Sampling rate for time calculations
    SAMPLING_RATE = 360
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize the Evaluation Layer.
        
        Args:
            max_history: Maximum number of results to keep in history
        """
        self.max_history = max_history
        self.results: List[ClassificationResult] = []
        self.false_detections: List[ClassificationResult] = []
        
        # BPM calculation
        self.recent_beat_times: deque = deque(maxlen=10)
    
    def reset(self) -> None:
        """Reset all evaluation state."""
        self.results = []
        self.false_detections = []
        self.recent_beat_times.clear()
    
    def get_ground_truth(self, beat_type: str) -> str:
        """
        Convert annotation to ground truth label.
        
        Args:
            beat_type: Original MIT-BIH annotation (N, V, A, etc.)
            
        Returns:
            'NORMAL' if beat_type is 'N', otherwise 'ABNORMAL'
        """
        return 'NORMAL' if beat_type == self.NORMAL_BEAT_TYPE else 'ABNORMAL'
    
    def add_result(self, r_peak: int, beat_type: str, predicted: str, probability: float,
                   beat_waveform: Optional[List[float]] = None,
                   r_peak_pos_in_beat: int = 70,
                   beat_length: int = 188) -> ClassificationResult:
        """
        Add a classification result and compare with ground truth.
        
        Args:
            r_peak: R-peak sample index
            beat_type: Original annotation (N, V, A, etc.)
            predicted: Model prediction ('NORMAL' or 'ABNORMAL')
            probability: Abnormal probability (0.0 to 1.0)
            beat_waveform: Raw beat samples (optional, for visualization)
            r_peak_pos_in_beat: Position of R-peak in beat
            beat_length: Length of beat waveform
            
        Returns:
            ClassificationResult with evaluation, or None if prediction is 'WAITING'
            (context-aware model buffer not full yet)
        """
        # Skip 'WAITING' predictions (context-aware model not ready)
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
        
        # Add to history
        self.results.append(result)
        if len(self.results) > self.max_history:
            self.results.pop(0)
        
        # Track false detections
        if not correct:
            self.false_detections.append(result)
        
        # Update BPM calculation
        self.recent_beat_times.append(r_peak)
        
        return result
    
    def calculate_bpm(self) -> Optional[int]:
        """
        Calculate heart rate from recent beats.
        
        Returns:
            BPM value or None if not enough data
        """
        if len(self.recent_beat_times) < 2:
            return None
        
        times = list(self.recent_beat_times)
        intervals = []
        
        for i in range(1, len(times)):
            interval = (times[i] - times[i-1]) / self.SAMPLING_RATE
            # Only count reasonable intervals (30-200 BPM range)
            if 0.3 < interval < 2.0:
                intervals.append(interval)
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        return int(60 / avg_interval)
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dict with:
                - total: Total classified beats
                - normal_count: Predicted normal count
                - abnormal_count: Predicted abnormal count
                - correct: Correct predictions
                - incorrect: Incorrect predictions
                - accuracy: Overall accuracy (%)
                - sensitivity: True positive rate for abnormal
                - specificity: True negative rate for abnormal
                - precision: Positive predictive value for abnormal
                - f1_score: F1 score for abnormal class
                - bpm: Current heart rate estimate
        """
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
        
        # Count predictions
        total = len(self.results)
        normal_count = sum(1 for r in self.results if r.predicted == 'NORMAL')
        abnormal_count = sum(1 for r in self.results if r.predicted == 'ABNORMAL')
        correct = sum(1 for r in self.results if r.correct)
        incorrect = total - correct
        
        # Confusion matrix components
        # True Positives: Predicted abnormal AND actually abnormal
        tp = sum(1 for r in self.results if r.predicted == 'ABNORMAL' and r.ground_truth == 'ABNORMAL')
        # True Negatives: Predicted normal AND actually normal
        tn = sum(1 for r in self.results if r.predicted == 'NORMAL' and r.ground_truth == 'NORMAL')
        # False Positives: Predicted abnormal BUT actually normal
        fp = sum(1 for r in self.results if r.predicted == 'ABNORMAL' and r.ground_truth == 'NORMAL')
        # False Negatives: Predicted normal BUT actually abnormal
        fn = sum(1 for r in self.results if r.predicted == 'NORMAL' and r.ground_truth == 'ABNORMAL')
        
        # Calculate metrics
        accuracy = (correct / total * 100) if total > 0 else 0.0
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0  # Recall for abnormal
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0  # True negative rate
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
        """
        Get recent classification results.
        
        Args:
            count: Number of recent results to return
            
        Returns:
            List of result dicts (most recent first)
        """
        recent = self.results[-count:][::-1]  # Reverse for most recent first
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
        """
        Get recent false detections.
        
        Args:
            count: Number of false detections to return
            
        Returns:
            List of false detection dicts (most recent first)
        """
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
        """
        Export all results as a pandas DataFrame.
        
        Returns:
            DataFrame with all classification results
        """
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
