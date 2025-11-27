"""
Training module for ECG arrhythmia classification.
"""

from .train_teacher_v2_robust import (
    create_v2_cnn_model,
    TemporalShiftAugmentation,
    TimeWarpAugmentation,
    AmplitudeAugmentation,
    ConsistencyRegularization,
    RobustTrainer,
    train_teacher_v2_robust,
)

from .train_student_distill import (
    create_student_model,
    DistillationLoss,
    StudentTrainer,
    train_student_distill,
)

__all__ = [
    "create_v2_cnn_model",
    "TemporalShiftAugmentation",
    "TimeWarpAugmentation",
    "AmplitudeAugmentation",
    "ConsistencyRegularization",
    "RobustTrainer",
    "train_teacher_v2_robust",
    "create_student_model",
    "DistillationLoss",
    "StudentTrainer",
    "train_student_distill",
]
