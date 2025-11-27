"""
Evaluation module for ECG arrhythmia classification.
"""

from .evaluate_robustness import (
    evaluate_robustness,
    plot_robustness_curves,
    plot_confusion_matrices,
    generate_summary_table,
    evaluate_model_suite,
)

__all__ = [
    "evaluate_robustness",
    "plot_robustness_curves",
    "plot_confusion_matrices",
    "generate_summary_table",
    "evaluate_model_suite",
]
