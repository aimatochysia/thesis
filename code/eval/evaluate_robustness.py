#!/usr/bin/env python
"""
Robustness Evaluation Script

Generate robustness curves, confusion matrices, and summary tables for
ECG classification models.

Usage:
    python evaluate_robustness.py --model_path model.h5 --data_path ecg.csv \
        --output_dir outputs/plots

Features:
- Robustness curves: performance vs temporal shifts
- Confusion matrices at different shift levels
- Summary tables with F1, AUC, accuracy
- Parameter count and inference time metrics
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import shift as scipy_shift

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, roc_curve, auc
)


# Default configuration
DEFAULT_CONFIG = {
    "input_len": 188,
    "fs": 360,
    "shifts_ms": [-40, -30, -20, -10, 0, 10, 20, 30, 40],
    "random_state": 42,
}


def load_model(model_path: str) -> tf.keras.Model:
    """Load a Keras model from path."""
    return tf.keras.models.load_model(model_path)


def load_test_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data from CSV."""
    df = pd.read_csv(data_path, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    # Split to get test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Reshape for Conv1D
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_test, y_test


def apply_temporal_shift(
    X: np.ndarray,
    shift_ms: int,
    fs: int = 360,
    input_len: int = 188,
) -> np.ndarray:
    """
    Apply temporal shift to ECG data.

    Args:
        X: Input data (samples, timesteps, features)
        shift_ms: Shift amount in milliseconds
        fs: Sampling rate
        input_len: Input length

    Returns:
        Shifted data
    """
    # Convert ms to samples (scaled for input length)
    scale = input_len / (fs * 0.8)  # Assume ~0.8s beat
    shift_samples = int(shift_ms * fs / 1000 * scale)

    X_shifted = np.zeros_like(X)
    for i in range(len(X)):
        X_shifted[i] = scipy_shift(
            X[i].squeeze(), shift_samples, mode='nearest'
        ).reshape(-1, 1)

    return X_shifted


def evaluate_robustness(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shifts_ms: List[int],
    config: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate model robustness to temporal shifts.

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        shifts_ms: List of shift amounts in milliseconds
        config: Configuration dict

    Returns:
        Dict with metrics for each shift
    """
    cfg = config or DEFAULT_CONFIG
    fs = cfg.get("fs", 360)
    input_len = X_test.shape[1]

    results = {
        "shifts_ms": shifts_ms,
        "accuracy": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "predictions": {},
        "probabilities": {},
    }

    for shift_ms in shifts_ms:
        # Apply shift
        X_shifted = apply_temporal_shift(X_test, shift_ms, fs, input_len)

        # Predict
        y_pred_proba = model.predict(X_shifted, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        results["accuracy"].append(float(accuracy_score(y_test, y_pred)))
        results["f1"].append(float(f1_score(y_test, y_pred, pos_label=1)))
        results["precision"].append(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)))
        results["recall"].append(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)))

        try:
            results["auc"].append(float(roc_auc_score(y_test, y_pred_proba[:, 1])))
        except Exception:
            results["auc"].append(0.0)

        results["predictions"][shift_ms] = y_pred
        results["probabilities"][shift_ms] = y_pred_proba

    return results


def plot_robustness_curves(
    results: Dict,
    output_path: str,
    model_name: str = "Model",
) -> None:
    """
    Plot robustness curves showing performance vs temporal shift.

    Args:
        results: Results from evaluate_robustness
        output_path: Path to save plot
        model_name: Name of the model for plot title
    """
    shifts = results["shifts_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].plot(shifts, results["accuracy"], 'b-o', linewidth=2, markersize=8)
    axes[0, 0].fill_between(shifts, results["accuracy"], alpha=0.2)
    axes[0, 0].set_xlabel('Temporal Shift (ms)', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Accuracy vs Temporal Shift', fontsize=14)
    axes[0, 0].axhline(y=results["accuracy"][len(shifts)//2], color='r', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    # AUC
    axes[0, 1].plot(shifts, results["auc"], 'g-o', linewidth=2, markersize=8)
    axes[0, 1].fill_between(shifts, results["auc"], alpha=0.2, color='green')
    axes[0, 1].set_xlabel('Temporal Shift (ms)', fontsize=12)
    axes[0, 1].set_ylabel('AUC', fontsize=12)
    axes[0, 1].set_title('AUC vs Temporal Shift', fontsize=14)
    axes[0, 1].axhline(y=results["auc"][len(shifts)//2], color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(shifts, results["f1"], 'r-o', linewidth=2, markersize=8)
    axes[1, 0].fill_between(shifts, results["f1"], alpha=0.2, color='red')
    axes[1, 0].set_xlabel('Temporal Shift (ms)', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score (Abnormal)', fontsize=12)
    axes[1, 0].set_title('F1 Score vs Temporal Shift', fontsize=14)
    axes[1, 0].axhline(y=results["f1"][len(shifts)//2], color='r', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(shifts, results["precision"], 'm-o', linewidth=2, markersize=8, label='Precision')
    axes[1, 1].plot(shifts, results["recall"], 'c-s', linewidth=2, markersize=8, label='Recall')
    axes[1, 1].set_xlabel('Temporal Shift (ms)', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Precision & Recall vs Temporal Shift', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} - Robustness Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrices(
    results: Dict,
    y_test: np.ndarray,
    output_path: str,
    shifts_to_plot: List[int] = [-40, 0, 40],
    model_name: str = "Model",
) -> None:
    """
    Plot confusion matrices at different shift levels.

    Args:
        results: Results from evaluate_robustness
        y_test: True labels
        output_path: Path to save plot
        shifts_to_plot: Which shifts to plot
        model_name: Name of the model for plot title
    """
    n_plots = len(shifts_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    for ax, shift_ms in zip(axes, shifts_to_plot):
        y_pred = results["predictions"][shift_ms]
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal']
        )
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'Shift: {shift_ms} ms', fontsize=12)

    plt.suptitle(f'{model_name} - Confusion Matrices', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves(
    results: Dict,
    y_test: np.ndarray,
    output_path: str,
    shifts_to_plot: List[int] = [-40, -20, 0, 20, 40],
    model_name: str = "Model",
) -> None:
    """
    Plot ROC curves at different shift levels.

    Args:
        results: Results from evaluate_robustness
        y_test: True labels
        output_path: Path to save plot
        shifts_to_plot: Which shifts to plot
        model_name: Name of the model for plot title
    """
    plt.figure(figsize=(10, 8))

    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(shifts_to_plot)))

    for shift_ms, color in zip(shifts_to_plot, colors):
        y_proba = results["probabilities"][shift_ms][:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'Shift {shift_ms:+d}ms (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves at Different Shifts', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_summary_table(
    results: Dict,
    model_name: str = "Model",
    include_all_shifts: bool = False,
) -> pd.DataFrame:
    """
    Generate summary table with key metrics.

    Args:
        results: Results from evaluate_robustness
        model_name: Name of the model
        include_all_shifts: Whether to include all shifts or just summary

    Returns:
        DataFrame with summary metrics
    """
    shifts = results["shifts_ms"]
    zero_idx = shifts.index(0) if 0 in shifts else len(shifts) // 2

    if include_all_shifts:
        data = {
            "Shift (ms)": shifts,
            "Accuracy": results["accuracy"],
            "AUC": results["auc"],
            "F1 (Abnormal)": results["f1"],
            "Precision": results["precision"],
            "Recall": results["recall"],
        }
        return pd.DataFrame(data)
    else:
        # Summary statistics
        data = {
            "Metric": [
                "Accuracy (0ms)",
                "Accuracy (mean)",
                "Accuracy (std)",
                "AUC (0ms)",
                "AUC (mean)",
                "AUC (std)",
                "F1 (0ms)",
                "F1 (mean)",
                "F1 (std)",
                "Robustness (max drop)",
            ],
            "Value": [
                results["accuracy"][zero_idx],
                np.mean(results["accuracy"]),
                np.std(results["accuracy"]),
                results["auc"][zero_idx],
                np.mean(results["auc"]),
                np.std(results["auc"]),
                results["f1"][zero_idx],
                np.mean(results["f1"]),
                np.std(results["f1"]),
                results["accuracy"][zero_idx] - min(results["accuracy"]),
            ]
        }
        return pd.DataFrame(data)


def count_params(model: tf.keras.Model) -> int:
    """Count trainable parameters."""
    return sum([np.prod(w.shape) for w in model.trainable_weights])


def measure_inference_time(model: tf.keras.Model, X: np.ndarray, n_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    # Warm up
    for _ in range(10):
        model.predict(X[:1], verbose=0)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X[:1], verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times)


def evaluate_model_suite(
    model_paths: Dict[str, str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate multiple models and generate comparison plots.

    Args:
        model_paths: Dict mapping model names to paths
        X_test: Test data
        y_test: Test labels
        output_dir: Directory to save outputs
        config: Configuration dict

    Returns:
        Dict with results for each model
    """
    cfg = config or DEFAULT_CONFIG
    shifts_ms = cfg.get("shifts_ms", [-40, -30, -20, -10, 0, 10, 20, 30, 40])

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for name, path in model_paths.items():
        print(f"\nEvaluating {name}...")
        model = load_model(path)

        # Evaluate robustness
        results = evaluate_robustness(model, X_test, y_test, shifts_ms, cfg)

        # Add model info
        results["params"] = count_params(model)
        results["inference_time_ms"] = measure_inference_time(model, X_test)

        # Generate plots
        plot_robustness_curves(
            results,
            os.path.join(output_dir, f'robustness_curves_{name.lower().replace(" ", "_")}.png'),
            name
        )

        plot_confusion_matrices(
            results, y_test,
            os.path.join(output_dir, f'confusion_matrices_{name.lower().replace(" ", "_")}.png'),
            shifts_to_plot=[-40, 0, 40],
            model_name=name
        )

        plot_roc_curves(
            results, y_test,
            os.path.join(output_dir, f'roc_curves_{name.lower().replace(" ", "_")}.png'),
            model_name=name
        )

        # Save summary table
        summary = generate_summary_table(results, name, include_all_shifts=True)
        summary.to_csv(
            os.path.join(output_dir, f'metrics_{name.lower().replace(" ", "_")}.csv'),
            index=False
        )

        all_results[name] = results

    # Generate comparison plot
    if len(model_paths) > 1:
        plt.figure(figsize=(14, 10))

        for i, (metric, ylabel) in enumerate([
            ("accuracy", "Accuracy"),
            ("auc", "AUC"),
            ("f1", "F1 Score (Abnormal)"),
        ]):
            plt.subplot(2, 2, i + 1)

            for name, results in all_results.items():
                plt.plot(results["shifts_ms"], results[metric], '-o',
                         linewidth=2, markersize=6, label=name)

            plt.xlabel('Temporal Shift (ms)', fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            plt.title(f'{ylabel} vs Temporal Shift', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Parameter and timing comparison
        plt.subplot(2, 2, 4)
        names = list(all_results.keys())
        params = [all_results[n]["params"] / 1000 for n in names]
        times = [all_results[n]["inference_time_ms"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax1 = plt.gca()
        bars1 = ax1.bar(x - width/2, params, width, label='Parameters (K)', color='steelblue')
        ax1.set_ylabel('Parameters (thousands)', fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha='right')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, times, width, label='Inference Time', color='coral')
        ax2.set_ylabel('Inference Time (ms)', fontsize=11)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('Model Size and Speed Comparison', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
        plt.close()

        # Save comparison CSV
        comparison_data = {"Model": names}
        comparison_data["Parameters"] = [all_results[n]["params"] for n in names]
        comparison_data["Inference Time (ms)"] = [all_results[n]["inference_time_ms"] for n in names]
        comparison_data["Accuracy (0ms)"] = [
            all_results[n]["accuracy"][all_results[n]["shifts_ms"].index(0)]
            for n in names
        ]
        comparison_data["AUC (0ms)"] = [
            all_results[n]["auc"][all_results[n]["shifts_ms"].index(0)]
            for n in names
        ]
        comparison_data["F1 (0ms)"] = [
            all_results[n]["f1"][all_results[n]["shifts_ms"].index(0)]
            for n in names
        ]
        comparison_data["Accuracy (mean)"] = [np.mean(all_results[n]["accuracy"]) for n in names]
        comparison_data["Max Accuracy Drop"] = [
            all_results[n]["accuracy"][all_results[n]["shifts_ms"].index(0)] -
            min(all_results[n]["accuracy"])
            for n in names
        ]

        pd.DataFrame(comparison_data).to_csv(
            os.path.join(output_dir, 'model_comparison_table.csv'),
            index=False
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (.h5) or comma-separated paths")
    parser.add_argument("--model_names", type=str, default=None,
                        help="Comma-separated model names (optional)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="outputs/plots",
                        help="Directory to save outputs")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set size for data split")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Parse model paths
    model_paths = args.model_path.split(',')
    if args.model_names:
        model_names = args.model_names.split(',')
    else:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]

    model_dict = dict(zip(model_names, model_paths))

    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data(
        args.data_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Test samples: {len(X_test)}")

    # Evaluate
    config = DEFAULT_CONFIG.copy()
    config["random_state"] = args.random_state

    results = evaluate_model_suite(
        model_dict,
        X_test,
        y_test,
        args.output_dir,
        config
    )

    print(f"\nEvaluation complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
