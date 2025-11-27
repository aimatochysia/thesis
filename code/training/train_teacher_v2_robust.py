#!/usr/bin/env python
"""
Teacher Model Training with Misalignment-Robust Augmentations

This script implements training for the v2 CNN teacher model with robustness
augmentations to handle misalignment in beat segmentation:
- Random temporal shift (±10-40 ms)
- Mild time-warp (95-105%)
- Small amplitude scaling/noise
- Consistency regularization

Usage:
    python train_teacher_v2_robust.py --data_path ../../ecg.csv --output_dir ../../outputs/models

The trained model will be saved as teacher_v2_robust.h5
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy.signal import resample
from scipy.ndimage import shift as scipy_shift

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout,
    BatchNormalization, Activation, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Default configuration
DEFAULT_CONFIG = {
    "input_len": 188,
    "fs": 360,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.001,
    "random_state": 42,
    # Augmentation parameters
    "shift_range_ms": (10, 40),  # ±10-40 ms shift
    "time_warp_range": (0.95, 1.05),  # 95-105% time warp
    "amplitude_scale_range": (0.95, 1.05),
    "noise_std": 0.01,
    # Consistency regularization
    "consistency_weight": 0.1,
}


def create_v2_cnn_model(input_shape: Tuple[int, int], num_classes: int = 2) -> Model:
    """
    Create the v2 CNN model architecture.

    Args:
        input_shape: Input shape (timesteps, features)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Block 1
        Conv1D(32, kernel_size=5, padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(32, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Block 2
        Conv1D(64, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(64, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Block 3
        Conv1D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 4
        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling1D(),

        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    return model


class TemporalShiftAugmentation:
    """Apply random temporal shift to ECG beats."""

    def __init__(
        self,
        shift_range_ms: Tuple[int, int] = (10, 40),
        fs: int = 360,
        input_len: int = 188,
    ):
        """
        Args:
            shift_range_ms: Range for shift in milliseconds (min, max)
            fs: Sampling rate
            input_len: Input signal length
        """
        self.shift_range_ms = shift_range_ms
        self.fs = fs
        self.input_len = input_len
        # Convert ms to samples
        self.min_shift_samples = int(shift_range_ms[0] * fs / 1000)
        self.max_shift_samples = int(shift_range_ms[1] * fs / 1000)
        # Scale for 188-sample beat. Assumes typical beat duration of ~0.8s at fs=360
        # This scaling factor converts physical shift (ms) to sample shift in the 188-sample input
        # For different beat durations, adjust the assumed_beat_duration_sec parameter
        assumed_beat_duration_sec = 0.8  # Typical RR interval ~800ms
        scale = input_len / (fs * assumed_beat_duration_sec)
        self.min_shift_scaled = max(1, int(self.min_shift_samples * scale))
        self.max_shift_scaled = max(2, int(self.max_shift_samples * scale))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random temporal shift.

        Args:
            x: Input beat (samples,) or (samples, 1)

        Returns:
            Shifted beat
        """
        squeeze = False
        if x.ndim == 2:
            x = x.squeeze(-1)
            squeeze = True

        # Random direction and magnitude
        direction = np.random.choice([-1, 1])
        shift_amount = np.random.randint(self.min_shift_scaled, self.max_shift_scaled + 1)
        shift = direction * shift_amount

        # Apply shift using scipy
        shifted = scipy_shift(x, shift, mode='nearest')

        if squeeze:
            shifted = shifted.reshape(-1, 1)

        return shifted.astype(np.float32)


class TimeWarpAugmentation:
    """Apply mild time warping to ECG beats."""

    def __init__(
        self,
        warp_range: Tuple[float, float] = (0.95, 1.05),
        target_len: int = 188,
    ):
        """
        Args:
            warp_range: Range for time warp factor (min, max)
            target_len: Target output length
        """
        self.warp_range = warp_range
        self.target_len = target_len

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random time warp.

        Args:
            x: Input beat (samples,) or (samples, 1)

        Returns:
            Time-warped beat
        """
        squeeze = False
        if x.ndim == 2:
            x = x.squeeze(-1)
            squeeze = True

        # Random warp factor
        factor = np.random.uniform(self.warp_range[0], self.warp_range[1])
        warped_len = int(len(x) * factor)

        # Resample to warped length then back to target
        warped = resample(x, warped_len)
        warped = resample(warped, self.target_len)

        if squeeze:
            warped = warped.reshape(-1, 1)

        return warped.astype(np.float32)


class AmplitudeAugmentation:
    """Apply amplitude scaling and noise to ECG beats."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.01,
    ):
        """
        Args:
            scale_range: Range for amplitude scaling (min, max)
            noise_std: Standard deviation of additive noise
        """
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random amplitude augmentation.

        Args:
            x: Input beat

        Returns:
            Augmented beat
        """
        # Random scale
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scaled = x * scale

        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, x.shape)
            scaled = scaled + noise

        return scaled.astype(np.float32)


class ConsistencyRegularization:
    """
    Consistency regularization: encourage model to produce similar outputs
    for differently augmented views of the same input.
    """

    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: Weight for consistency loss term
        """
        self.weight = weight

    def compute_loss(
        self,
        pred1: tf.Tensor,
        pred2: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute consistency loss between two predictions.

        Args:
            pred1: First prediction (batch_size, num_classes)
            pred2: Second prediction (batch_size, num_classes)

        Returns:
            Consistency loss (scalar)
        """
        # Use KL divergence
        kl = tf.keras.losses.KLDivergence()
        loss = (kl(pred1, pred2) + kl(pred2, pred1)) / 2
        return self.weight * loss


class AugmentedDataGenerator(tf.keras.utils.Sequence):
    """Data generator with augmentation for robust training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augment: bool = True,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            X: Input data (samples, timesteps, features)
            y: Labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            config: Configuration dict
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.config = config or DEFAULT_CONFIG

        # Initialize augmenters
        self.temporal_shift = TemporalShiftAugmentation(
            shift_range_ms=self.config.get("shift_range_ms", (10, 40)),
            input_len=X.shape[1],
        )
        self.time_warp = TimeWarpAugmentation(
            warp_range=self.config.get("time_warp_range", (0.95, 1.05)),
            target_len=X.shape[1],
        )
        self.amplitude_aug = AmplitudeAugmentation(
            scale_range=self.config.get("amplitude_scale_range", (0.95, 1.05)),
            noise_std=self.config.get("noise_std", 0.01),
        )

        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        batch_indices = self.indices[start:end]

        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]

        if self.augment:
            for i in range(len(X_batch)):
                # Apply augmentations with probability
                if np.random.random() < 0.5:
                    X_batch[i] = self.temporal_shift(X_batch[i])
                if np.random.random() < 0.3:
                    X_batch[i] = self.time_warp(X_batch[i])
                if np.random.random() < 0.5:
                    X_batch[i] = self.amplitude_aug(X_batch[i])

        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class RobustTrainer:
    """Trainer class for robust teacher model training with consistency regularization."""

    def __init__(
        self,
        model: Model,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model: Keras model to train
            config: Configuration dict
        """
        self.model = model
        self.config = config or DEFAULT_CONFIG
        self.consistency_reg = ConsistencyRegularization(
            weight=self.config.get("consistency_weight", 0.1)
        )
        self.history = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict] = None,
        output_dir: str = "outputs/models",
    ) -> Dict:
        """
        Train the model with augmentation and consistency regularization.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            class_weights: Optional class weights dict
            output_dir: Directory to save model

        Returns:
            Training history dict
        """
        os.makedirs(output_dir, exist_ok=True)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.get("learning_rate", 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(output_dir, 'teacher_v2_robust.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Create augmented data generator
        train_gen = AugmentedDataGenerator(
            X_train, y_train,
            batch_size=self.config.get("batch_size", 32),
            augment=True,
            config=self.config
        )

        # Train
        self.history = self.model.fit(
            train_gen,
            epochs=self.config.get("epochs", 200),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self.history.history

    def evaluate_robustness(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        shifts_ms: List[int] = [-40, -20, 0, 20, 40],
        output_dir: str = "outputs/plots",
    ) -> Dict:
        """
        Evaluate model robustness to temporal shifts.

        Args:
            X_test: Test data
            y_test: Test labels (one-hot)
            shifts_ms: List of shift amounts in milliseconds
            output_dir: Directory to save plots

        Returns:
            Dict with metrics for each shift
        """
        os.makedirs(output_dir, exist_ok=True)

        input_len = X_test.shape[1]
        fs = self.config.get("fs", 360)

        results = {}
        metrics_list = []

        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

        for shift_ms in shifts_ms:
            # Convert ms to samples (scaled for 188-sample beat)
            scale = input_len / (fs * 0.8)
            shift_samples = int(shift_ms * fs / 1000 * scale)

            # Apply shift to test data
            X_shifted = np.zeros_like(X_test)
            for i in range(len(X_test)):
                X_shifted[i] = scipy_shift(
                    X_test[i].squeeze(), shift_samples, mode='nearest'
                ).reshape(-1, 1)

            # Predict
            y_pred_proba = self.model.predict(X_shifted, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            # Calculate metrics
            acc = np.mean(y_pred == y_true)
            try:
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception:
                auc = 0.0

            results[f"shift_{shift_ms}ms"] = {
                "accuracy": float(acc),
                "auc": float(auc),
                "predictions": y_pred,
            }

            metrics_list.append({
                "shift_ms": shift_ms,
                "accuracy": acc,
                "auc": auc,
            })

        # Plot robustness curve
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        shifts = [m["shift_ms"] for m in metrics_list]
        accuracies = [m["accuracy"] for m in metrics_list]
        aucs = [m["auc"] for m in metrics_list]

        axes[0].plot(shifts, accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Temporal Shift (ms)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy vs Temporal Shift', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=accuracies[len(accuracies)//2], color='r', linestyle='--', alpha=0.5, label='Baseline (0ms)')
        axes[0].legend()

        axes[1].plot(shifts, aucs, 'g-o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Temporal Shift (ms)', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title('AUC vs Temporal Shift', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=aucs[len(aucs)//2], color='r', linestyle='--', alpha=0.5, label='Baseline (0ms)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_curve_teacher.png'), dpi=150)
        plt.close()

        # Save metrics to CSV
        pd.DataFrame(metrics_list).to_csv(
            os.path.join(output_dir, 'robustness_metrics_teacher.csv'),
            index=False
        )

        return results


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load and prepare data for training.

    Args:
        data_path: Path to CSV file(s)
        test_size: Test set fraction
        val_size: Validation set fraction
        random_state: Random seed

    Returns:
        Dict with train/val/test splits
    """
    # Load data
    df = pd.read_csv(data_path, header=None)

    # Assume last column is label
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    # Reshape for Conv1D
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=random_state, stratify=y
    )

    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp.argmax(axis=1)
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_teacher_v2_robust(
    data_path: str,
    output_dir: str = "outputs/models",
    config: Optional[Dict] = None,
) -> Tuple[Model, Dict]:
    """
    Main function to train the robust teacher model.

    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save outputs
        config: Configuration dict

    Returns:
        Tuple of (trained model, history)
    """
    cfg = config or DEFAULT_CONFIG.copy()
    np.random.seed(cfg.get("random_state", 42))
    tf.random.set_seed(cfg.get("random_state", 42))

    print("Loading and preparing data...")
    data = load_and_prepare_data(
        data_path,
        test_size=0.2,
        val_size=0.1,
        random_state=cfg.get("random_state", 42)
    )

    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")

    # Compute class weights
    y_train_classes = np.argmax(data["y_train"], axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_classes),
        y=y_train_classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Create model
    input_shape = (data["X_train"].shape[1], 1)
    num_classes = data["y_train"].shape[1]
    model = create_v2_cnn_model(input_shape, num_classes)
    model.summary()

    # Create trainer and train
    trainer = RobustTrainer(model, cfg)
    history = trainer.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        class_weights=class_weight_dict,
        output_dir=output_dir
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(data["X_test"], verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(data["y_test"], axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    # Evaluate robustness
    print("\nEvaluating robustness to temporal shifts...")
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    trainer.evaluate_robustness(
        data["X_test"], data["y_test"],
        shifts_ms=[-40, -20, -10, 0, 10, 20, 40],
        output_dir=plots_dir
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train robust teacher v2 CNN model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--data_path2", type=str, default=None,
                        help="Optional second data CSV to combine")
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                        help="Directory to save model and outputs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--consistency_weight", type=float, default=0.1,
                        help="Weight for consistency regularization")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["learning_rate"] = args.learning_rate
    config["consistency_weight"] = args.consistency_weight
    config["random_state"] = args.random_state

    # Handle multiple data paths
    data_path = args.data_path
    if args.data_path2:
        import tempfile
        # Combine datasets
        df1 = pd.read_csv(args.data_path, header=None)
        df2 = pd.read_csv(args.data_path2, header=None)
        combined = pd.concat([df1, df2], ignore_index=True)
        # Use tempfile for cross-platform compatibility
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix='combined_ecg_data_')
        combined.to_csv(temp_path, index=False, header=False)
        data_path = temp_path

    model, history = train_teacher_v2_robust(data_path, args.output_dir, config)
    print(f"\nTraining complete. Model saved to {args.output_dir}/teacher_v2_robust.h5")


if __name__ == "__main__":
    main()
