#!/usr/bin/env python
"""
Student Model Training with Knowledge Distillation

This script implements training for a compact student model using knowledge
distillation from the robust teacher model:
- Compact 1D CNN (<300k params) with depthwise separable convolutions
- Distillation loss: KL divergence to teacher's soft outputs + BCE to labels
- Optional consistency loss under small shifts
- Class weighting for abnormal class

Usage:
    python train_student_distill.py --teacher_path teacher_v2_robust.h5 \
        --data_path ../../ecg.csv --output_dir ../../outputs/models

The trained student model will be saved as student_distilled.h5
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import shift as scipy_shift

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, GlobalAveragePooling1D, Dense, Dropout,
    BatchNormalization, Activation, Input, SeparableConv1D, Add
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


# Default configuration
DEFAULT_CONFIG = {
    "input_len": 188,
    "fs": 360,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.001,
    "random_state": 42,
    # Distillation parameters
    "temperature": 3.0,  # Temperature for softening probabilities (T=2-4)
    "alpha": 0.7,  # Weight for distillation loss (vs hard label loss)
    # Consistency loss (optional)
    "use_consistency": True,
    "consistency_weight": 0.05,
    "shift_range_ms": (10, 20),
}


def create_student_model(input_shape: Tuple[int, int], num_classes: int = 2) -> Model:
    """
    Create a compact student model with depthwise separable convolutions.
    Target: <300k parameters

    Architecture:
    - 3-4 blocks with depthwise separable convs
    - BatchNorm + ReLU
    - GlobalAvgPool
    - Small dense head

    Args:
        input_shape: Input shape (timesteps, features)
        num_classes: Number of output classes

    Returns:
        Keras model
    """
    inputs = Input(shape=input_shape)

    # Block 1: Initial conv to increase channels
    x = Conv1D(16, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 2: Depthwise separable
    x = SeparableConv1D(32, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv1D(32, kernel_size=3, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # Block 3: Depthwise separable
    x = SeparableConv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv1D(64, kernel_size=3, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    # Block 4: Final separable conv
    x = SeparableConv1D(96, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # Global pooling
    x = GlobalAveragePooling1D()(x)

    # Small dense head
    x = Dense(48, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='student_model')
    return model


def create_baseline_tiny_model(input_shape: Tuple[int, int], num_classes: int = 2) -> Model:
    """
    Create a baseline tiny model (same size as student) without distillation.
    Used for comparison.

    Args:
        input_shape: Input shape (timesteps, features)
        num_classes: Number of output classes

    Returns:
        Keras model
    """
    # Same architecture as student
    return create_student_model(input_shape, num_classes)


class DistillationLoss:
    """
    Combined distillation loss:
    - KL divergence between student and teacher soft outputs (temperature-scaled)
    - Binary/categorical cross-entropy with hard labels
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
    ):
        """
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = tf.keras.losses.KLDivergence()
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy()

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred_student: tf.Tensor,
        y_pred_teacher: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute combined distillation loss.

        Args:
            y_true: Ground truth labels (one-hot)
            y_pred_student: Student predictions
            y_pred_teacher: Teacher predictions

        Returns:
            Combined loss
        """
        # Soft targets (temperature-scaled)
        soft_teacher = tf.nn.softmax(
            tf.math.log(y_pred_teacher + 1e-10) / self.temperature
        )
        soft_student = tf.nn.softmax(
            tf.math.log(y_pred_student + 1e-10) / self.temperature
        )

        # KL divergence loss (scaled by T^2 as per Hinton et al.)
        kl_loss = self.kl_loss(soft_teacher, soft_student) * (self.temperature ** 2)

        # Hard label loss
        hard_loss = self.ce_loss(y_true, y_pred_student)

        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss

        return total_loss


class StudentTrainer:
    """Trainer class for student model with knowledge distillation."""

    def __init__(
        self,
        student_model: Model,
        teacher_model: Model,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            student_model: Student model to train
            teacher_model: Pre-trained teacher model
            config: Configuration dict
        """
        self.student = student_model
        self.teacher = teacher_model
        self.config = config or DEFAULT_CONFIG
        self.distill_loss = DistillationLoss(
            temperature=self.config.get("temperature", 3.0),
            alpha=self.config.get("alpha", 0.7),
        )
        self.history = None

    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> Tuple[float, float]:
        """
        Single training step with distillation.

        Args:
            X_batch: Input batch
            y_batch: Label batch
            optimizer: Optimizer

        Returns:
            Tuple of (loss, accuracy)
        """
        # Get teacher predictions (no gradient)
        with tf.stop_gradient():
            teacher_preds = self.teacher(X_batch, training=False)

        # Forward pass and compute loss
        with tf.GradientTape() as tape:
            student_preds = self.student(X_batch, training=True)
            loss = self.distill_loss(y_batch, student_preds, teacher_preds)

            # Optional consistency loss
            if self.config.get("use_consistency", False):
                # Apply small shift to inputs
                X_shifted = self._apply_shift(X_batch)
                student_preds_shifted = self.student(X_shifted, training=True)
                kl = tf.keras.losses.KLDivergence()
                consistency_loss = (
                    kl(student_preds, student_preds_shifted) +
                    kl(student_preds_shifted, student_preds)
                ) / 2
                loss += self.config.get("consistency_weight", 0.05) * consistency_loss

        # Backward pass
        gradients = tape.gradient(loss, self.student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Compute accuracy
        y_pred = tf.argmax(student_preds, axis=1)
        y_true = tf.argmax(y_batch, axis=1)
        acc = tf.reduce_mean(tf.cast(y_pred == y_true, tf.float32))

        return float(loss), float(acc)

    def _apply_shift(self, X: np.ndarray) -> np.ndarray:
        """Apply random small shift to batch."""
        shift_range = self.config.get("shift_range_ms", (10, 20))
        fs = self.config.get("fs", 360)
        input_len = X.shape[1]
        scale = input_len / (fs * 0.8)

        min_shift = int(shift_range[0] * fs / 1000 * scale)
        max_shift = int(shift_range[1] * fs / 1000 * scale)

        X_shifted = np.zeros_like(X)
        for i in range(len(X)):
            direction = np.random.choice([-1, 1])
            shift = direction * np.random.randint(min_shift, max_shift + 1)
            X_shifted[i] = scipy_shift(
                X[i].squeeze(), shift, mode='nearest'
            ).reshape(-1, 1)

        return X_shifted

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
        Train the student model with knowledge distillation.

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

        # Compile student for validation metrics
        self.student.compile(
            optimizer=Adam(learning_rate=self.config.get("learning_rate", 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        optimizer = Adam(learning_rate=self.config.get("learning_rate", 0.001))
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 200)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30

        history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }

        n_batches = int(np.ceil(len(X_train) / batch_size))

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0
            epoch_acc = 0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(X_train))
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                loss, acc = self.train_step(X_batch, y_batch, optimizer)
                epoch_loss += loss
                epoch_acc += acc

            epoch_loss /= n_batches
            epoch_acc /= n_batches

            # Validation
            val_results = self.student.evaluate(X_val, y_val, verbose=0)
            val_loss, val_acc = val_results[0], val_results[1]

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.student.save(os.path.join(output_dir, 'student_distilled.h5'))
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        self.student = tf.keras.models.load_model(
            os.path.join(output_dir, 'student_distilled.h5')
        )

        self.history = history
        return history


def count_params(model: Model) -> int:
    """Count trainable parameters in a model."""
    return sum([np.prod(w.shape) for w in model.trainable_weights])


def measure_inference_time(model: Model, X: np.ndarray, n_runs: int = 100) -> float:
    """
    Measure average inference time.

    Args:
        model: Model to measure
        X: Sample input
        n_runs: Number of runs to average

    Returns:
        Average inference time in milliseconds
    """
    # Warm up
    for _ in range(10):
        model.predict(X[:1], verbose=0)

    # Measure
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X[:1], verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times)


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shifts_ms: List[int] = [-40, -20, 0, 20, 40],
    config: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate model performance including robustness.

    Args:
        model: Model to evaluate
        X_test: Test data
        y_test: Test labels (one-hot)
        shifts_ms: List of shift amounts for robustness evaluation
        config: Configuration dict

    Returns:
        Dict with evaluation metrics
    """
    cfg = config or DEFAULT_CONFIG
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    results = {}

    # Base metrics
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    results["accuracy"] = float(np.mean(y_pred == y_true))
    try:
        results["auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
    except Exception:
        results["auc"] = 0.0
    results["f1_abnormal"] = float(f1_score(y_true, y_pred, pos_label=1))

    # Robustness metrics
    input_len = X_test.shape[1]
    fs = cfg.get("fs", 360)
    robustness = {}

    for shift_ms in shifts_ms:
        scale = input_len / (fs * 0.8)
        shift_samples = int(shift_ms * fs / 1000 * scale)

        X_shifted = np.zeros_like(X_test)
        for i in range(len(X_test)):
            X_shifted[i] = scipy_shift(
                X_test[i].squeeze(), shift_samples, mode='nearest'
            ).reshape(-1, 1)

        y_pred_shifted = np.argmax(model.predict(X_shifted, verbose=0), axis=1)
        robustness[f"{shift_ms}ms"] = float(np.mean(y_pred_shifted == y_true))

    results["robustness"] = robustness

    # Parameter count and inference time
    results["params"] = count_params(model)
    results["inference_time_ms"] = measure_inference_time(model, X_test)

    return results


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Load and prepare data for training."""
    df = pd.read_csv(data_path, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int32)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y)

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


def train_student_distill(
    teacher_path: str,
    data_path: str,
    output_dir: str = "outputs/models",
    config: Optional[Dict] = None,
) -> Tuple[Model, Dict]:
    """
    Main function to train the student model with distillation.

    Args:
        teacher_path: Path to pre-trained teacher model (.h5)
        data_path: Path to training data CSV
        output_dir: Directory to save outputs
        config: Configuration dict

    Returns:
        Tuple of (trained student model, comparison results)
    """
    cfg = config or DEFAULT_CONFIG.copy()
    np.random.seed(cfg.get("random_state", 42))
    tf.random.set_seed(cfg.get("random_state", 42))

    print("Loading teacher model...")
    teacher = tf.keras.models.load_model(teacher_path)
    print(f"Teacher parameters: {count_params(teacher):,}")

    print("\nLoading and preparing data...")
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

    # Create student model
    input_shape = (data["X_train"].shape[1], 1)
    num_classes = data["y_train"].shape[1]
    student = create_student_model(input_shape, num_classes)
    print(f"\nStudent model parameters: {count_params(student):,}")
    student.summary()

    # Train student with distillation
    print("\n" + "="*60)
    print("Training student with knowledge distillation...")
    print("="*60)
    trainer = StudentTrainer(student, teacher, cfg)
    history = trainer.train(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        class_weights=class_weight_dict,
        output_dir=output_dir
    )

    # Also train a baseline (same architecture, no distillation)
    print("\n" + "="*60)
    print("Training baseline tiny model (no distillation)...")
    print("="*60)
    baseline = create_baseline_tiny_model(input_shape, num_classes)
    baseline.compile(
        optimizer=Adam(learning_rate=cfg.get("learning_rate", 0.001)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'baseline_tiny.h5'),
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

    baseline.fit(
        data["X_train"], data["y_train"],
        epochs=cfg.get("epochs", 200),
        batch_size=cfg.get("batch_size", 32),
        validation_data=(data["X_val"], data["y_val"]),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Load best student
    student = tf.keras.models.load_model(
        os.path.join(output_dir, 'student_distilled.h5')
    )

    # Evaluate all models
    print("\n" + "="*60)
    print("Evaluating models...")
    print("="*60)

    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    teacher_results = evaluate_model(teacher, data["X_test"], data["y_test"], config=cfg)
    student_results = evaluate_model(student, data["X_test"], data["y_test"], config=cfg)
    baseline_results = evaluate_model(baseline, data["X_test"], data["y_test"], config=cfg)

    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'Teacher':>12} {'Student':>12} {'Baseline':>12}")
    print("-"*60)
    print(f"{'Parameters':.<25} {teacher_results['params']:>12,} {student_results['params']:>12,} {baseline_results['params']:>12,}")
    print(f"{'Accuracy':.<25} {teacher_results['accuracy']:>12.4f} {student_results['accuracy']:>12.4f} {baseline_results['accuracy']:>12.4f}")
    print(f"{'AUC':.<25} {teacher_results['auc']:>12.4f} {student_results['auc']:>12.4f} {baseline_results['auc']:>12.4f}")
    print(f"{'F1 (Abnormal)':.<25} {teacher_results['f1_abnormal']:>12.4f} {student_results['f1_abnormal']:>12.4f} {baseline_results['f1_abnormal']:>12.4f}")
    print(f"{'Inference Time (ms)':.<25} {teacher_results['inference_time_ms']:>12.2f} {student_results['inference_time_ms']:>12.2f} {baseline_results['inference_time_ms']:>12.2f}")

    print("\nRobustness (Accuracy at different shifts):")
    for shift_key in teacher_results['robustness']:
        print(f"  {shift_key:.<20} {teacher_results['robustness'][shift_key]:>12.4f} {student_results['robustness'][shift_key]:>12.4f} {baseline_results['robustness'][shift_key]:>12.4f}")

    # Save comparison to CSV
    comparison_data = {
        "Metric": ["Parameters", "Accuracy", "AUC", "F1 (Abnormal)", "Inference Time (ms)"],
        "Teacher": [
            teacher_results['params'],
            teacher_results['accuracy'],
            teacher_results['auc'],
            teacher_results['f1_abnormal'],
            teacher_results['inference_time_ms']
        ],
        "Student (Distilled)": [
            student_results['params'],
            student_results['accuracy'],
            student_results['auc'],
            student_results['f1_abnormal'],
            student_results['inference_time_ms']
        ],
        "Baseline (Tiny)": [
            baseline_results['params'],
            baseline_results['accuracy'],
            baseline_results['auc'],
            baseline_results['f1_abnormal'],
            baseline_results['inference_time_ms']
        ]
    }

    # Add robustness metrics
    for shift_key in teacher_results['robustness']:
        comparison_data["Metric"].append(f"Acc @ {shift_key}")
        comparison_data["Teacher"].append(teacher_results['robustness'][shift_key])
        comparison_data["Student (Distilled)"].append(student_results['robustness'][shift_key])
        comparison_data["Baseline (Tiny)"].append(baseline_results['robustness'][shift_key])

    pd.DataFrame(comparison_data).to_csv(
        os.path.join(plots_dir, 'model_comparison.csv'),
        index=False
    )

    # Plot robustness comparison
    shifts_ms = [-40, -20, 0, 20, 40]
    teacher_accs = [teacher_results['robustness'][f"{s}ms"] for s in shifts_ms]
    student_accs = [student_results['robustness'][f"{s}ms"] for s in shifts_ms]
    baseline_accs = [baseline_results['robustness'][f"{s}ms"] for s in shifts_ms]

    plt.figure(figsize=(10, 6))
    plt.plot(shifts_ms, teacher_accs, 'b-o', linewidth=2, markersize=8, label='Teacher')
    plt.plot(shifts_ms, student_accs, 'g-s', linewidth=2, markersize=8, label='Student (Distilled)')
    plt.plot(shifts_ms, baseline_accs, 'r-^', linewidth=2, markersize=8, label='Baseline (Tiny)')
    plt.xlabel('Temporal Shift (ms)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Robustness Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_comparison.png'), dpi=150)
    plt.close()

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history['accuracy'], label='Train')
    axes[0].plot(history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Student Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['loss'], label='Train')
    axes[1].plot(history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Student Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'student_training_history.png'), dpi=150)
    plt.close()

    return student, {
        "teacher": teacher_results,
        "student": student_results,
        "baseline": baseline_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to pre-trained teacher model (.h5)")
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
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Temperature for distillation (2-4 recommended)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for distillation loss (vs hard label)")
    parser.add_argument("--use_consistency", action="store_true",
                        help="Enable consistency loss")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["learning_rate"] = args.learning_rate
    config["temperature"] = args.temperature
    config["alpha"] = args.alpha
    config["use_consistency"] = args.use_consistency
    config["random_state"] = args.random_state

    # Handle multiple data paths
    data_path = args.data_path
    if args.data_path2:
        df1 = pd.read_csv(args.data_path, header=None)
        df2 = pd.read_csv(args.data_path2, header=None)
        combined = pd.concat([df1, df2], ignore_index=True)
        temp_path = "/tmp/combined_ecg_data.csv"
        combined.to_csv(temp_path, index=False, header=False)
        data_path = temp_path

    student, results = train_student_distill(
        args.teacher_path,
        data_path,
        args.output_dir,
        config
    )

    print(f"\nTraining complete.")
    print(f"Student model saved to {args.output_dir}/student_distilled.h5")
    print(f"Baseline model saved to {args.output_dir}/baseline_tiny.h5")
    print(f"Comparison results saved to outputs/plots/model_comparison.csv")


if __name__ == "__main__":
    main()
