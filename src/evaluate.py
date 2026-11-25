"""Full test-set evaluation and visualization script for AgroVision AI."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src import config
from src.data_loader import (
    build_tf_dataset,
    format_class_label,
    load_splits_manifest,
)


def load_class_names(path: Path = config.CLASS_NAMES_PATH) -> list[str]:
    """Load ordered class names saved during training."""
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["class_names"])


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    test_accuracy: float,
    output_path: Path = config.CONFUSION_MATRIX_FIG,
) -> Path:
    """Plot and save an annotated confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_labels = [format_class_label(c) for c in class_names]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    annot_labels = np.array(
        [
            [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]
    )

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Image Count"},
        ax=ax,
    )
    ax.set_title(
        f"Test-Set Confusion Matrix (N = {int(cm.sum()):,}, Accuracy = {test_accuracy * 100:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_training_curves(
    history_path: Path = config.HISTORY_PATH,
    accuracy_fig_path: Path = config.TRAINING_ACCURACY_FIG,
    loss_fig_path: Path = config.TRAINING_LOSS_FIG,
) -> tuple[Path, Path]:
    """Plot and save training/validation accuracy and loss curves from training history."""
    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload["history"]
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch = payload.get("best_epoch_by_val_loss")

    sns.set_theme(style="whitegrid")
    accuracy_fig_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fig_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Training & Validation Accuracy
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(epochs, history["accuracy"], marker="o", linewidth=2, label="Train Accuracy")
    ax.plot(epochs, history["val_accuracy"], marker="s", linewidth=2, label="Validation Accuracy")
    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            color="#2e7d32",
            linestyle="--",
            alpha=0.75,
            label=f"Best Checkpoint (Epoch {best_epoch})",
        )
    ax.set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(epochs)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(accuracy_fig_path, dpi=150)
    plt.close(fig)

    # 2. Training & Validation Loss
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(epochs, history["loss"], marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="s", linewidth=2, label="Validation Loss")
    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            color="#2e7d32",
            linestyle="--",
            alpha=0.75,
            label=f"Best Checkpoint (Epoch {best_epoch})",
        )
    ax.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparse Categorical Cross-Entropy Loss")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)

    return accuracy_fig_path, loss_fig_path


def evaluate_model() -> dict:
    """Evaluate the saved model strictly on the held-out test split."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {config.MODEL_PATH}")

    splits, manifest_classes = load_splits_manifest(config.SPLITS_MANIFEST_PATH)
    class_names = load_class_names(config.CLASS_NAMES_PATH)
    if class_names != manifest_classes:
        raise ValueError(
            f"Class mismatch between {config.CLASS_NAMES_PATH} and {config.SPLITS_MANIFEST_PATH}"
        )

    test_records = splits["test"]
    test_ds = build_tf_dataset(
        test_records,
        batch_size=config.BATCH_SIZE,
        training=False,
    )

    model = tf.keras.models.load_model(config.MODEL_PATH)

    # Run deterministic single-pass inference over all batches of the held-out test split
    y_true_batches = []
    y_prob_batches = []
    for batch_images, batch_labels in test_ds:
        y_true_batches.append(batch_labels.numpy())
        y_prob_batches.append(model.predict_on_batch(batch_images))

    y_true = np.concatenate(y_true_batches, axis=0).astype(int)
    y_probs = np.concatenate(y_prob_batches, axis=0)
    y_pred = np.argmax(y_probs, axis=1).astype(int)

    if len(y_true) != len(test_records) or len(y_pred) != len(test_records):
        raise AssertionError(
            f"Expected {len(test_records)} test predictions, got {len(y_pred)}"
        )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(loss_fn(y_true, y_probs).numpy())
    test_accuracy = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    per_class_metrics = {
        class_names[i]: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1_score": float(per_f1[i]),
            "support": int(per_sup[i]),
        }
        for i in range(len(class_names))
    }

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    cm_fig = plot_confusion_matrix(cm, class_names, test_accuracy)
    acc_fig, loss_fig = plot_training_curves()

    results = {
        "model_path": str(config.MODEL_PATH.relative_to(config.BASE_DIR).as_posix()),
        "manifest_path": str(config.SPLITS_MANIFEST_PATH.relative_to(config.BASE_DIR).as_posix()),
        "evaluation_split": "test",
        "test_samples": int(len(y_true)),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_avg": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1_score": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1_score": float(weighted_f1),
        },
        "per_class_metrics": per_class_metrics,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "figures": {
            "confusion_matrix": str(cm_fig.relative_to(config.BASE_DIR).as_posix()),
            "training_accuracy": str(acc_fig.relative_to(config.BASE_DIR).as_posix()),
            "training_loss": str(loss_fig.relative_to(config.BASE_DIR).as_posix()),
        },
    }

    config.EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVALUATION_RESULTS_PATH.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(f"=== TEST-SET EVALUATION (N = {len(y_true):,}) ===")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(
        f"Macro Avg:     P={macro_p:.4f} | R={macro_r:.4f} | F1={macro_f1:.4f}"
    )
    print(
        f"Weighted Avg:  P={weighted_p:.4f} | R={weighted_r:.4f} | F1={weighted_f1:.4f}"
    )
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)
    print("=== CONFUSION MATRIX ===")
    print(cm)
    print(f"\nSaved results to: {config.EVALUATION_RESULTS_PATH}")
    print("Saved figures:", results["figures"])
    return results


if __name__ == "__main__":
    evaluate_model()
