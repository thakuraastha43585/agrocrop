"""Training script for AgroVision AI (MobileNetV2 with balanced class weights)."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data_loader import prepare_datasets
from src.model import build_mobilenetv2_classifier


class EpochSummaryLogger(tf.keras.callbacks.Callback):
    """Print per-epoch timing, learning rate, loss, and accuracy cleanly."""

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        elapsed = time.time() - self._epoch_start
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(
            f"[Epoch {epoch + 1}] "
            f"time={elapsed:.1f}s | lr={lr:.6g} | "
            f"loss={logs.get('loss', float('nan')):.4f} | "
            f"acc={logs.get('accuracy', float('nan')):.4f} | "
            f"val_loss={logs.get('val_loss', float('nan')):.4f} | "
            f"val_acc={logs.get('val_accuracy', float('nan')):.4f}",
            flush=True,
        )


def compute_balanced_class_weights(
    train_records: list[dict], class_names: list[str]
) -> tuple[dict[int, float], dict[str, int]]:
    """Compute exact per-class training counts and balanced Scikit-learn class weights."""
    train_labels = np.array([int(r["label"]) for r in train_records], dtype=np.int64)
    label_indices = np.arange(len(class_names), dtype=np.int64)

    counts_by_idx = Counter(train_labels.tolist())
    class_counts = {
        class_names[idx]: counts_by_idx.get(idx, 0) for idx in range(len(class_names))
    }

    weights_array = compute_class_weight(
        class_weight="balanced",
        classes=label_indices,
        y=train_labels,
    )
    class_weights = {
        int(idx): float(weight) for idx, weight in zip(label_indices, weights_array)
    }
    return class_weights, class_counts


def save_class_names(class_names: list[str], path: Path = config.CLASS_NAMES_PATH) -> None:
    """Persist ordered class names so inference maps indices consistently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": class_names,
        "class_to_index": {name: idx for idx, name in enumerate(class_names)},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_saved_model_inference(
    model_path: Path,
    test_ds: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    """Load the saved .keras model and verify real predictions on one test batch."""
    loaded_model = tf.keras.models.load_model(model_path)
    sample_images, sample_labels = next(iter(test_ds.take(1)))
    probs = loaded_model.predict(sample_images, verbose=0)

    expected_shape = (int(sample_images.shape[0]), len(class_names))
    if probs.shape != expected_shape:
        raise AssertionError(
            f"Unexpected prediction shape {probs.shape}, expected {expected_shape}"
        )

    prob_sums = probs.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-4):
        raise AssertionError("Softmax probabilities do not sum to 1.0")

    pred_indices = np.argmax(probs, axis=1)
    true_indices = sample_labels.numpy().astype(int)
    batch_acc = float(np.mean(pred_indices == true_indices))

    return {
        "batch_size": int(sample_images.shape[0]),
        "prediction_shape": list(probs.shape),
        "prob_sum_min": float(prob_sums.min()),
        "prob_sum_max": float(prob_sums.max()),
        "sample_batch_accuracy": batch_acc,
        "sample_predictions": [
            {
                "true_label": class_names[int(t)],
                "predicted_label": class_names[int(p)],
                "confidence": float(probs[i, int(p)]),
            }
            for i, (t, p) in enumerate(zip(true_indices[:5], pred_indices[:5]))
        ],
    }


def train_model(epochs: int = config.INITIAL_EPOCHS) -> dict:
    """Run MobileNetV2 transfer learning with balanced class weights on CPU."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    # Load verified manifest-based datasets (augments train_ds only; val/test unchanged)
    data = prepare_datasets(force_rebuild_manifest=False, generate_plots=False)
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]
    test_ds = data["test_ds"]
    splits = data["splits"]
    class_names = data["class_names"]

    class_weights, class_counts = compute_balanced_class_weights(
        splits["train"], class_names
    )

    print("=== TRAINING CLASS DISTRIBUTION & BALANCED CLASS WEIGHTS ===")
    total_train = len(splits["train"])
    for idx, name in enumerate(class_names):
        cnt = class_counts[name]
        pct = 100.0 * cnt / total_train
        print(
            f"  [{idx}] {name:<24} | count={cnt:>4} ({pct:5.2f}%) | "
            f"class_weight={class_weights[idx]:.4f}"
        )

    save_class_names(class_names, config.CLASS_NAMES_PATH)

    model, base_model = build_mobilenetv2_classifier(
        num_classes=len(class_names),
        input_shape=config.INPUT_SHAPE,
        dropout_rate=config.DROPOUT_RATE,
        learning_rate=config.INITIAL_LEARNING_RATE,
        freeze_base=True,
    )

    trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )
    print("\n=== MODEL SUMMARY ===")
    model.summary()
    print(
        f"Trainable params: {trainable_params:,} | "
        f"Non-trainable params: {non_trainable_params:,}",
        flush=True,
    )

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.TRAINING_LOG_CSV_PATH), append=False),
        EpochSummaryLogger(),
    ]

    print(f"\n=== STARTING TRAINING ({epochs} epochs, batch_size={config.BATCH_SIZE}) ===", flush=True)
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    total_train_seconds = time.time() - t_start

    serializable_history = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    val_losses = serializable_history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else None

    history_payload = {
        "epochs_requested": epochs,
        "epochs_completed": len(serializable_history.get("loss", [])),
        "best_epoch_by_val_loss": best_epoch,
        "total_training_time_seconds": round(total_train_seconds, 2),
        "class_names": class_names,
        "train_class_counts": class_counts,
        "class_weights": {class_names[i]: class_weights[i] for i in range(len(class_names))},
        "history": serializable_history,
    }
    config.HISTORY_PATH.write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )

    model_config_payload = {
        "architecture": "MobileNetV2",
        "weights": "imagenet",
        "base_model_frozen": not base_model.trainable,
        "input_shape": list(config.INPUT_SHAPE),
        "num_classes": len(class_names),
        "class_names": class_names,
        "preprocessing": "Applied once in src.data_loader via mobilenet_v2.preprocess_input ([-1, 1])",
        "augmentation": "Applied only to training split in src.data_loader",
        "head": {
            "pooling": "GlobalAveragePooling2D",
            "dropout_rate": config.DROPOUT_RATE,
            "classifier": "Dense(num_classes, activation='softmax')",
        },
        "optimizer": "Adam",
        "initial_learning_rate": config.INITIAL_LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy",
        "batch_size": config.BATCH_SIZE,
        "random_seed": config.RANDOM_SEED,
        "split_counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "saved_model_path": str(config.MODEL_PATH.relative_to(config.BASE_DIR).as_posix()),
    }
    config.MODEL_CONFIG_PATH.write_text(
        json.dumps(model_config_payload, indent=2), encoding="utf-8"
    )

    inference_check = verify_saved_model_inference(
        config.MODEL_PATH, test_ds, class_names
    )
    print("\n=== SAVED MODEL INFERENCE CHECK (1 TEST BATCH) ===")
    print(json.dumps(inference_check, indent=2))

    return {
        "history": history_payload,
        "model_config": model_config_payload,
        "inference_check": inference_check,
    }


if __name__ == "__main__":
    train_model()
