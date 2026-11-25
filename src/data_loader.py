"""Dataset validation, stratified manifest splitting, visualization, and tf.data pipelines.

This module reads directly from `config.DATASET_SOURCE_DIR` without copying or modifying
images in `data/raw/`. It produces a deterministic, leakage-free stratified split manifest
(`data/processed/splits_manifest.csv`), builds CPU-optimized `tf.data.Dataset` pipelines
with MobileNetV2 preprocessing and training-only augmentation, and generates exploratory
figures under `reports/figures/`.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config


def format_class_label(class_name: str) -> str:
    """Convert a folder name like 'Tomato_Early_blight' into a readable label."""
    return class_name.replace("_", " ").title()


def verify_image_file(image_path: Path) -> bool:
    """Return True if `image_path` is a supported, readable image file."""
    if not image_path.is_file():
        return False
    if image_path.suffix.lower() not in config.VALID_IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False


def collect_dataset_records(
    dataset_dir: Path = config.DATASET_SOURCE_DIR,
    selected_classes: list[str] = config.SELECTED_CLASSES,
    verify_images: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Collect valid image records from the canonical PlantVillage class folders.

    Reads only direct children of `dataset_dir / class_name` (non-recursive) so
    nested duplicate folders inside `data/raw/` are never included.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_names = sorted(selected_classes)
    records: list[dict] = []
    excluded_counts: dict[str, int] = {}
    seen_paths: set[Path] = set()

    for label_idx, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")

        excluded = 0
        for entry in sorted(class_dir.iterdir()):
            if not entry.is_file():
                continue
            if verify_images and not verify_image_file(entry):
                excluded += 1
                continue

            resolved = entry.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            rel_path = entry.relative_to(config.BASE_DIR).as_posix()
            records.append(
                {
                    "filepath": rel_path,
                    "class_name": class_name,
                    "label": label_idx,
                }
            )
        excluded_counts[class_name] = excluded

    if not records:
        raise RuntimeError(f"No valid images found under {dataset_dir}")

    return records, excluded_counts


def assert_disjoint_splits(splits: dict[str, list[dict]]) -> None:
    """Verify that no image filepath appears in more than one split."""
    split_sets = {
        name: {item["filepath"] for item in items} for name, items in splits.items()
    }
    split_names = list(split_sets.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise AssertionError(
                    f"Data leakage detected between '{left}' and '{right}': "
                    f"{len(overlap)} overlapping file(s)."
                )


def create_stratified_splits(
    records: list[dict],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> dict[str, list[dict]]:
    """Create deterministic stratified train/validation/test splits."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    labels = [r["label"] for r in records]
    train_records, temp_records = train_test_split(
        records,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    temp_labels = [r["label"] for r in temp_records]
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=seed,
    )

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    assert_disjoint_splits(splits)
    return splits


def save_splits_manifest(
    splits: dict[str, list[dict]],
    manifest_path: Path = config.SPLITS_MANIFEST_PATH,
) -> Path:
    """Persist path-based split assignments to CSV without duplicating image files."""
    assert_disjoint_splits(splits)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class_name", "label", "filepath"]
        )
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for row in splits[split_name]:
                writer.writerow(
                    {
                        "split": split_name,
                        "class_name": row["class_name"],
                        "label": row["label"],
                        "filepath": row["filepath"],
                    }
                )
    return manifest_path


def load_splits_manifest(
    manifest_path: Path = config.SPLITS_MANIFEST_PATH,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Load split assignments from an existing CSV manifest and verify integrity."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Splits manifest not found: {manifest_path}")

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    label_to_name: dict[int, str] = {}

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_name = row["split"]
            if split_name not in splits:
                raise ValueError(f"Unexpected split name in manifest: {split_name}")
            label = int(row["label"])
            class_name = row["class_name"]
            label_to_name[label] = class_name
            splits[split_name].append(
                {
                    "filepath": row["filepath"],
                    "class_name": class_name,
                    "label": label,
                }
            )

    assert_disjoint_splits(splits)
    class_names = [label_to_name[idx] for idx in sorted(label_to_name)]
    return splits, class_names


def build_augmentation_layer(seed: int = config.RANDOM_SEED) -> tf.keras.Sequential:
    """Build training-only spatial/photometric augmentation layers for [0, 255] images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.10, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.10, seed=seed),
            tf.keras.layers.RandomBrightness(
                0.10, value_range=(0.0, 255.0), seed=seed
            ),
        ],
        name="train_augmentation",
    )


def _decode_and_resize(
    filepath: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode an image file from disk and resize to MobileNetV2's 224x224 input size."""
    image_bytes = tf.io.read_file(filepath)
    image = tf.io.decode_image(
        image_bytes, channels=config.IMAGE_CHANNELS, expand_animations=False
    )
    image.set_shape([None, None, config.IMAGE_CHANNELS])
    image = tf.image.resize(
        image, config.IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.float32)
    return image, label


def build_tf_dataset(
    records: list[dict],
    batch_size: int = config.BATCH_SIZE,
    training: bool = False,
    seed: int = config.RANDOM_SEED,
    augmentation_layer: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Create a CPU-optimized `tf.data.Dataset` with MobileNetV2 preprocessing.

    Images are resized to 224x224, augmented (only when `training=True`), and
    normalized to [-1, 1] via `tf.keras.applications.mobilenet_v2.preprocess_input`.
    """
    abs_paths = [str(config.BASE_DIR / r["filepath"]) for r in records]
    labels = [int(r["label"]) for r in records]

    ds = tf.data.Dataset.from_tensor_slices((abs_paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(abs_paths), seed=seed, reshuffle_each_iteration=True
        )
        options = tf.data.Options()
        options.deterministic = False
        ds = ds.with_options(options)

    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if training:
        aug = augmentation_layer or build_augmentation_layer(seed=seed)
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def summarize_split_counts(
    splits: dict[str, list[dict]], class_names: list[str]
) -> dict[str, dict[str, int]]:
    """Return exact per-class image counts for each split."""
    summary: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        counts = Counter(r["class_name"] for r in splits[split_name])
        summary[split_name] = {c: counts.get(c, 0) for c in class_names}
    return summary


def plot_class_distribution(
    splits: dict[str, list[dict]],
    class_names: list[str],
    output_path: Path = config.CLASS_DISTRIBUTION_FIG,
) -> Path:
    """Plot and save class counts across train, validation, and test splits."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_split_counts(splits, class_names)
    display_names = [format_class_label(c) for c in class_names]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(class_names))
    width = 0.25
    split_specs = [
        ("train", "Train (70%)", "#2e7d32"),
        ("val", "Validation (15%)", "#1565c0"),
        ("test", "Test (15%)", "#ef6c00"),
    ]

    for offset, (split_key, label, color) in zip((-width, 0.0, width), split_specs):
        counts = [summary[split_key][c] for c in class_names]
        bars = ax.bar(x + offset, counts, width, label=label, color=color)
        for bar in bars:
            height = int(bar.get_height())
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    total_images = sum(len(splits[s]) for s in ("train", "val", "test"))
    ax.set_title(
        f"PlantVillage Tomato Dataset — Stratified Split Distribution (Total: {total_images:,} images)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Disease Category")
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_sample_images(
    train_records: list[dict],
    class_names: list[str],
    output_path: Path = config.SAMPLE_IMAGES_FIG,
    samples_per_class: int = 3,
    seed: int = config.RANDOM_SEED,
) -> Path:
    """Save a grid of original (unaugmented) training images resized to 224x224."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, samples_per_class, figsize=(3.2 * samples_per_class, 2.8 * n_classes)
    )

    for row_idx, class_name in enumerate(class_names):
        class_items = [r for r in train_records if r["class_name"] == class_name]
        chosen_indices = rng.choice(
            len(class_items),
            size=min(samples_per_class, len(class_items)),
            replace=False,
        )
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]
            ax.axis("off")
            if col_idx >= len(chosen_indices):
                continue
            img_path = config.BASE_DIR / class_items[int(chosen_indices[col_idx])]["filepath"]
            with Image.open(img_path) as img:
                resized = img.convert("RGB").resize(config.IMAGE_SIZE)
                ax.imshow(resized)
            if col_idx == 1:
                ax.set_title(format_class_label(class_name), fontsize=10, fontweight="bold")

    fig.suptitle(
        "Sample Training Leaf Images (Resized to 224x224 RGB)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_augmented_samples(
    train_ds: tf.data.Dataset,
    class_names: list[str],
    output_path: Path = config.AUGMENTED_IMAGES_FIG,
    num_images: int = 8,
) -> Path:
    """Save a visual preview of augmented & MobileNetV2-preprocessed training images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_batch, labels_batch = next(iter(train_ds.take(1)))
    count = min(num_images, int(images_batch.shape[0]))
    cols = 4
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx >= count:
            continue
        # Rescale MobileNetV2 [-1, 1] tensors back to [0, 1] solely for plotting
        display_img = np.clip((images_batch[idx].numpy() + 1.0) / 2.0, 0.0, 1.0)
        label_idx = int(labels_batch[idx].numpy())
        ax.imshow(display_img)
        ax.set_title(format_class_label(class_names[label_idx]), fontsize=9)

    fig.suptitle(
        "Training Pipeline Output Preview (Augmented + Rescaled for Display)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def prepare_datasets(
    force_rebuild_manifest: bool = False,
    generate_plots: bool = True,
) -> dict:
    """Prepare manifest splits, build all three tf.data pipelines, and optionally plot."""
    if force_rebuild_manifest or not config.SPLITS_MANIFEST_PATH.exists():
        records, excluded = collect_dataset_records()
        splits = create_stratified_splits(records)
        save_splits_manifest(splits, config.SPLITS_MANIFEST_PATH)
        class_names = sorted(config.SELECTED_CLASSES)
    else:
        splits, class_names = load_splits_manifest(config.SPLITS_MANIFEST_PATH)
        excluded = {c: 0 for c in class_names}

    augmentation_layer = build_augmentation_layer(seed=config.RANDOM_SEED)
    train_ds = build_tf_dataset(
        splits["train"],
        batch_size=config.BATCH_SIZE,
        training=True,
        seed=config.RANDOM_SEED,
        augmentation_layer=augmentation_layer,
    )
    val_ds = build_tf_dataset(
        splits["val"],
        batch_size=config.BATCH_SIZE,
        training=False,
    )
    test_ds = build_tf_dataset(
        splits["test"],
        batch_size=config.BATCH_SIZE,
        training=False,
    )

    figures = {}
    if generate_plots:
        figures["class_distribution"] = plot_class_distribution(splits, class_names)
        figures["sample_images"] = plot_sample_images(splits["train"], class_names)
        figures["augmented_samples"] = plot_augmented_samples(train_ds, class_names)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "splits": splits,
        "class_names": class_names,
        "excluded_counts": excluded,
        "summary": summarize_split_counts(splits, class_names),
        "figures": figures,
    }


if __name__ == "__main__":
    result = prepare_datasets(force_rebuild_manifest=True, generate_plots=True)
    print("Classes:", result["class_names"])
    print("Excluded invalid files:", result["excluded_counts"])
    for split_name in ("train", "val", "test"):
        total_split = len(result["splits"][split_name])
        print(f"{split_name.upper()} ({total_split}): {result['summary'][split_name]}")
    print("Generated figures:", {k: str(v) for k, v in result["figures"].items()})
