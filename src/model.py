"""MobileNetV2 transfer learning model architecture for AgroVision AI."""

from __future__ import annotations

import tensorflow as tf

from src import config


def build_mobilenetv2_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int] = config.INPUT_SHAPE,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.INITIAL_LEARNING_RATE,
    freeze_base: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build and compile a MobileNetV2 transfer learning image classifier.

    Note:
        `src.data_loader` already normalizes input tensors to [-1, 1] using
        `tf.keras.applications.mobilenet_v2.preprocess_input`. This model does
        not add a second rescaling/preprocessing layer so normalization is
        applied exactly once.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # Pass training=False while base_model is frozen so BatchNorm keeps ImageNet stats.
    x = base_model(inputs, training=False if freeze_base else None)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="agrovision_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_layers(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_from: int = -30,
    learning_rate: float = config.FINE_TUNE_LEARNING_RATE,
) -> tf.keras.Model:
    """Unfreeze top layers of MobileNetV2 (keeping BatchNorm frozen) for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
