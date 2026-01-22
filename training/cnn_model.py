# training/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

def create_cnn_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Create a CNN model for binary classification.

    Architecture:
    - 4 convolutional blocks with BatchNormalization and ReLU
    - GlobalAveragePooling2D instead of Flatten
    - Dense layer with batch normalization and dropout
    - Output layer with sigmoid for binary classification

    Returns:
        Compiled Keras Model
    """

    weight_decay = 1e-4  # L2 regularization for Conv and Dense layers

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Conv Block 2
        layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Conv Block 3
        layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Conv Block 4
        layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),

        # Fully connected
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
        layers.BatchNormalization(),

        # Output
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print(
        f"[CNN] Model compiled "
        f"(filters=32,64,128,256, dense={dense_units}, "
        f"dropout={dropout_rate}, lr={learning_rate})"
    )

    return model
