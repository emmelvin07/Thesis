# training/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_cnn_model(input_shape=(128, 128, 3),
                     dense_units=128,
                     dropout_rate=0.3,
                     learning_rate=1e-3):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),

        # 🔴 CRITICAL CHANGE
        layers.GlobalAveragePooling2D(),

        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(
        f"[CNN] Model compiled "
        f"(filters=32,64,128,256, dense={dense_units}, "
        f"dropout={dropout_rate}, lr={learning_rate})"
    )

    return model