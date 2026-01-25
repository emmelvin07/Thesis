# training/train.py
import os
import cv2
import tensorflow as tf
import numpy as np
import shutil
import time
import csv
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from training.abc_algorithm import ABCAlgorithm
from training.cnn_model import create_cnn_model

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
PLOT_DIR = os.path.join(os.path.dirname(BASE_DIR), "plots")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --------------------------------------------------
# Extract frames
# --------------------------------------------------
def extract_frames(max_frames_per_video=20):
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

    for cls in ["real", "fake"]:
        src = os.path.join(DATASET_DIR, cls)
        dst = os.path.join(FRAME_DIR, cls)
        os.makedirs(dst, exist_ok=True)

        for file in os.listdir(src):
            path = os.path.join(src, file)
            name = os.path.splitext(file)[0]

            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                cap = cv2.VideoCapture(path)
                count = 0
                while count < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, IMG_SIZE)
                    cv2.imwrite(os.path.join(dst, f"{name}_{count:04d}.jpg"), frame)
                    count += 1
                cap.release()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_dataset():
    extract_frames()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen

# --------------------------------------------------
# Plot Metrics (NEW)
# --------------------------------------------------
def plot_metrics(history):
    os.makedirs(PLOT_DIR, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    precision = np.array(history.history["precision"])
    recall = np.array(history.history["recall"])
    val_precision = np.array(history.history["val_precision"])
    val_recall = np.array(history.history["val_recall"])

    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

    plots = {
        "accuracy": ("accuracy", "val_accuracy", "Accuracy"),
        "loss": ("loss", "val_loss", "Loss"),
        "precision": ("precision", "val_precision", "Precision"),
        "recall": ("recall", "val_recall", "Recall"),
        "auc": ("auc", "val_auc", "AUC")
    }

    for name, (train_key, val_key, title) in plots.items():
        plt.figure()
        plt.plot(epochs, history.history[train_key], label=f"Train {title}")
        plt.plot(epochs, history.history[val_key], label=f"Val {title}")
        plt.xlabel("Epochs")
        plt.ylabel(title)
        plt.title(f"{title} vs Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"))
        plt.show()

    plt.figure()
    plt.plot(epochs, f1, label="Train F1-score")
    plt.plot(epochs, val_f1, label="Val F1-score")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "f1_score.png"))
    plt.show()

# --------------------------------------------------
# ABC Objective
# --------------------------------------------------
def build_objective(train_gen, val_gen):
    def objective(params):
        log_lr, dense = params
        lr = 10 ** log_lr
        dense = int(dense)

        try:
            tf.keras.backend.clear_session()

            model = create_cnn_model(
                learning_rate=lr,
                dense_units=dense,
                dropout_rate=0.3
            )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=1,
                steps_per_epoch=min(10, len(train_gen)),
                validation_steps=min(5, len(val_gen)),
                verbose=0
            )

            loss = history.history["val_loss"][-1]
            return float(loss) if np.isfinite(loss) else 1e6

        except Exception as e:
            print("[OBJ ERROR]", e)
            return 1e6

    return objective

# --------------------------------------------------
# Save CSV Log
# --------------------------------------------------
def save_log(row):
    header = [
        "timestamp", "num_real", "num_fake",
        "learning_rate", "dense_units",
        "val_loss", "val_accuracy",
        "precision", "recall", "f1_score",
        "roc_auc", "TN", "FP", "FN", "TP",
        "training_time_s"
    ]

    write_header = not os.path.exists(LOG_FILE)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    start = time.time()

    train_gen, val_gen = load_dataset()
    num_real = len(os.listdir(os.path.join(DATASET_DIR, "real")))
    num_fake = len(os.listdir(os.path.join(DATASET_DIR, "fake")))

    abc = ABCAlgorithm(
        num_bees=14,
        limit=7,
        max_iter=8,
        bounds=[(-4, -2), (32, 256)],
        rng_seed=42
    )

    best_params, best_loss = abc.optimize(build_objective(train_gen, val_gen))

    lr = 10 ** best_params[0]
    dense = int(best_params[1])

    model = create_cnn_model(
        learning_rate=lr,
        dense_units=dense,
        dropout_rate=0.3
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    plot_metrics(history)

    val_gen.reset()
    y_true = val_gen.classes
    y_prob = model.predict(val_gen).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    model_path = os.path.join(MODEL_DIR, f"abc_cnn_model_{num_real}_{num_fake}.h5")
    model.save(model_path)

    save_log([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_real, num_fake,
        lr, dense,
        best_loss, acc,
        p, r, f1, roc,
        tn, fp, fn, tp,
        round(time.time() - start, 2)
    ])

    print(f"\n[SYSTEM] Training complete. Model saved to {model_path}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
