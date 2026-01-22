# training/train.py
import os
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc
)
from training.abc_algorithm import ABCAlgorithm
from training.cnn_model import create_cnn_model
import numpy as np
import shutil
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.csv")

# --------------------------------------------------
# Extract frames
# --------------------------------------------------
def extract_frames(
    dataset_dir=DATASET_DIR,
    frame_dir=FRAME_DIR,
    img_size=(224, 224),
    max_frames_per_video=20
):
    start = time.time()

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    for cls in ["real", "fake"]:
        src = os.path.join(dataset_dir, cls)
        dst = os.path.join(frame_dir, cls)
        os.makedirs(dst, exist_ok=True)

        if not os.path.exists(src):
            continue

        for filename in os.listdir(src):
            file_path = os.path.join(src, filename)
            name = os.path.splitext(filename)[0]

            if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                cap = cv2.VideoCapture(file_path)
                count = 0
                while count < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, img_size)
                    cv2.imwrite(
                        os.path.join(dst, f"{name}_{cls}_{count:04d}.jpg"),
                        frame
                    )
                    count += 1
                cap.release()

            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(file_path, dst)

    print(f"[DATA] Frame extraction completed in {time.time() - start:.1f}s")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_dataset(img_size=(224, 224), batch_size=32):
    extract_frames()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen

# --------------------------------------------------
# ABC Objective
# --------------------------------------------------
def build_objective(train_gen, val_gen):
    def objective(params):
        lr = 10 ** params[0]
        dense_units = int(params[1])

        try:
            model = create_cnn_model(
                learning_rate=lr,
                dense_units=dense_units,
                dropout_rate=0.3
            )
            h = model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0)
            loss = float(h.history["val_loss"][-1])
            K.clear_session()
            return loss
        except:
            K.clear_session()
            return float("inf")
    return objective

# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_all(history, y_true, y_pred, y_prob, num_real, num_fake):
    tag = f"{num_real}_{num_fake}"
    plot_dir = os.path.join(MODEL_DIR, f"plots_{tag}")
    os.makedirs(plot_dir, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(plot_dir, f"accuracy_{tag}.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(plot_dir, f"loss_{tag}.png"))
    plt.close()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(plot_dir, f"roc_auc_{tag}.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{tag}.png"))
    plt.close()

    # Precision / Recall / F1
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    plt.figure()
    plt.bar(["Precision", "Recall", "F1"], [p, r, f1])
    plt.ylim(0, 1)
    plt.title("Metrics")
    plt.savefig(os.path.join(plot_dir, f"metrics_{tag}.png"))
    plt.close()

    return roc_auc, p, r, f1, cm

# --------------------------------------------------
# Save CSV log
# --------------------------------------------------
def save_log(row):
    os.makedirs(MODEL_DIR, exist_ok=True)

    header = [
        "timestamp", "num_real", "num_fake",
        "learning_rate", "dense_units",
        "val_loss", "val_accuracy",
        "precision", "recall", "f1_score",
        "roc_auc", "TN", "FP", "FN", "TP",
        "training_time_s"
    ]

    write_header = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    start_time = time.time()

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
        epochs=5,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    val_gen.reset()
    y_true = val_gen.classes
    y_prob = model.predict(val_gen).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    val_accuracy = accuracy_score(y_true, y_pred)
    roc_auc, p, r, f1, cm = plot_all(
        history, y_true, y_pred, y_prob, num_real, num_fake
    )

    tn, fp, fn, tp = cm.ravel()
    training_time = time.time() - start_time

    model_path = os.path.join(
        MODEL_DIR, f"abc_cnn_model_{num_real}_{num_fake}.h5"
    )
    model.save(model_path)

    save_log([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_real, num_fake,
        lr, dense,
        best_loss, val_accuracy,
        p, r, f1,
        roc_auc,
        tn, fp, fn, tp,
        round(training_time, 2)
    ])

    print(f"[SYSTEM] Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()
