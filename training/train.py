# training/train.py
import os
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from training.abc_algorithm import ABCAlgorithm
from training.cnn_model import create_cnn_model
import numpy as np
import shutil
import time
import csv
from datetime import datetime

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.csv")

# ---------------------------------------------------------
# Extract frames from videos / images
# ---------------------------------------------------------
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

    classes = ["real", "fake"]

    for cls in classes:
        src = os.path.join(dataset_dir, cls)
        dst = os.path.join(frame_dir, cls)
        os.makedirs(dst, exist_ok=True)

        if not os.path.exists(src):
            print(f"[WARN] Missing folder: {src}")
            continue

        for filename in os.listdir(src):
            file_path = os.path.join(src, filename)
            video_name = os.path.splitext(filename)[0]

            # Video → frames
            if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                cap = cv2.VideoCapture(file_path)
                count = 0
                while count < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, img_size)
                    outname = f"{video_name}_{cls}_{count:04d}.jpg"
                    cv2.imwrite(os.path.join(dst, outname), frame)
                    count += 1
                cap.release()

            # Images → copy
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(file_path, dst)

    elapsed = time.time() - start
    print(f"[DATA] Frame extraction completed in {elapsed:.1f}s")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_dataset(img_size=(224, 224), batch_size=32):
    extract_frames()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        FRAME_DIR,
        class_mode="binary",
        target_size=img_size,
        batch_size=batch_size,
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        FRAME_DIR,
        class_mode="binary",
        target_size=img_size,
        batch_size=batch_size,
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen

# ---------------------------------------------------------
# ABC Objective Function
# ---------------------------------------------------------
def build_objective(train_gen, val_gen):
    def objective(params):
        log_lr = params[0]
        lr = 10 ** log_lr
        dense_units = int(params[1])

        try:
            model = create_cnn_model(
                learning_rate=lr,
                dense_units=dense_units,
                dropout_rate=0.3
            )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=1,
                verbose=0
            )

            val_loss = float(history.history["val_loss"][-1])
            K.clear_session()
            return val_loss

        except Exception as e:
            print("[OBJ ERROR]", e)
            K.clear_session()
            return float("inf")

    return objective

# ---------------------------------------------------------
# Count dataset videos
# ---------------------------------------------------------
def count_videos(dataset_dir=DATASET_DIR):
    num_real = len(os.listdir(os.path.join(dataset_dir, "real")))
    num_fake = len(os.listdir(os.path.join(dataset_dir, "fake")))
    return num_real, num_fake

# ---------------------------------------------------------
# Save results to CSV (FIXED STRUCTURE)
# ---------------------------------------------------------
def save_log(log_data):
    os.makedirs(MODEL_DIR, exist_ok=True)

    header = [
        "timestamp",
        "num_real",
        "num_fake",
        "learning_rate",
        "dense_units",
        "val_loss",
        "val_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "TN",
        "FP",
        "FN",
        "TP",
        "training_time_s"
    ]

    write_header = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        writer.writerow([
            log_data[0],                  # timestamp
            int(log_data[1]),             # num_real
            int(log_data[2]),             # num_fake
            f"{log_data[3]:.6f}",          # learning_rate
            int(log_data[4]),             # dense_units
            f"{log_data[5]:.6f}",          # val_loss
            f"{log_data[6]:.4f}",          # val_accuracy
            f"{log_data[7]:.4f}",          # precision
            f"{log_data[8]:.4f}",          # recall
            f"{log_data[9]:.4f}",          # f1_score
            f"{log_data[10]:.6f}",         # roc_auc
            int(log_data[11]),             # TN
            int(log_data[12]),             # FP
            int(log_data[13]),             # FN
            int(log_data[14]),             # TP
            f"{log_data[15]:.2f}"          # training_time_s
        ])

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def main():
    start_time = time.time()

    train_gen, val_gen = load_dataset()
    print(f"[DATA] Training samples: {train_gen.samples}")
    print(f"[DATA] Validation samples: {val_gen.samples}")

    num_real, num_fake = count_videos()
    print(f"[DATA] Dataset size: {num_real} real, {num_fake} fake")

    bounds = [
        (-4, -2),     # log10(learning rate)
        (32, 256)     # dense units
    ]


    abc = ABCAlgorithm(
        num_bees=12,
        limit=6,
        max_iter=7,
        bounds=bounds,
        rng_seed=42
    )

    objective_fn = build_objective(train_gen, val_gen)
    best_params, best_score = abc.optimize(objective_fn)

    lr = 10 ** best_params[0]
    dense_units = int(best_params[1])

    print("\n[RESULTS]")
    print(f"Best LR = {lr:.6f}")
    print(f"Dense units = {dense_units}")
    print(f"Validation loss = {best_score:.6f}")

    # Final model training
    final_model = create_cnn_model(
        learning_rate=lr,
        dense_units=dense_units,
        dropout_rate=0.3
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    final_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluation
    val_gen.reset()
    y_true = val_gen.classes
    y_pred_prob = final_model.predict(val_gen, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    val_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except:
        roc_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    training_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"abc_cnn_model_{num_real}_{num_fake}.h5"
    )
    final_model.save(model_path)
    print(f"\n[SYSTEM] Saved model to: {model_path}")

    # Save CSV log
    log_data = [
        timestamp,
        num_real,
        num_fake,
        lr,
        dense_units,
        best_score,
        val_accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        tn,
        fp,
        fn,
        tp,
        round(training_time, 2)
    ]

    save_log(log_data)
    print(f"[SYSTEM] Logged results to: {LOG_FILE}")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
