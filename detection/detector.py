import os
import cv2
import numpy as np
import tensorflow as tf
import glob
import csv
import random

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
INPUT_DIR = os.path.join(BASE_DIR, "detector_input")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = (128, 128)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------
# Console table helper
# ------------------------------------------------------
def print_table(rows):
    headers = ["TestCase", "Video", "Expected", "Actual", "Result"]
    widths = [10, 30, 10, 10, 10]

    def fmt(row):
        return " | ".join(str(col).ljust(w) for col, w in zip(row, widths))

    print(fmt(headers))
    print("-" * sum(widths))
    for r in rows:
        print(fmt(r))

# ------------------------------------------------------
# Model selection
# ------------------------------------------------------
def list_models():
    return sorted(glob.glob(os.path.join(MODEL_DIR, "abc_cnn_model*.h5")))

def select_model():
    models = list_models()
    if not models:
        raise FileNotFoundError("No trained models found.")

    print("Available models:")
    for i, m in enumerate(models):
        print(f"{i+1}: {os.path.basename(m)}")
    print("0: Use latest model")

    choice = input("Select model number: ").strip()
    try:
        choice = int(choice)
        if choice == 0:
            return max(models, key=os.path.getmtime)
        return models[choice - 1]
    except:
        return max(models, key=os.path.getmtime)

MODEL_PATH = select_model()
print(f"\n[SYSTEM] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[SYSTEM] Model loaded successfully!\n")

# ------------------------------------------------------
# Video prediction
# ------------------------------------------------------
def predict_video(video_path, sample_frames=20):
    cap = cv2.VideoCapture(video_path)
    preds = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // sample_frames)

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        idx += step

        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame, verbose=0)[0][0]
        preds.append(pred)

    cap.release()

    avg_score = float(np.mean(preds))
    label = "FAKE" if avg_score >= 0.5 else "REAL"
    return label, avg_score

# ------------------------------------------------------
# MAIN TESTING LOOP
# ------------------------------------------------------
if __name__ == "__main__":
    print("=== TESTING PHASE ===\n")

    results = []
    table_rows = []

    y_true = []
    y_pred = []
    y_scores = []

    # --------------------------------------------------
    # Collect videos
    # --------------------------------------------------
    all_videos = []
    for true_label in ["real", "fake"]:
        folder = os.path.join(INPUT_DIR, true_label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                all_videos.append((os.path.join(folder, file), true_label))

    random.shuffle(all_videos)

    # --------------------------------------------------
    # Run predictions
    # --------------------------------------------------
    for idx, (video_path, true_label) in enumerate(all_videos, start=1):
        pred_label, score = predict_video(video_path)

        expected = true_label.upper()
        actual = pred_label
        result = "PASSED" if expected == actual else "FAILED"
        test_case = f"TC{idx:02d}"

        true = 1 if expected == "FAKE" else 0
        pred = 1 if actual == "FAKE" else 0

        y_true.append(true)
        y_pred.append(pred)
        y_scores.append(score)

        results.append([
            test_case,
            os.path.basename(video_path),
            expected,
            actual,
            result,
            round(score, 4)
        ])

        table_rows.append([
            test_case,
            os.path.basename(video_path),
            expected,
            actual,
            result
        ])

        print(
            f"{test_case} | {os.path.basename(video_path)} | "
            f"Expected: {expected} | Actual: {actual} | "
            f"Result: {result} | Score: {score:.4f}"
        )

    # --------------------------------------------------
    # Print table
    # --------------------------------------------------
    print("\n=== PER-VIDEO TEST RESULTS ===\n")
    print_table(table_rows)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TestCase",
            "Video",
            "Expected Output",
            "Actual Output",
            "Result",
            "Score"
        ])
        writer.writerows(results)

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== TEST SUMMARY ===\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"Accuracy  : {acc * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall * 100:.2f}%\n")
        f.write(f"F1-Score  : {f1 * 100:.2f}%\n")
        f.write(f"ROC-AUC   : {roc_auc * 100:.2f}%\n")

    # --------------------------------------------------
    # Final console summary
    # --------------------------------------------------
    print("\n=== TEST SUMMARY ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
    print(f"\nResults saved in: {RESULTS_DIR}")
