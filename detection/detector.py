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
IMG_SIZE = (224, 224)  # Match latest CNN model input

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------
# Model selection
# ------------------------------------------------------
def list_models():
    return sorted(glob.glob(os.path.join(MODEL_DIR, "abc_cnn_model_*.h5")))

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
print(f"[SYSTEM] Loading model: {MODEL_PATH}")
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

    if preds:
        avg_score = float(np.mean(preds))
        label = "FAKE" if avg_score >= 0.5 else "REAL"
    else:
        avg_score = 0.0
        label = "UNKNOWN"

    return label, avg_score

# ------------------------------------------------------
# MAIN TESTING LOOP
# ------------------------------------------------------
if __name__ == "__main__":
    print("=== TESTING PHASE ===\n")

    results = []
    y_true = []
    y_pred = []
    y_scores = []

    # --------------------------------------------------
    # Gather all videos with their true labels
    # --------------------------------------------------
    all_videos = []
    for true_label in ["real", "fake"]:
        folder = os.path.join(INPUT_DIR, true_label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(folder, file)
                all_videos.append((video_path, true_label))

    if not all_videos:
        print("[WARNING] No videos found in detector_input/real or detector_input/fake.")
    else:
        # Shuffle entire dataset (real + fake)
        random.shuffle(all_videos)

        # --------------------------------------------------
        # Run predictions
        # --------------------------------------------------
        for video_path, true_label in all_videos:
            pred_label, score = predict_video(video_path)

            true = 1 if true_label == "fake" else 0
            pred = 1 if pred_label == "FAKE" else 0

            y_true.append(true)
            y_pred.append(pred)
            y_scores.append(score)

            correct = "YES" if true == pred else "NO"

            results.append([
                os.path.basename(video_path),
                true_label.upper(),
                pred_label,
                round(score, 4),
                correct
            ])

            # ---------------- VERBOSE TEST CASE OUTPUT ----------------
            print(f"Test Case   : {os.path.basename(video_path)}")
            print(f"Expected    : {true_label.upper()}")
            print(f"Actual      : {pred_label}")
            print(f"Score       : {score:.4f}")
            print(f"Correct     : {correct}\n")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    total_videos = len(y_true)
    correct_predictions = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])

    if total_videos > 0:
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores)
    else:
        cm = np.array([[]])
        acc = precision = recall = f1 = roc_auc = 0.0

    # --------------------------------------------------
    # Save CSV Results
    # --------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "True Label", "Predicted Label", "Score", "Correct"])
        writer.writerows(results)

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== TEST SUMMARY ===\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"Total Videos Evaluated: {total_videos}\n")
        f.write(f"Correct Predictions   : {correct_predictions}\n")
        f.write(f"Accuracy  : {acc * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall * 100:.2f}%\n")
        f.write(f"F1-Score  : {f1 * 100:.2f}%\n")
        f.write(f"ROC-AUC   : {roc_auc * 100:.2f}%\n")

    # --------------------------------------------------
    # Console Output
    # --------------------------------------------------
    print("\n=== TEST SUMMARY ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Total Videos Evaluated: {total_videos}")
    print(f"Correct Predictions   : {correct_predictions}")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
    print(f"\nResults saved in: {RESULTS_DIR}")
