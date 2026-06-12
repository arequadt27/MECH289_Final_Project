"""
1D CNN stress classifier for WESAD wrist signals.
Uses raw/preprocessed time-series windows instead of handcrafted features.
Inputs:
    WESAD/S*/S*.pkl
Outputs:
    cnn1d_results.csv
    cnn1d_window_predictions.csv
    cnn1d_confusion_matrix.png
"""
import os
import pickle
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")

# Config
DATA_ROOT = os.getcwd()
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
print(DATA_ROOT)
print(os.listdir(DATA_ROOT))

FS = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
    "labels": 700,
}
TARGET_FS = 64
WINDOW_SEC = 60
OVERLAP = 0.50
VALID_LABELS = {1, 2, 3}
RANDOM_SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 8

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# Load subjects and modalities
def load_subject(subject_id):
    pkl_path = os.path.join(DATA_ROOT, "WESAD", subject_id, f"{subject_id}.pkl")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    wrist = raw["signal"]["wrist"]
    return {
        "ACC": wrist["ACC"].astype(np.float64),
        "BVP": wrist["BVP"].flatten().astype(np.float64),
        "EDA": wrist["EDA"].flatten().astype(np.float64),
        "TEMP": wrist["TEMP"].flatten().astype(np.float64),
        "labels": raw["label"].flatten().astype(np.int32),
    }


# Preprocessing pipeline
def butter_lowpass(sig, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, sig)


def butter_bandpass(sig, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def preprocess_subject(data):
    acc_mag = np.sqrt((data["ACC"] ** 2).sum(axis=1))
    eda = butter_lowpass(data["EDA"], cutoff=1.0, fs=FS["EDA"])
    bvp = butter_bandpass(data["BVP"], low=0.5, high=5.0, fs=FS["BVP"])
    temp = pd.Series(data["TEMP"]).rolling(5, center=True, min_periods=1).mean().to_numpy()
    acc = pd.Series(acc_mag).rolling(5, center=True, min_periods=1).mean().to_numpy()
    return {
        "EDA": eda,
        "BVP": bvp,
        "TEMP": temp,
        "ACC": acc,
        "labels": data["labels"],
    }


# Resampling / labels
def resample_to_target(sig, original_fs, target_fs=TARGET_FS):
    old_t = np.arange(len(sig)) / original_fs
    duration = old_t[-1]
    new_t = np.arange(0, duration, 1 / target_fs)
    return np.interp(new_t, old_t, sig)


def downsample_labels_majority(labels, src_fs, tgt_fs):
    n_src = len(labels)
    n_tgt = int(n_src * tgt_fs / src_fs)
    src_idx = np.arange(n_src, dtype=np.int64)
    tgt_bins = (src_idx * tgt_fs // src_fs).astype(np.int32).clip(0, n_tgt - 1)
    max_lbl = max(1, int(labels.clip(0).max()) + 1)
    out = np.zeros(n_tgt, dtype=np.int32)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(tgt_bins)) + 1, [n_src]))
    for j in range(len(bounds) - 1):
        s, e = bounds[j], bounds[j + 1]
        bin_id = int(tgt_bins[s])
        chunk = labels[s:e].clip(0).astype(np.intp)
        out[bin_id] = int(np.bincount(chunk, minlength=max_lbl).argmax())
    return out


def binarize(raw_labels):
    return (raw_labels == 2).astype(np.int32)


# Windowing
def make_multimodal_windows(proc, subject_id):
    eda = resample_to_target(proc["EDA"], FS["EDA"])
    bvp = resample_to_target(proc["BVP"], FS["BVP"])
    temp = resample_to_target(proc["TEMP"], FS["TEMP"])
    acc = resample_to_target(proc["ACC"], FS["ACC"])
    min_len = min(len(eda), len(bvp), len(temp), len(acc))
    X_signal = np.stack([
        eda[:min_len],
        bvp[:min_len],
        temp[:min_len],
        acc[:min_len],
    ], axis=1)
    raw_labels = downsample_labels_majority(proc["labels"], FS["labels"], TARGET_FS)[:min_len]
    bin_labels = binarize(raw_labels)
    win_len = WINDOW_SEC * TARGET_FS
    step = int(win_len * (1 - OVERLAP))
    X_windows, y_windows, subjects, start_indices = [], [], [], []
    for start in range(0, min_len - win_len + 1, step):
        end = start + win_len
        raw_chunk = raw_labels[start:end]
        raw_majority = int(np.bincount(raw_chunk.clip(0).astype(np.intp)).argmax())
        if raw_majority not in VALID_LABELS:
            continue
        y_chunk = bin_labels[start:end]
        y = int(np.bincount(y_chunk.astype(np.intp)).argmax())
        X_windows.append(X_signal[start:end])
        y_windows.append(y)
        subjects.append(subject_id)
        start_indices.append(start)
    return X_windows, y_windows, subjects, start_indices


# CNN model
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=7, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


# Main
def main():
    print("=" * 70)
    print("1D CNN WESAD Wrist-Based Stress Detection")
    print("=" * 70)

    X_all, y_all, groups_all, starts_all = [], [], [], []
    for sid in SUBJECTS:
        print(f"Loading {sid}...")
        try:
            data = load_subject(sid)
        except FileNotFoundError:
            print(f"Skipping {sid}: file not found")
            continue
        proc = preprocess_subject(data)
        Xw, yw, sw, st = make_multimodal_windows(proc, sid)
        X_all.extend(Xw)
        y_all.extend(yw)
        groups_all.extend(sw)
        starts_all.extend(st)
        print(f"{sid}: {len(yw)} windows, stress={sum(yw)}, non-stress={len(yw)-sum(yw)}")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    groups = np.array(groups_all)
    starts = np.array(starts_all)

    print("\nDataset summary:")
    print(f"X shape: {X.shape}")
    print(f"y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"subjects: {len(np.unique(groups))}")

    logo = LeaveOneGroupOut()
    records = []
    pred_rows = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        test_subject = groups[test_idx[0]]
        print(f"\nFold {fold}/{len(np.unique(groups))}: test subject = {test_subject}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        n_train, t_len, n_ch = X_train.shape
        n_test = X_test.shape[0]
        X_train_flat = X_train.reshape(-1, n_ch)
        X_test_flat = X_test.reshape(-1, n_ch)
        X_train_sc = scaler.fit_transform(X_train_flat).reshape(n_train, t_len, n_ch)
        X_test_sc = scaler.transform(X_test_flat).reshape(n_test, t_len, n_ch)

        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        class_weight = {
            0: 1.0,
            1: n_neg / max(n_pos, 1),
        }

        model = build_cnn(input_shape=(t_len, n_ch))
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5
        )
        model.fit(
            X_train_sc,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        y_prob = model.predict(X_test_sc, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print(
            f"F1={metrics['f1']:.3f} "
            f"Recall={metrics['recall']:.3f} "
            f"Precision={metrics['precision']:.3f} "
            f"PR-AUC={metrics['pr_auc']:.3f} "
            f"Accuracy={metrics['accuracy']:.3f}"
        )

        records.append({
            "subject_id": test_subject,
            **metrics,
            "n_test_windows": len(y_test),
            "n_stress_windows": int(y_test.sum()),
        })
        for idx, prob, pred in zip(test_idx, y_prob, y_pred):
            pred_rows.append({
                "subject_id": groups[idx],
                "window_start_sample_64hz": int(starts[idx]),
                "label": int(y[idx]),
                "cnn_prob": float(prob),
                "cnn_pred": int(pred),
            })
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    results_df = pd.DataFrame(records)
    pred_df = pd.DataFrame(pred_rows)

    print("\n" + "=" * 70)
    print("Mean ± std across subjects")
    print("=" * 70)
    for col in ["f1", "recall", "precision", "pr_auc", "accuracy"]:
        print(f"{col:<10}: {results_df[col].mean():.3f} ± {results_df[col].std():.3f}")

    results_path = os.path.join(DATA_ROOT, "cnn1d_results.csv")
    pred_path = os.path.join(DATA_ROOT, "cnn1d_window_predictions.csv")
    results_df.to_csv(results_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {pred_path}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Stress", "Stress"]
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("1D CNN Aggregated Confusion Matrix — LOSO")
    plt.tight_layout()
    cm_path = os.path.join(DATA_ROOT, "cnn1d_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
