# -*- coding: utf-8 -*-
"""
Binary 1D-CNN with sequence windows + paper-ready evaluation plots/tables.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from io import StringIO

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
np.random.seed(1)
tf.random.set_seed(2)

# --------------------------------------------------
# Constants / paths
# --------------------------------------------------
SEQ_LEN = 10
CSV_FILE = "splite/train.csv"
OUTDIR = "paper_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# Load
# --------------------------------------------------
raw = pd.read_csv(CSV_FILE)

# Labels
labels = raw["Label"].astype(np.float32).values

# Features (drop label)
features = raw.drop(columns=["Label"]).copy()

# --------------------------------------------------
# Column-aware parsing (NEW)
#   - Hex for ID and all D* byte columns
#   - DLC: auto-detect hex vs decimal
#   - Timestamp: decimal
#   - Others: decimal
# --------------------------------------------------
def parse_hex(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    if s.lower().startswith("0x"): s = s[2:]
    return float(int(s, 16))

def parse_dec(x):
    return pd.to_numeric(x, errors="coerce")

def looks_hex_series(s: pd.Series) -> bool:
    """True if any value clearly looks hex (has A–F or 0x prefix)."""
    s = s.dropna().astype(str).str.strip()
    if s.empty: return False
    return bool(s.str.contains(r'^(0x|0X)|[A-Fa-f]', regex=True).any())

# columns that should be hex
hex_cols = [c for c in features.columns if c == "ID" or c.upper().startswith("D")]
dec_cols = [c for c in features.columns if c not in hex_cols]

# Timestamp: decimal
if "Timestamp" in features.columns:
    features["Timestamp"] = parse_dec(features["Timestamp"])

# DLC: detect; default to decimal (classic CAN 0..8)
if "DLC" in features.columns:
    if looks_hex_series(features["DLC"]):
        features["DLC"] = features["DLC"].apply(parse_hex)
    else:
        features["DLC"] = parse_dec(features["DLC"])
    # sanity: DLC should be <= 15 (CAN FD up to 64 bytes encoded)
    if features["DLC"].max() > 15:
        raise ValueError("DLC parsed >15; check CSV format/base.")

# Remaining decimals (if any)
for c in dec_cols:
    if c not in ("Timestamp", "DLC"):
        features[c] = parse_dec(features[c])

# Hex for ID and bytes
features[hex_cols] = features[hex_cols].applymap(parse_hex)

# Finish up
features = features.fillna(0).astype(np.float32)

# Sanity checks
byte_cols = [c for c in hex_cols if c.upper().startswith("D")]
if byte_cols:
    assert features[byte_cols].max().max() <= 255, "Parsed byte >255 → parsing bug"
# Allow up to 29-bit extended IDs; tighten if you expect only 11-bit
assert features["ID"].max() <= 0x1FFFFFFF, "Unexpected ID range (check 11/29-bit and parsing)"

# --------------------------------------------------
# Sequence builder (no scaling yet)
# --------------------------------------------------
def create_sequences(df, labels, seq_len):
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
        y.append(labels[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_raw, y = create_sequences(features, labels, SEQ_LEN)
num_features = X_raw.shape[2]

# --------------------------------------------------
# Stratified split (on flattened window)
# --------------------------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for tr_idx, te_idx in sss.split(X_raw.reshape(X_raw.shape[0], -1), y):
    X_train, X_test = X_raw[tr_idx], X_raw[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

# --------------------------------------------------
# Scale AFTER split (fit on training only)
# --------------------------------------------------
scaler = MinMaxScaler()
nfeat = X_train.shape[2]
X_train_flat = X_train.reshape(-1, nfeat)
X_test_flat  = X_test.reshape(-1, nfeat)

scaler.fit(X_train_flat)
X_train = scaler.transform(X_train_flat).reshape(-1, SEQ_LEN, nfeat)
X_test  = scaler.transform(X_test_flat).reshape(-1, SEQ_LEN, nfeat)

# --------------------------------------------------
# Emit C++ scaling vectors (per-frame min/max, tiled for SEQ_LEN)
# --------------------------------------------------
per_frame_min = scaler.data_min_.astype(np.float64)
per_frame_max = scaler.data_max_.astype(np.float64)
per_frame = per_frame_min.shape[0]

cpp_min_vector = np.tile(per_frame_min, SEQ_LEN)
cpp_max_vector = np.tile(per_frame_max, SEQ_LEN)

def emit_vec_block(name, arr, per_row):
    lines = []
    lines.append(f"std::vector<float> {name} = {{")
    for i in range(0, len(arr), per_row):
        row = arr[i:i+per_row]
        row_txt = ", ".join(f"{float(x):.8f}f" for x in row)
        tail = "," if i + per_row < len(arr) else ""
        lines.append(f"    {row_txt}{tail}")
    lines.append("};")
    return "\n".join(lines) + "\n"

hdr = StringIO()
hdr.write(f"// Per-frame features = {per_frame} ; Window len = {SEQ_LEN*per_frame}\n")
hdr.write("// Columns (per frame) in this order:\n")
for idx, col in enumerate(features.columns):
    hdr.write(f"//  [{idx:02d}] {col}\n")
hdr.write("\n")
hdr.write(emit_vec_block("min_vals", cpp_min_vector, per_frame))
hdr.write("\n")
hdr.write(emit_vec_block("max_vals", cpp_max_vector, per_frame))

with open(os.path.join(OUTDIR, "scaling_vectors.hpp"), "w", encoding="utf-8") as f:
    f.write(hdr.getvalue())

print(f'Wrote clean vectors to "{os.path.join(OUTDIR, "scaling_vectors.hpp")}".')

# --------------------------------------------------
# (Optional) Save flat CSVs for your C++ pipeline
# --------------------------------------------------
pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).to_csv(os.path.join(OUTDIR, "CNN_X_train.csv"), index=False, header=False)
pd.DataFrame(X_test.reshape(X_test.shape[0], -1)).to_csv(os.path.join(OUTDIR, "CNN_X_test.csv"), index=False, header=False)
pd.DataFrame(y_train).to_csv(os.path.join(OUTDIR, "CNN_y_train.csv"), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(OUTDIR, "CNN_y_test.csv"), index=False, header=False)

# --------------------------------------------------
# CNN Model
# --------------------------------------------------
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(16, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_cnn((SEQ_LEN, num_features))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_split=0.1, verbose=1)

# --------------------------------------------------
# Save model
# --------------------------------------------------
model.save(os.path.join(OUTDIR, "binary_classification_model_cnn3.keras"))

# ---------------- Paper-ready Evaluation & Plots ----------------
def _save_lineplot(x, ys, labels, title, xlabel, ylabel, path):
    plt.figure()
    for yv, lab in zip(ys, labels):
        plt.plot(x, yv, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# Training curves
epochs = np.arange(1, len(history.history["accuracy"]) + 1)
_save_lineplot(
    epochs,
    [history.history["accuracy"], history.history["val_accuracy"]],
    ["Train", "Validation"],
    "Accuracy vs Epoch",
    "Epoch", "Accuracy",
    os.path.join(OUTDIR, "accuracy.png")
)
_save_lineplot(
    epochs,
    [history.history["loss"], history.history["val_loss"]],
    ["Train", "Validation"],
    "Loss vs Epoch",
    "Epoch", "Binary Cross-Entropy",
    os.path.join(OUTDIR, "loss.png")
)

# Test predictions
y_true = np.asarray(y_test).ravel().astype(int)
y_proba = model.predict(X_test, batch_size=1024, verbose=0).ravel()
y_pred_05 = (y_proba >= 0.5).astype(int)

# Confusion matrices
def _plot_confusion(cm, class_names, title, path, normalize=False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

cm = confusion_matrix(y_true, y_pred_05, labels=[0, 1])
_plot_confusion(cm, ["0", "1"], "Confusion Matrix (thr=0.5)",
                os.path.join(OUTDIR, "confusion_matrix_counts.png"))
_plot_confusion(cm, ["0", "1"], "Confusion Matrix (Normalized, thr=0.5)",
                os.path.join(OUTDIR, "confusion_matrix_normalized.png"), normalize=True)

# ROC curve & AUC
fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "roc_curve.png"), dpi=300)
plt.close()

# Precision-Recall curve & Average Precision
prec, rec, pr_thresholds = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)

plt.figure()
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Test)")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pr_curve.png"), dpi=300)
plt.close()

# Calibration (reliability) curve
prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot(prob_pred, prob_true, marker="o")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve (Test)")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "calibration_curve.png"), dpi=300)
plt.close()

# Threshold by Youden's J
j_scores = tpr - fpr
best_j_ix = int(np.argmax(j_scores))
best_thr = float(roc_thresholds[best_j_ix])
y_pred_best = (y_proba >= best_thr).astype(int)
cm_best = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
_plot_confusion(cm_best, ["0", "1"],
                f"Confusion Matrix (thr={best_thr:.3f})",
                os.path.join(OUTDIR, "confusion_matrix_bestJ.png"))

# Scalar metrics table
metrics = {
    "AUC_ROC": float(roc_auc_score(y_true, y_proba)),
    "AP_PR": float(average_precision_score(y_true, y_proba)),
    "Accuracy@0.5": float(accuracy_score(y_true, y_pred_05)),
    "BalancedAcc@0.5": float(balanced_accuracy_score(y_true, y_pred_05)),
    "Precision@0.5": float(precision_score(y_true, y_pred_05, zero_division=0)),
    "Recall@0.5": float(recall_score(y_true, y_pred_05, zero_division=0)),
    "F1@0.5": float(f1_score(y_true, y_pred_05, zero_division=0)),
    "Brier": float(brier_score_loss(y_true, y_proba)),
    "BestJ_Threshold": best_thr,
    "Accuracy@BestJ": float(accuracy_score(y_true, y_pred_best)),
    "BalancedAcc@BestJ": float(balanced_accuracy_score(y_true, y_pred_best)),
    "Precision@BestJ": float(precision_score(y_true, y_pred_best, zero_division=0)),
    "Recall@BestJ": float(recall_score(y_true, y_pred_best, zero_division=0)),
    "F1@BestJ": float(f1_score(y_true, y_pred_best, zero_division=0)),
}
pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, "metrics_summary.csv"), index=False)

# Classification report
report = classification_report(y_true, y_pred_05, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUTDIR, "classification_report_thr0.5.csv"))

# Per-sample predictions
pred_df = pd.DataFrame({
    "y_true": y_true,
    "y_proba": y_proba,
    "y_pred_thr0.5": y_pred_05,
    "y_pred_bestJ": y_pred_best
})
pred_df.to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)

print(f"Saved figures & tables to: {OUTDIR}")
