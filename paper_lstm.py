#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM with sequence windows + paper-ready evaluation/plots (CNN-matched).
Outputs saved under paper_outputs_lstm/:
  - accuracy.png, loss.png
  - roc_curve.png, pr_curve.png, calibration_curve.png
  - confusion_matrix_counts.png, confusion_matrix_normalized.png, confusion_matrix_bestJ.png
  - metrics_summary.csv, classification_report_thr0.5.csv, test_predictions.csv
  - LSTM_X_train.csv, LSTM_X_test.csv, LSTM_y_train.csv, LSTM_y_test.csv
  - scaling_vectors_lstm.hpp
  - binary_classification_model_lstm.keras
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

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
from io import StringIO

# ---------------- Reproducibility ----------------
np.random.seed(1)
tf.random.set_seed(2)

# ---------------- Constants / paths ----------------
SEQ_LEN  = 10
CSV_FILE = "splite/train.csv"
OUTDIR   = "paper_outputs_lstm"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Column-aware CAN parsing ----------------
def parse_hex(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    if s.lower().startswith("0x"): s = s[2:]
    return float(int(s, 16))

def parse_dec(x):
    return pd.to_numeric(x, errors="coerce")

def looks_hex_series(s: pd.Series) -> bool:
    s = s.dropna().astype(str).str.strip()
    if s.empty: return False
    return bool(s.str.contains(r'^(0x|0X)|[A-Fa-f]', regex=True).any())

# ---------------- Load ----------------
raw = pd.read_csv(CSV_FILE)
labels = raw["Label"].astype(np.float32).values
features = raw.drop(columns=["Label"]).copy()

# Hex for ID and D*; others decimal; Timestamp always decimal; DLC auto-detect
hex_cols = [c for c in features.columns if c == "ID" or c.upper().startswith("D")]
dec_cols = [c for c in features.columns if c not in hex_cols]

if "Timestamp" in features.columns:
    features["Timestamp"] = parse_dec(features["Timestamp"])

if "DLC" in features.columns:
    features["DLC"] = features["DLC"].apply(parse_hex if looks_hex_series(features["DLC"]) else parse_dec)
    if features["DLC"].max() > 15:
        raise ValueError("DLC parsed >15; check base/format.")

for c in dec_cols:
    if c not in ("Timestamp", "DLC"):
        features[c] = parse_dec(features[c])

features[hex_cols] = features[hex_cols].applymap(parse_hex)
features = features.fillna(0).astype(np.float32)

# ---------------- Sequence builder ----------------
def create_sequences(df, labels, seq_len):
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
        y.append(labels[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_raw, y = create_sequences(features, labels, SEQ_LEN)
num_features = X_raw.shape[2]

# ---------------- Stratified split ----------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for tr_idx, te_idx in sss.split(X_raw.reshape(X_raw.shape[0], -1), y):
    X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]
    y_train,      y_test    = y[tr_idx],    y[te_idx]

# ---------------- Scale AFTER split (fit on train only) ----------------
scaler = MinMaxScaler()
nfeat = X_train_raw.shape[2]
X_train_flat = X_train_raw.reshape(-1, nfeat)
X_test_flat  = X_test_raw.reshape(-1, nfeat)

scaler.fit(X_train_flat)
X_train = scaler.transform(X_train_flat).reshape(-1, SEQ_LEN, nfeat)
X_test  = scaler.transform(X_test_flat).reshape(-1, SEQ_LEN, nfeat)

# ---------------- Emit C++ scaling vectors (tiled over SEQ_LEN) ----------------
per_frame_min = scaler.data_min_.astype(np.float64)
per_frame_max = scaler.data_max_.astype(np.float64)
cpp_min_vector = np.tile(per_frame_min, SEQ_LEN)
cpp_max_vector = np.tile(per_frame_max, SEQ_LEN)

def emit_vec_block(name, arr, per_row):
    lines = [f"std::vector<float> {name} = {{"]
    for i in range(0, len(arr), per_row):
        row = arr[i:i+per_row]
        row_txt = ", ".join(f"{float(x):.8f}f" for x in row)
        tail = "," if i + per_row < len(arr) else ""
        lines.append(f"    {row_txt}{tail}")
    lines.append("};")
    return "\n".join(lines) + "\n"

hdr = StringIO()
hdr.write(f"// Per-frame features = {nfeat} ; Window len = {SEQ_LEN*nfeat}\n")
hdr.write("// Columns (per frame) in this order:\n")
for idx, col in enumerate(features.columns):
    hdr.write(f"//  [{idx:02d}] {col}\n")
hdr.write("\n")
hdr.write(emit_vec_block("min_vals", cpp_min_vector, nfeat))
hdr.write("\n")
hdr.write(emit_vec_block("max_vals", cpp_max_vector, nfeat))

with open(os.path.join(OUTDIR, "scaling_vectors_lstm.hpp"), "w", encoding="utf-8") as f:
    f.write(hdr.getvalue())

# ---------------- Save flat CSVs ----------------
pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).to_csv(os.path.join(OUTDIR, "LSTM_X_train.csv"), index=False, header=False)
pd.DataFrame(X_test.reshape(X_test.shape[0], -1)).to_csv(os.path.join(OUTDIR, "LSTM_X_test.csv"), index=False, header=False)
pd.DataFrame(y_train).to_csv(os.path.join(OUTDIR, "LSTM_y_train.csv"), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(OUTDIR, "LSTM_y_test.csv"), index=False, header=False)

# ---------------- LSTM model ----------------
def build_lstm(input_shape):
    inputs = Input(shape=input_shape)                 # (SEQ_LEN, num_features)
    x = LSTM(32, activation="tanh", return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(16, activation="tanh", return_sequences=False)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_lstm((SEQ_LEN, num_features))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=200, batch_size=512, validation_split=0.1, verbose=1)

# ---------------- Save model ----------------
model.save(os.path.join(OUTDIR, "binary_classification_model_lstm.keras"))

# ---------------- Plots & evaluation (CNN-matched) ----------------
def _save_lineplot(x, ys, labels, title, xlabel, ylabel, path):
    plt.figure()
    for yv, lab in zip(ys, labels):
        plt.plot(x, yv, label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.grid(True, linestyle=":")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

epochs = np.arange(1, len(history.history["accuracy"]) + 1)
_save_lineplot(
    epochs,
    [history.history["accuracy"], history.history["val_accuracy"]],
    ["Train", "Validation"],
    "Accuracy vs Epoch (LSTM)",
    "Epoch", "Accuracy",
    os.path.join(OUTDIR, "accuracy.png")
)
_save_lineplot(
    epochs,
    [history.history["loss"], history.history["val_loss"]],
    ["Train", "Validation"],
    "Loss vs Epoch (LSTM)",
    "Epoch", "Binary Cross-Entropy",
    os.path.join(OUTDIR, "loss.png")
)

def _plot_confusion(cm, class_names, title, path, normalize=False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
        cm = cm.astype(float) / row_sums; fmt = ".2f"
    else:
        fmt = "d"
    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names)); plt.xticks(ticks, class_names); plt.yticks(ticks, class_names)
    thresh = cm.max()/2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# Test predictions
y_true  = y_test.astype(int)
y_proba = model.predict(X_test, batch_size=1024, verbose=0).ravel()
y_pred_05 = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred_05, labels=[0,1])
_plot_confusion(cm, ["0","1"], "Confusion Matrix (thr=0.5, LSTM)",
                os.path.join(OUTDIR, "confusion_matrix_counts.png"))
_plot_confusion(cm, ["0","1"], "Confusion Matrix (Normalized, thr=0.5, LSTM)",
                os.path.join(OUTDIR, "confusion_matrix_normalized.png"), normalize=True)

# ROC / PR / Calibration
fpr, tpr, roc_thr = roc_curve(y_true, y_proba); rocAUC = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {rocAUC:.3f}"); plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (LSTM)"); plt.legend(); plt.grid(True, linestyle=":")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "roc_curve.png"), dpi=300); plt.close()

prec, rec, pr_thr = precision_recall_curve(y_true, y_proba); AP = average_precision_score(y_true, y_proba)
plt.figure(); plt.plot(rec, prec, label=f"AP = {AP:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve (LSTM)")
plt.legend(); plt.grid(True, linestyle=":"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pr_curve.png"), dpi=300); plt.close()

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(prob_pred, prob_true, marker="o")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration Curve (LSTM)"); plt.grid(True, linestyle=":")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "calibration_curve.png"), dpi=300); plt.close()

# Best Youden's J
J = tpr - fpr
best_j_ix = int(np.argmax(J))
best_thr = float(roc_thr[best_j_ix])
y_pred_best = (y_proba >= best_thr).astype(int)
cm_best = confusion_matrix(y_true, y_pred_best, labels=[0,1])
_plot_confusion(cm_best, ["0","1"], f"Confusion Matrix (thr={best_thr:.3f}, LSTM)",
                os.path.join(OUTDIR, "confusion_matrix_bestJ.png"))

# Metrics dump
metrics = {
    "AUC_ROC": float(roc_auc_score(y_true, y_proba)),
    "AP_PR": float(AP),
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

# Classification report & per-sample predictions
pd.DataFrame(classification_report(y_true, y_pred_05, output_dict=True, zero_division=0))\
  .transpose().to_csv(os.path.join(OUTDIR, "classification_report_thr0.5.csv"))
pd.DataFrame({"y_true": y_true, "y_proba": y_proba,
              "y_pred_thr0.5": y_pred_05, "y_pred_bestJ": y_pred_best})\
  .to_csv(os.path.join(OUTDIR, "test_predictions.csv"), index=False)

# Pretty print key metrics
def print_line(tag, m):
    print(f"{tag} @0.5  | Acc: {m['Accuracy@0.5']:.6f}  "
          f"Prec: {m['Precision@0.5']:.6f}  "
          f"Recall: {m['Recall@0.5']:.6f}  "
          f"F1: {m['F1@0.5']:.6f}")
    print(f"{tag} @BestJ({m['BestJ_Threshold']:.3f}) | Acc: {m['Accuracy@BestJ']:.6f}  "
          f"Prec: {m['Precision@BestJ']:.6f}  "
          f"Recall: {m['Recall@BestJ']:.6f}  "
          f"F1: {m['F1@BestJ']:.6f}")

print_line("LSTM", metrics)
print(f"\nSaved LSTM model, figures & tables to: {OUTDIR}")
