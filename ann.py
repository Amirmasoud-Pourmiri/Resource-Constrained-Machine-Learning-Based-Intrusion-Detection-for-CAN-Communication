# #!/usr/bin/env python
# # coding: utf-8

# # Import necessary libraries
# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import seaborn as sns
# sns.set(color_codes=True)
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Input, Dense, Flatten
# from tensorflow.keras.models import Model
# from numpy.random import seed
# import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from tensorflow.compat.v1 import set_random_seed


# # Set random seeds for reproducibility
# seed(1)
# set_random_seed(2)

# # Define the ANN binary classification model
# def binary_classification_model(X):
#     inputs = Input(shape=(X.shape[1], X.shape[2]))
#     x = Flatten()(inputs)
#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dense(16, activation='relu')(x)
#     outputs = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=inputs, outputs=outputs)  # Define the model
#     return model

# # Convert hex to decimal where applicable
# def hex_to_decimal(val):
#     try:
#         if isinstance(val, str) and val.startswith('0x'):
#             return int(val, 16)
#         return float(val)
#     except ValueError:
#         return np.nan

# # Function to create sequences
# def create_sequences(data, sequence_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - sequence_length):
#         seq = data.iloc[i:i + sequence_length].values
#         label = data.iloc[i + sequence_length]['Label']
#         sequences.append(seq)
#         labels.append(label)
#     return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

# # Load and preprocess data
# CSV_FILE = "splite/train.csv"

# # Read the dataset
# data = pd.read_csv(CSV_FILE)

# # Extract the relevant columns and drop the label
# labels = data['Label'].values
# features = data.drop(columns=['Label'])  # Drop 'Label'

# # Convert hex to decimal
# features = features.applymap(hex_to_decimal)

# # Check for non-numeric values and handle them
# features = features.apply(pd.to_numeric, errors='coerce')

# # Fill NaN values with 0
# features = features.fillna(0)

# # Convert all data to float32
# features = features.astype(np.float32)

# # Scale the data
# scaler = MinMaxScaler()
# features_scaled = scaler.fit_transform(features)

# # Combine scaled data and labels into a DataFrame
# features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
# features_scaled['Label'] = labels

# # Prepare training and test data
# sequence_length = 10  # Number of timesteps per sequence
# X, y = create_sequences(features_scaled, sequence_length)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# datasets = [
#     ("X_train", X_train),
#     ("X_test", X_test),
#     ("y_train", y_train),
#     ("y_test", y_test)
# ]

# # Loop through each dataset
# print("write dataset")
# for name, data in datasets:
#     # Check if the data is 3D
#     if len(data.shape) == 3:
#         # Reshape 3D data to 2D
#         data = data.reshape(data.shape[0], -1)
    
#     # Convert to DataFrame and save to CSV
#     pd.DataFrame(data).to_csv(f"ANN_{name}.csv", index=False, header=False)
# # Build and compile the ANN model
# model = binary_classification_model(X_train)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# # Train the ANN model
# nb_epochs = 1
# batch_size = 512
# print("Training started...")
# history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05, verbose=1).history
# print("Training completed.")

# # Predict and evaluate the ANN model
# y_pred = model.predict(X_test)
# y_pred_class = (y_pred > 0.5).astype(int)

# print("Classification Report:")
# print(classification_report(y_test, y_pred_class, zero_division=1))
# print("Accuracy Score:", accuracy_score(y_test, y_pred_class))

# # Save the ANN model
# model.save("binary_classification_model_ann.keras")
# print("Model saved")

# # Plot training history
# plt.figure(figsize=(16, 9))
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("ann_training_history_plot.png")
# plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANN (1D windowed) with CNN-like outputs:
- parsing, split, scaling-after-split, sequences
- training curves, ROC/PR/Calibration
- confusion matrices (0.5 & BestJ)
- metrics_summary.csv, classification_report_thr0.5.csv, test_predictions.csv
- ANN_X_train.csv, ANN_X_test.csv, ANN_y_train.csv, ANN_y_test.csv
- scaling_vectors_ann.hpp
- trained model: binary_classification_model_ann.keras (under paper_outputs_ann/)
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
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from io import StringIO

# ---------------- Reproducibility ----------------
np.random.seed(1)
tf.random.set_seed(2)

# ---------------- Constants / paths ----------------
SEQ_LEN  = 10
CSV_FILE = "splite/train.csv"
OUTDIR   = "paper_outputs_ann"
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

# Hex for ID and all D*; the rest decimal (Timestamp always decimal)
hex_cols = [c for c in features.columns if c == "ID" or c.upper().startswith("D")]
dec_cols = [c for c in features.columns if c not in hex_cols]

if "Timestamp" in features.columns:
    features["Timestamp"] = parse_dec(features["Timestamp"])

if "DLC" in features.columns:
    features["DLC"] = features["DLC"].apply(parse_hex if looks_hex_series(features["DLC"]) else parse_dec)
    if features["DLC"].max() > 15:
        raise ValueError("DLC parsed > 15 → check base/format.")

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

# ---------------- Emit C++ scaling vectors ----------------
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

with open(os.path.join(OUTDIR, "scaling_vectors_ann.hpp"), "w", encoding="utf-8") as f:
    f.write(hdr.getvalue())

# ---------------- Save flat CSVs for C++ pipeline ----------------
pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).to_csv(os.path.join(OUTDIR, "ANN_X_train.csv"), index=False, header=False)
pd.DataFrame(X_test.reshape(X_test.shape[0], -1)).to_csv(os.path.join(OUTDIR, "ANN_X_test.csv"), index=False, header=False)
pd.DataFrame(y_train).to_csv(os.path.join(OUTDIR, "ANN_y_train.csv"), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(OUTDIR, "ANN_y_test.csv"), index=False, header=False)

# ---------------- ANN model ----------------
def build_ann(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_ann((SEQ_LEN, num_features))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=200, batch_size=512, validation_split=0.1, verbose=1)

# ---------------- Save model ----------------
model.save(os.path.join(OUTDIR, "binary_classification_model_ann.keras"))

# ---------------- Plots & evaluation (CNN-like) ----------------
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
    "Accuracy vs Epoch (ANN)",
    "Epoch", "Accuracy",
    os.path.join(OUTDIR, "accuracy.png")
)
_save_lineplot(
    epochs,
    [history.history["loss"], history.history["val_loss"]],
    ["Train", "Validation"],
    "Loss vs Epoch (ANN)",
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
_plot_confusion(cm, ["0","1"], "Confusion Matrix (thr=0.5, ANN)",
                os.path.join(OUTDIR, "confusion_matrix_counts.png"))
_plot_confusion(cm, ["0","1"], "Confusion Matrix (Normalized, thr=0.5, ANN)",
                os.path.join(OUTDIR, "confusion_matrix_normalized.png"), normalize=True)

# ROC / PR / Calibration
fpr, tpr, roc_thr = roc_curve(y_true, y_proba); rocAUC = auc(fpr, tpr)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {rocAUC:.3f}"); plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (ANN)"); plt.legend(); plt.grid(True, linestyle=":")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "roc_curve.png"), dpi=300); plt.close()

prec, rec, pr_thr = precision_recall_curve(y_true, y_proba); AP = average_precision_score(y_true, y_proba)
plt.figure(); plt.plot(rec, prec, label=f"AP = {AP:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve (ANN)")
plt.legend(); plt.grid(True, linestyle=":"); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pr_curve.png"), dpi=300); plt.close()

prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
plt.figure(); plt.plot([0,1],[0,1],"--"); plt.plot(prob_pred, prob_true, marker="o")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration Curve (ANN)"); plt.grid(True, linestyle=":")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "calibration_curve.png"), dpi=300); plt.close()

# Best Youden's J
J = tpr - fpr
best_j_ix = int(np.argmax(J))
best_thr = float(roc_thr[best_j_ix])
y_pred_best = (y_proba >= best_thr).astype(int)
cm_best = confusion_matrix(y_true, y_pred_best, labels=[0,1])
_plot_confusion(cm_best, ["0","1"], f"Confusion Matrix (thr={best_thr:.3f}, ANN)",
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

print_line("ANN", metrics)
print(f"\nSaved ANN model, figures & tables to: {OUTDIR}")
