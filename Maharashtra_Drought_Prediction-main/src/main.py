"""
main.py
--------
Final entry point for the Drought Prediction Project (LSTM Model).

Workflow:
1. Load processed dataset
2. Load trained LSTM model + encoders
3. Generate time-series predictions
4. Evaluate and save results
5. Visualize key outputs
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/processed/maharashtra_model_ready.csv"
LSTM_MODEL_PATH = "models/LSTM/lstm_model_tf.keras"
DISTRICT_ENCODER_PATH = "models/RandomForest/label_encoder.pkl"   # reuse from RF
RESULTS_DIR = "Results_visualize"
OUTPUT_PATH = "model_predictions_lstm.csv"
SEQ_LENGTH = 12

# -----------------------------
# Step 1: Prepare Environment
# -----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"✅ Results directory ready: {RESULTS_DIR}")

# -----------------------------
# Step 2: Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["district", "date"]).reset_index(drop=True)
print("✅ Data loaded successfully:", df.shape)

# -----------------------------
# Step 3: Load Model and Encoders
# -----------------------------
lstm_model = load_model(LSTM_MODEL_PATH)
import joblib
district_encoder = joblib.load(DISTRICT_ENCODER_PATH)
print("✅ LSTM model and encoder loaded.")

# -----------------------------
# Step 4: Prepare LSTM Sequences
# -----------------------------
df["district_encoded"] = district_encoder.transform(df["district"])

lstm_features = [
    "rainfall_mm", "SoilMoi_0_10", "SoilMoi_10_40",
    "et_mean_mm", "ndvi", "mean_temp_c", "month_sin", "month_cos",
    "district_encoded"
]

X_sequences, X_district, y_sequences, seq_dates = [], [], [], []

for district_id in sorted(df["district_encoded"].unique()):
    district_data = df[df["district_encoded"] == district_id].copy()
    for i in range(SEQ_LENGTH, len(district_data)):
        X_sequences.append(district_data[lstm_features].iloc[i-SEQ_LENGTH:i].values)
        X_district.append(district_id)
        y_sequences.append(district_data["target_1m_ahead"].iloc[i])
        seq_dates.append(district_data["date"].iloc[i])

X_sequences = np.array(X_sequences)
X_district = np.array(X_district)
y_sequences = np.array(y_sequences)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_sequences)
print("✅ Prepared LSTM input sequences:", X_sequences.shape)

# -----------------------------
# Step 5: Generate Predictions
# -----------------------------
y_pred = np.argmax(
    lstm_model.predict([X_sequences, X_district], verbose=0),
    axis=1
)
print("✅ Predictions generated successfully.")

# -----------------------------
# Step 6: Evaluation
# -----------------------------
acc = accuracy_score(y_encoded, y_pred)
print(f"\n📊 LSTM Accuracy: {acc:.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/lstm_confusion_matrix.png", dpi=300)
plt.close()
print("✅ Confusion matrix saved.")

# -----------------------------
# Step 7: Save Predictions
# -----------------------------
results_df = pd.DataFrame({
    "date": seq_dates,
    "district_encoded": X_district,
    "true_label": y_sequences,
    "predicted_label": label_encoder.inverse_transform(y_pred)
})
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved: {OUTPUT_PATH}")

# -----------------------------
# Step 8: Visualize Drought Trends
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=results_df, x="predicted_label", palette="Set2")
plt.title("Predicted Drought Class Distribution (LSTM)")
plt.xlabel("Drought Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/lstm_drought_class_distribution.png", dpi=300)
plt.close()
print("✅ Drought class distribution saved.")

print("\n🎯 LSTM Drought Prediction Pipeline Completed Successfully!")
