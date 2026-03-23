import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Config
# -----------------------------
SEQ_LENGTH = 12
LSTM_MODEL_PATH = "models/LSTM/lstm_model_tf.keras"
RF_MODEL_PATH = "models/RandomForest/rf_smote_best.pkl"
RF_ENCODER_PATH = "models/RandomForest/label_encoder.pkl"
RF_FEATURES_PATH = "models/RandomForest/rf_features_order.pkl"
XGB_MODEL_PATH = "models/XGBoost/xgb_model_1m.pkl"
XGB_FEATURES_PATH = "models/XGBoost/XGB_feature_order.pkl"
DATA_PATH = "data/processed/maharashtra_model_ready.csv"
OUTPUT_PATH = "model_predictions_comparison_timeseries.csv"

# -----------------------------
# Step 1: Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)
print("✅ Data loaded and sorted by date. Shape:", df.shape)

# -----------------------------
# Step 2: Time Series Split
# -----------------------------
train_ratio = 0.8
train_size = int(len(df) * train_ratio)

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# -----------------------------
# Step 3: Load Models
# -----------------------------
lstm_model = load_model(LSTM_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
district_encoder = joblib.load(RF_ENCODER_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

# -----------------------------
# Step 4: Prepare LSTM sequences for TEST SET
# -----------------------------
lstm_features = [
    'rainfall_mm', 'SoilMoi_0_10', 'SoilMoi_10_40', 'et_mean_mm', 'ndvi',
    'mean_temp_c', 'month_sin', 'month_cos', 'district_encoded'
]

test_df['district_encoded'] = district_encoder.transform(test_df['district'])

X_sequences, X_district, y_sequences, sequence_dates = [], [], [], []

for district_id in sorted(test_df['district_encoded'].unique()):
    district_data = test_df[test_df['district_encoded'] == district_id].copy()
    for i in range(SEQ_LENGTH, len(district_data)):
        X_sequences.append(district_data[lstm_features].iloc[i-SEQ_LENGTH:i].values)
        X_district.append(district_id)
        y_sequences.append(district_data['target_1m_ahead'].iloc[i])
        sequence_dates.append(district_data['date'].iloc[i])

X_sequences = np.array(X_sequences)
X_district_arr = np.array(X_district[-len(X_sequences):])
y_sequences = np.array(y_sequences)

target_encoder = LabelEncoder()
y_sequences_encoded = target_encoder.fit_transform(y_sequences)

lstm_preds = np.argmax(
    lstm_model.predict([X_sequences, X_district_arr], verbose=0),
    axis=1
)
print("✅ LSTM predictions on test set complete.")

# -----------------------------
# Step 5: RF Predictions
# -----------------------------
rf_features_order = joblib.load(RF_FEATURES_PATH)

for col in rf_features_order:
    if col not in test_df.columns:
        test_df[col] = 0

X_rf = test_df[rf_features_order].copy()
y_rf = test_df['target_1m_ahead'].values

rf_preds = rf_model.predict(X_rf)
print("✅ RF predictions on test set complete.")

# -----------------------------
# Step 6: XGB Predictions
# -----------------------------
xgb_features_order = joblib.load(XGB_FEATURES_PATH)

for col in xgb_features_order:
    if col not in test_df.columns:
        test_df[col] = 0

X_xgb = test_df[xgb_features_order].copy()
le_xgb = LabelEncoder()
y_xgb_encoded = le_xgb.fit_transform(test_df['target_1m_ahead'].values)

xgb_preds = xgb_model.predict(X_xgb)
print("✅ XGB predictions on test set complete.")

# -----------------------------
# Step 7: Evaluation
# -----------------------------
print("\n--- Model Accuracy on TIME-SERIES TEST SET ---")
print("LSTM Accuracy:", accuracy_score(y_sequences_encoded, lstm_preds))
print("RF Accuracy:", accuracy_score(y_rf, rf_preds))
print("XGB Accuracy:", accuracy_score(y_xgb_encoded, xgb_preds))

print("\n--- RF Classification Report ---")
print(classification_report(y_rf, rf_preds))

print("\n--- XGB Classification Report ---")
print(classification_report(y_xgb_encoded, xgb_preds))

# -----------------------------
# Step 8: Save Comparison CSV
# -----------------------------
comparison_df = pd.DataFrame({
    "date": sequence_dates[-len(lstm_preds):],
    "district_encoded": X_district[-len(lstm_preds):],
    "true_label": y_sequences[-len(lstm_preds):],
    "lstm_pred": target_encoder.inverse_transform(lstm_preds),
    "rf_pred": rf_preds[-len(lstm_preds):],  # align lengths
    "xgb_pred": le_xgb.inverse_transform(xgb_preds[-len(lstm_preds):])
})

comparison_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Comparison for time-series test set saved to: {OUTPUT_PATH}")



