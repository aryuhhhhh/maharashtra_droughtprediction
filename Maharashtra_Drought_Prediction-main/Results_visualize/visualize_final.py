"""
visualize_final.py
------------------
Generates visual analytics for model comparison and drought trends:
- Accuracy, Precision, Recall, F1 comparison (Overall vs Time-series)
- Confusion matrices (LSTM, RF, XGB) on Time-series
- Drought class distributions on Time-series
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
TS_CSV = "model_predictions_comparison_timeseries.csv"
RESULTS_DIR = "Results_visualize"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Step 1: Hard-coded Overall metrics (from previous evaluation)
# Replace these with actual values from your other file
# -----------------------------
overall_metrics = {
    "Model": ["LSTM", "RF", "XGB"],
    "Accuracy": [0.92, 0.85, 0.82],
    "Precision": [0.91, 0.84, 0.82],
    "Recall": [0.92, 0.85, 0.82],
    "F1-score": [0.91, 0.84, 0.82]
}

df_overall = pd.DataFrame(overall_metrics)
df_overall["Dataset"] = "Overall"

# -----------------------------
# Step 2: Load Time-series predictions
# -----------------------------
df_ts = pd.read_csv(TS_CSV)
df_ts['true_label'] = df_ts['true_label'].astype(str)
df_ts['lstm_pred'] = df_ts['lstm_pred'].astype(str)
df_ts['rf_pred']   = df_ts['rf_pred'].astype(str)
df_ts['xgb_pred']  = df_ts['xgb_pred'].astype(str)

# -----------------------------
# Step 3: Compute metrics for Time-series
# -----------------------------
def compute_metrics(df):
    true = df['true_label']
    lstm = df['lstm_pred']
    rf   = df['rf_pred']
    xgb  = df['xgb_pred']

    lstm_r = classification_report(true, lstm, output_dict=True, zero_division=0)
    rf_r   = classification_report(true, rf, output_dict=True, zero_division=0)
    xgb_r  = classification_report(true, xgb, output_dict=True, zero_division=0)

    return {
        "Model": ["LSTM", "RF", "XGB"],
        "Accuracy": [lstm_r["accuracy"], rf_r["accuracy"], xgb_r["accuracy"]],
        "Precision": [
            lstm_r["weighted avg"]["precision"],
            rf_r["weighted avg"]["precision"],
            xgb_r["weighted avg"]["precision"]
        ],
        "Recall": [
            lstm_r["weighted avg"]["recall"],
            rf_r["weighted avg"]["recall"],
            xgb_r["weighted avg"]["recall"]
        ],
        "F1-score": [
            lstm_r["weighted avg"]["f1-score"],
            rf_r["weighted avg"]["f1-score"],
            xgb_r["weighted avg"]["f1-score"]
        ]
    }

ts_metrics = compute_metrics(df_ts)
df_ts_metrics = pd.DataFrame(ts_metrics)
df_ts_metrics["Dataset"] = "Time-series"

# -----------------------------
# Step 4: Combine metrics for plotting
# -----------------------------
metrics_df = pd.concat([df_overall, df_ts_metrics], ignore_index=True)
metrics_melted = metrics_df.melt(id_vars=["Model","Dataset"], var_name="Metric", value_name="Score")

# -----------------------------
# Step 5: Plot Model Performance Comparison
# -----------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="viridis", ci=None)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Model Performance: Overall vs Time-series")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_performance_comparison_dual.png", dpi=300)
plt.close()
print("📊 Saved: model_performance_comparison_dual.png")

# -----------------------------
# Step 6: Confusion Matrices (Time-series dataset)
# -----------------------------
models = {
    "LSTM": (df_ts['true_label'], df_ts['lstm_pred']),
    "RandomForest": (df_ts['true_label'], df_ts['rf_pred']),
    "XGBoost": (df_ts['true_label'], df_ts['xgb_pred'])
}

classes = sorted(df_ts['true_label'].unique())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, (true, pred)) in zip(axes, models.items()):
    cm = confusion_matrix(true, pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrices_ts.png", dpi=300)
plt.close()
print("📈 Saved: confusion_matrices_ts.png")

# -----------------------------
# Step 7: Drought Class Distribution (Time-series dataset)
# -----------------------------
melted = df_ts.melt(id_vars=["date", "district_encoded"],
                    value_vars=["true_label", "lstm_pred", "rf_pred", "xgb_pred"],
                    var_name="Source", value_name="DroughtClass")

plt.figure(figsize=(8,5))
sns.countplot(data=melted, x="DroughtClass", hue="Source", palette="Set2")
plt.title("Drought Class Distribution Across Models (Time-series)")
plt.xlabel("Drought Class")
plt.ylabel("Count")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/drought_class_distribution_ts.png")
plt.close()
print("🧮 Saved: drought_class_distribution_ts.png")

# -----------------------------
# Step 8: Summary
# -----------------------------
print("\n✅ All visualizations completed successfully!")
print(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
