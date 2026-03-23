"""
data_feature_building.py
------------------------
Prepare final model-ready dataset for drought prediction.
- Removes SPI, SPEI, and drought_class columns
- Adds month cyclical encodings (sin/cos)
- Adds shifted drought targets (1m, 2m, 3m ahead)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------------------------------------
# Main feature building function
# -------------------------------------------------
def build_model_ready_dataset(
    input_file: Path = Path("data/processed/maharashtra_features_final.csv"),
    output_file: Path = Path("data/processed/maharashtra_model_ready.csv")
):
    logging.info(f"📂 Loading preprocessed dataset: {input_file}")
    df = pd.read_csv(input_file)

    # -------------------------------------------------
    # 1️⃣ Remove columns not needed for modeling
    # -------------------------------------------------
    drop_cols = [
        "spi_3", "spi_6", "spei_3", "spei_6",
        "drought_class"  # remove as well, we’ll keep future_drought_class as target base
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    logging.info("🧹 Removed SPI, SPEI, and drought_class columns.")

    # -------------------------------------------------
    # 2️⃣ Add month cyclic encoding
    # -------------------------------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    logging.info("🌀 Added month_sin and month_cos encoding.")

    # -------------------------------------------------
    # 3️⃣ Add shifted drought targets (1m, 2m, 3m ahead)
    # -------------------------------------------------
    if "future_drought_class" in df.columns:
        df["target_1m_ahead"] = df.groupby("district")["future_drought_class"].shift(-1)
        df["target_2m_ahead"] = df.groupby("district")["future_drought_class"].shift(-2)
        df["target_3m_ahead"] = df.groupby("district")["future_drought_class"].shift(-3)
        logging.info("⏩ Created 1m, 2m, and 3m ahead target columns.")
    else:
        logging.warning("⚠️ future_drought_class column not found — skipping target generation.")

    # -------------------------------------------------
    # 4️⃣ Drop rows with missing target values
    # -------------------------------------------------
    df_final = df.dropna(subset=["target_1m_ahead", "target_2m_ahead", "target_3m_ahead"])
    df_final = df_final.reset_index(drop=True)

    # -------------------------------------------------
    # 5️⃣ Save model-ready dataset
    # -------------------------------------------------
    df_final.to_csv(output_file, index=False)
    logging.info(f"✅ Saved final model-ready dataset: {output_file} | shape: {df_final.shape}")

    return df_final


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    build_model_ready_dataset()
