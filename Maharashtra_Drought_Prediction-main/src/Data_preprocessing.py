"""
data_preprocessing.py
----------------------
Feature engineering for drought prediction:
- SPI, SPEI
- NDVI anomalies
- Soil moisture percentiles
- Rolling features
- Lag features
- Drought classification & forecast target
"""

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm, percentileofscore
from pathlib import Path
import logging

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def compute_spi(series, scale=3):
    """Compute Standardized Precipitation Index (SPI)."""
    series_roll = series.rolling(scale, min_periods=scale).sum() + 0.01
    valid = series_roll[series_roll > 0].dropna()
    if len(valid) < scale:
        return pd.Series(np.nan, index=series.index)
    params = gamma.fit(valid, floc=0)
    cdf = gamma.cdf(series_roll, *params)
    return pd.Series(norm.ppf(cdf), index=series.index)


def compute_spei(precip, pet, scale=3):
    """Compute Standardized Precipitation-Evapotranspiration Index (SPEI)."""
    balance = precip - pet
    series_roll = balance.rolling(scale, min_periods=scale).sum() + 0.01
    valid = series_roll[series_roll > 0].dropna()
    if len(valid) < scale:
        return pd.Series(np.nan, index=precip.index)
    params = gamma.fit(valid, floc=0)
    cdf = gamma.cdf(series_roll, *params)
    return pd.Series(norm.ppf(cdf), index=precip.index)


def classify_drought(spei_value):
    """Classify drought severity based on SPEI."""
    if spei_value <= -1.5:
        return "Extreme"
    elif spei_value <= -1.0:
        return "Moderate"
    else:
        return "No Drought"


# -----------------------------------------------------
# Main preprocessing function
# -----------------------------------------------------
def feature_engineering(
    input_file: Path = Path("data/processed/maharashtra_climate_master.csv"),
    output_file: Path = Path("data/processed/maharashtra_features_final.csv"),
    forecast_horizon: int = 3
):
    logging.info(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    # --- datetime & sorting ---
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values(["district", "date"])

    # --- SPI & SPEI ---
    logging.info("Computing SPI and SPEI indices...")
    df["spi_3"] = df.groupby("district")["rainfall_mm"].transform(lambda x: compute_spi(x, 3))
    df["spi_6"] = df.groupby("district")["rainfall_mm"].transform(lambda x: compute_spi(x, 6))

    df["spei_3"] = df.groupby("district").apply(
        lambda g: compute_spei(g["rainfall_mm"], g["et_mean_mm"], 3)
    ).reset_index(level=0, drop=True)
    df["spei_6"] = df.groupby("district").apply(
        lambda g: compute_spei(g["rainfall_mm"], g["et_mean_mm"], 6)
    ).reset_index(level=0, drop=True)

    # --- NDVI anomaly ---
    logging.info("Computing NDVI anomaly...")
    df["ndvi_anomaly"] = df.groupby("district")["ndvi"].transform(lambda x: (x - x.mean()) / x.std())

    # --- Soil moisture percentiles ---
    logging.info("Computing soil moisture percentiles...")
    soil_cols = ["SoilMoi_0_10", "SoilMoi_10_40", "SoilMoi_40_100", "SoilMoi_100_200"]
    for col in soil_cols:
        df[f"{col}_pct"] = df.groupby("district")[col].transform(
            lambda x: x.apply(lambda v: percentileofscore(x.dropna(), v))
        )

    # --- Rolling means (3-month) ---
    logging.info("Computing 3-month rolling means...")
    roll_cols = ["ndvi"] + soil_cols
    for col in roll_cols:
        df[f"{col}_roll3"] = df.groupby("district")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # --- Drought classification ---
    logging.info("Classifying drought severity...")
    df["drought_class"] = df["spei_3"].apply(classify_drought)

    # --- Forecast target (shifted drought class) ---
    logging.info(f"Shifting drought class by {forecast_horizon} months for forecasting...")
    df["future_drought_class"] = df.groupby("district")["drought_class"].shift(-forecast_horizon)

    # --- Lag features ---
    logging.info("Creating lag features...")
    lag_cols = {
        "rainfall_mm": 3,
        "ndvi": 3,
        "SoilMoi_0_10": 3,
        "et_mean_mm": 3
    }
    for col, max_lag in lag_cols.items():
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag{lag}"] = df.groupby("district")[col].shift(lag)

    # --- Additional rolling features ---
    logging.info("Computing rainfall rolling means (3 & 6 months)...")
    df["rainfall_roll3"] = df.groupby("district")["rainfall_mm"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["rainfall_roll6"] = df.groupby("district")["rainfall_mm"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)

    # --- Drop rows with missing values caused by lags/shift ---
    df_final = df.dropna().reset_index(drop=True)

    # --- Save final dataset ---
    df_final.to_csv(output_file, index=False)
    logging.info(f"✅ Feature-engineered dataset saved: {output_file} | shape: {df_final.shape}")

    return df_final


if __name__ == "__main__":
    feature_engineering()
