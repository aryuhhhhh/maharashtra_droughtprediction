"""
data_merging.py
----------------
Merges rainfall, soil moisture, ET, NDVI, and temperature datasets (2015–2024)
into a single master dataset for drought prediction.
"""

import pandas as pd
from functools import reduce
from pathlib import Path
import logging

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------
# Utility functions
# -----------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and log its shape."""
    df = pd.read_csv(path)
    logging.info(f"Loaded {path.name} → shape: {df.shape}")
    return df


def clean_district(df: pd.DataFrame) -> pd.DataFrame:
    """Clean district names (strip + title case)."""
    df["district"] = df["district"].astype(str).str.strip().str.title()
    return df


def preprocess_et(df: pd.DataFrame) -> pd.DataFrame:
    """Handle start_date → year, month in ET 2020–2024 file."""
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["year"] = df["start_date"].dt.year
        df["month"] = df["start_date"].dt.month
        df.drop(columns=["start_date"], inplace=True)
    return df


def merge_features(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple feature DataFrames on district, year, month."""
    return reduce(lambda left, right: pd.merge(left, right, on=["district", "year", "month"], how="outer"), dfs)


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate missing numeric values per district."""
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df.groupby("district")[col].transform(lambda g: g.interpolate(limit_direction="both"))
    return df


# -----------------------------------------------------
# Main pipeline
# -----------------------------------------------------
def merge_all_data():
    base_path = Path("data/raw")
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load files
    rain_15_19 = load_csv(base_path / "maharashtra_monthly_rainfall_2015_2019.csv")
    soil_15_19 = load_csv(base_path / "maharashtra_soil_moisture_era5_2015_2019.csv")
    et_15_19 = load_csv(base_path / "maharashtra_et_2015_2019.csv")
    ndvi_15_19 = load_csv(base_path / "maharashtra_monthly_ndvi_2015_2019.csv")
    temp_15_19 = load_csv(base_path / "maharashtra_monthly_temperature_2015_2019.csv")

    rain_20_24 = load_csv(base_path / "maharashtra_monthly_rainfall_2020_2024.csv")
    soil_20_24 = load_csv(base_path / "maharashtra_soil_moisture_era5_2020_2024.csv")
    et_20_24 = load_csv(base_path / "maharashtra_et_2020_2024.csv")
    ndvi_20_24 = load_csv(base_path / "maharashtra_monthly_ndvi_2020_2024.csv")
    temp_20_24 = load_csv(base_path / "maharashtra_monthly_temperature_2020_2024.csv")

    # Preprocess ET and district names
    et_20_24 = preprocess_et(et_20_24)
    datasets = [
        rain_15_19, soil_15_19, et_15_19, ndvi_15_19, temp_15_19,
        rain_20_24, soil_20_24, et_20_24, ndvi_20_24, temp_20_24,
    ]
    datasets = [clean_district(df) for df in datasets]
    (
        rain_15_19, soil_15_19, et_15_19, ndvi_15_19, temp_15_19,
        rain_20_24, soil_20_24, et_20_24, ndvi_20_24, temp_20_24
    ) = datasets

    # Merge two periods
    merged_15_19 = merge_features([rain_15_19, soil_15_19, et_15_19, ndvi_15_19, temp_15_19])
    merged_20_24 = merge_features([rain_20_24, soil_20_24, et_20_24, ndvi_20_24, temp_20_24])

    final_df = pd.concat([merged_15_19, merged_20_24], ignore_index=True)
    final_df = final_df.sort_values(by=["district", "year", "month"]).reset_index(drop=True)
    final_df = interpolate_missing(final_df)

    output_file = output_path / "maharashtra_climate_master.csv"
    final_df.to_csv(output_file, index=False)
    logging.info(f"✅ Saved merged dataset: {output_file} | shape: {final_df.shape}")

    return final_df


if __name__ == "__main__":
    merge_all_data()
