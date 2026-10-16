import pandas as pd
from config import (
    GEOSPATIAL_REQUIRED_COLS,
    CATEGORICAL_COLS,
    MIN_DISTANCE_MILES,
    MIN_DURATION_MINS,
    MAX_DURATION_MINS,
    MIN_PASSENGERS,
    MIN_FARE_USD,
)


def clean_taxi_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the 4-stage data cleaning and preprocessing pipeline:
    1. Deduplication
    2. Missing value handling (listwise deletion for geospatial, rule-based for payment)
    3. Type casting & temporal feature engineering
    4. Physics/domain anomaly removal
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    df = df_raw.copy()
    initial_rows = len(df)

    # STAGE 1: Remove exact duplicate records
    df = df.drop_duplicates()
    dups_removed = initial_rows - len(df)
    print(f"[Stage 1] Duplicates removed: {dups_removed}")

    # STAGE 2: Handle Missing Values
    # 2a. Drop rows missing critical origin/destination zones or boroughs (<1% of data)
    before_drop = len(df)
    df = df.dropna(subset=GEOSPATIAL_REQUIRED_COLS).copy()
    print(f"[Stage 2a] Rows dropped due to missing geospatial endpoints: {before_drop - len(df)}")

    # 2b. Domain-informed imputation for 'payment'
    # Credit card terminals log tips automatically; cash tips log as 0.
    cc_mask = df["payment"].isnull() & (df["tip"] > 0)
    df.loc[cc_mask, "payment"] = "credit card"
    mode_payment = df["payment"].mode()[0]
    df["payment"] = df["payment"].fillna(mode_payment)
    print(f"[Stage 2b] Missing 'payment' values imputed using tip rules and mode ('{mode_payment}').")

    # STAGE 3: Correct Data Types & Engineer Temporal Features
    df["pickup"] = pd.to_datetime(df["pickup"])
    df["dropoff"] = pd.to_datetime(df["dropoff"])

    df["trip_duration_mins"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60.0
    df["pickup_hour"] = df["pickup"].dt.hour
    df["day_of_week"] = df["pickup"].dt.day_name()

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    print("[Stage 3] Converted timestamps and engineered 'trip_duration_mins', 'pickup_hour', 'day_of_week'.")

    # STAGE 4: Filter Invalid Domain Anomalies & Outliers
    valid_mask = (
        (df["distance"] > MIN_DISTANCE_MILES)
        & (df["trip_duration_mins"] > MIN_DURATION_MINS)
        & (df["trip_duration_mins"] <= MAX_DURATION_MINS)
        & (df["passengers"] >= MIN_PASSENGERS)
        & (df["fare"] >= MIN_FARE_USD)
    )
    before_anomaly = len(df)
    df_clean = df[valid_mask].copy()
    print(f"[Stage 4] Anomalous/invalid records filtered: {before_anomaly - len(df_clean)}")

    total_removed = initial_rows - len(df_clean)
    print("-" * 60)
    print(f"Cleaned Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"Total rows removed: {total_removed} ({total_removed / initial_rows:.2%})")
    print(f"Remaining Missing Values: {df_clean.isnull().sum().sum()}")

    return df_clean