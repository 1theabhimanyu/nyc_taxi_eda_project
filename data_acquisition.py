import pandas as pd
from config import DATA_URL


def fetch_raw_data(url: str = DATA_URL) -> pd.DataFrame:
    """Downloads the raw NYC Taxi dataset from the public URL."""
    print(f"[INFO] Fetching dataset from: {url}")
    df_raw = pd.read_csv(url)
    return df_raw


def inspect_raw_data(df_raw: pd.DataFrame) -> None:
    """Prints initial shape, data types, and missing value counts."""
    print("\n" + "=" * 60)
    print("STEP 1: INITIAL DATA AUDIT")
    print("=" * 60)
    print(f"Initial Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\n")

    print("--- Data Types & Non-Null Counts ---")
    df_raw.info()

    print("\n--- Missing Values Count ---")
    missing_summary = df_raw.isnull().sum()
    missing_filtered = missing_summary[missing_summary > 0]
    if not missing_filtered.empty:
        for col, count in missing_filtered.items():
            pct = (count / len(df_raw)) * 100
            print(f"  - {col}: {count} missing ({pct:.2f}%)")
    else:
        print("  No missing values found.")