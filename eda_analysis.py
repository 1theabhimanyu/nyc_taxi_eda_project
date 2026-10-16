import pandas as pd
from config import NUMERICAL_COLS, SUMMARY_STATS_PATH


def generate_summary_statistics(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics, IQR, and skewness for numerical columns."""
    print("\n" + "=" * 60)
    print("STEP 3: DESCRIPTIVE SUMMARY STATISTICS")
    print("=" * 60)

    summary_stats = df_clean[NUMERICAL_COLS].describe().T
    summary_stats["skewness"] = df_clean[NUMERICAL_COLS].skew()
    summary_stats["IQR"] = summary_stats["75%"] - summary_stats["25%"]

    selected_cols = ["count", "mean", "50%", "std", "IQR", "min", "max", "skewness"]
    summary_table = summary_stats[selected_cols].round(2)

    print(summary_table.to_string())
    summary_table.to_csv(SUMMARY_STATS_PATH)
    print(f"\n[INFO] Summary statistics saved to: {SUMMARY_STATS_PATH}")

    return summary_table