from config import ensure_directories, CLEANED_DATA_PATH
from data_acquisition import fetch_raw_data, inspect_raw_data
from data_cleaning import clean_taxi_data
from eda_analysis import generate_summary_statistics
from visualizations import generate_all_visualizations


def main():
    # Ensure output folders exist
    ensure_directories()

    # Step 1: Acquire and audit raw data
    df_raw = fetch_raw_data()
    inspect_raw_data(df_raw)

    # Step 2: Clean and preprocess data
    df_clean = clean_taxi_data(df_raw)
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[INFO] Cleaned dataset saved to: {CLEANED_DATA_PATH}")

    # Step 3: Compute summary statistics
    generate_summary_statistics(df_clean)

    # Step 4: Generate and save all 4 visualizations
    generate_all_visualizations(df_raw, df_clean)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()