import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    PLOTS_DIR,
    THEME_STYLE,
    THEME_PALETTE,
    FIG_SIZE,
    FIG_DPI,
    NUMERICAL_COLS,
)


def configure_plot_style():
    """Sets global Matplotlib and Seaborn visual styles."""
    sns.set_theme(style=THEME_STYLE, palette=THEME_PALETTE)
    plt.rcParams["figure.figsize"] = FIG_SIZE
    plt.rcParams["figure.dpi"] = FIG_DPI


def plot_missing_and_distribution(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    """Visualization 1: Missing Value Percentages & Right-Skewed Trip Distance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left Plot: Missing Values in Raw Data
    missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
    missing_pct.plot(kind="barh", color="#d95f02", ax=axes[0], edgecolor="black")
    axes[0].set_title("1A. Percentage of Missing Values in Raw Dataset", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Missing Percentage (%)")
    axes[0].set_ylabel("Dataset Columns")

    # Right Plot: Distribution of Trip Distance (Histogram + KDE)
    sns.histplot(df_clean["distance"], bins=40, kde=True, color="#1b9e77", ax=axes[1])
    mean_dist = df_clean["distance"].mean()
    median_dist = df_clean["distance"].median()
    axes[1].axvline(mean_dist, color="red", linestyle="--", label=f"Mean: {mean_dist:.2f} mi")
    axes[1].axvline(median_dist, color="navy", linestyle="-", label=f"Median: {median_dist:.2f} mi")
    axes[1].set_title("1B. Distribution of Trip Distance (Right-Skewed)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Trip Distance (Miles)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "viz1_missing_and_distribution.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_distance_vs_tip(df_clean: pd.DataFrame) -> str:
    """Visualization 2: Bivariate Scatter Plot of Distance vs. Tip by Payment Method."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean,
        x="distance",
        y="tip",
        hue="payment",
        alpha=0.6,
        palette={"credit card": "#2b83ba", "cash": "#fdae61"},
    )
    plt.title("2. Trip Distance vs. Tip Amount Segmented by Payment Method", fontsize=13, fontweight="bold")
    plt.xlabel("Trip Distance (Miles)")
    plt.ylabel("Recorded Tip Amount ($ USD)")
    plt.legend(title="Payment Method")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "viz2_distance_vs_tip_scatter.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_borough_fare_boxplot(df_clean: pd.DataFrame) -> str:
    """Visualization 3: Box Plot of Total Fare Distribution Across Pickup Boroughs."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_clean,
        x="pickup_borough",
        y="total",
        palette="Set2",
        showfliers=True,
    )
    plt.title("3. Total Fare Distribution by Pickup Borough", fontsize=13, fontweight="bold")
    plt.xlabel("Pickup Borough")
    plt.ylabel("Total Charged Amount ($ USD)")
    plt.ylim(0, 100)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "viz3_borough_fare_boxplot.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_and_hourly_trend(df_clean: pd.DataFrame) -> str:
    """Visualization 4: Multivariate Correlation Matrix & Hourly Demand Line Plot."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left Plot: Correlation Heatmap
    corr_matrix = df_clean[NUMERICAL_COLS].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("4A. Feature Correlation Matrix", fontsize=12, fontweight="bold")

    # Right Plot: Hourly Trip Demand Trend
    hourly_counts = df_clean.groupby("pickup_hour").size()
    sns.lineplot(
        x=hourly_counts.index,
        y=hourly_counts.values,
        marker="o",
        linewidth=2.5,
        color="#7570b3",
        ax=axes[1],
    )
    axes[1].set_title("4B. Temporal Trend: Taxi Demand by Hour of Day", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (0 = Midnight, 23 = 11 PM)")
    axes[1].set_ylabel("Number of Trips")
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "viz4_correlation_and_hourly_trend.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def generate_all_visualizations(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Runs all visualization functions and logs output file paths."""
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING & SAVING VISUALIZATIONS")
    print("=" * 60)
    configure_plot_style()

    p1 = plot_missing_and_distribution(df_raw, df_clean)
    print(f"  [Saved] Visualization 1 -> {p1}")

    p2 = plot_distance_vs_tip(df_clean)
    print(f"  [Saved] Visualization 2 -> {p2}")

    p3 = plot_borough_fare_boxplot(df_clean)
    print(f"  [Saved] Visualization 3 -> {p3}")

    p4 = plot_correlation_and_hourly_trend(df_clean)
    print(f"  [Saved] Visualization 4 -> {p4}")