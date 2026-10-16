import os

# 1. Public Dataset URL
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"

# 2. Output Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_taxis.csv")
SUMMARY_STATS_PATH = os.path.join(OUTPUT_DIR, "summary_statistics.csv")

# 3. Plot Styling Constants
THEME_STYLE = "whitegrid"
THEME_PALETTE = "muted"
FIG_SIZE = (10, 6)
FIG_DPI = 120

# 4. Data Cleaning & Anomaly Thresholds
MIN_DISTANCE_MILES = 0.1
MIN_DURATION_MINS = 0.5
MAX_DURATION_MINS = 180.0
MIN_PASSENGERS = 1
MIN_FARE_USD = 0.01

# 5. Column Groupings
GEOSPATIAL_REQUIRED_COLS = [
    "pickup_zone",
    "dropoff_zone",
    "pickup_borough",
    "dropoff_borough",
]
CATEGORICAL_COLS = ["color", "payment", "pickup_borough", "dropoff_borough"]
NUMERICAL_COLS = [
    "passengers",
    "distance",
    "trip_duration_mins",
    "fare",
    "tip",
    "tolls",
    "total",
]


def ensure_directories():
    """Creates output directories if they do not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)