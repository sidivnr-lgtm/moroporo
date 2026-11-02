import pandas as pd
import numpy as np

# =========================================================
# PROJECT HEAT-SHIELD PH
# FORECAST BENCHMARK COMPARISON
# =========================================================

print("=" * 65)
print("PROJECT HEAT-SHIELD PH")
print("FORECAST BENCHMARK COMPARISON")
print("=" * 65)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("\nLoading Daily Heat Index Data.csv...")

try:
    df = pd.read_csv("Daily Heat Index Data.csv")
except FileNotFoundError:
    print("\nERROR: Daily Heat Index Data.csv was not found.")
    print("Make sure this Python file is in the same folder as the CSV.")
    input("\nPress ENTER to close...")
    raise SystemExit

# =========================================================
# 2. DEFINE STATIONS
# =========================================================

stations = [
    "Baguio City",
    "Basco Radar",
    "Catarman",
    "Davao City",
    "Dumaguete City",
    "General Santos City",
    "Laoag City",
    "Legazpi City",
    "Mactan Int Airport",
    "Malaybalay",
    "Puerto Princesa City",
    "San Jose",
    "Science Garden QC",
    "Surigao City",
    "Tacloban City",
    "Tuguegarao City",
    "Zamboanga City"
]

# =========================================================
# 3. STANDARDIZE COLUMN NAMES
# =========================================================

station_mapping = {
    0: "Date",
    1: "Baguio City",
    2: "Basco Radar",
    3: "Catarman",
    4: "Davao City",
    5: "Dumaguete City",
    6: "General Santos City",
    7: "Laoag City",
    8: "Legazpi City",
    9: "Mactan Int Airport",
    10: "Malaybalay",
    11: "Puerto Princesa City",
    12: "San Jose",
    13: "Science Garden QC",
    14: "Surigao City",
    15: "Tacloban City",
    16: "Tuguegarao City",
    17: "Zamboanga City"
}

if len(df.columns) == 18:
    df.columns = list(station_mapping.values())

# =========================================================
# 4. CONVERT DATE
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

df = df.dropna(subset=["Date"])

df["Year"] = df["Date"].dt.year

print("Dataset loaded successfully.")
print(
    "Years:",
    int(df["Year"].min()),
    "-",
    int(df["Year"].max())
)
print("Stations:", len(stations))

# =========================================================
# 5. CONVERT STATION DATA TO NUMERIC
# =========================================================

for station in stations:

    if station in df.columns:

        df[station] = pd.to_numeric(
            df[station],
            errors="coerce"
        )

# =========================================================
# 6. CALCULATE ANNUAL MAXIMUM HEAT INDEX
# =========================================================

annual_max = df.groupby("Year")[stations].max()

print("\nAnnual maximum Heat Index values calculated.")

# =========================================================
# 7. TRAIN / TEST DESIGN
#
# TRAINING: 2015–2023
# TEST:      2024
# =========================================================

TRAIN_START = 2015
TRAIN_END = 2023
TEST_YEAR = 2024

print("\nBenchmark design:")
print("Training period:", TRAIN_START, "-", TRAIN_END)
print("Independent test year:", TEST_YEAR)

# =========================================================
# 8. BENCHMARK CALCULATIONS
# =========================================================

results = []

print("\n")
print("=" * 65)
print("STATION-LEVEL BENCHMARK RESULTS")
print("=" * 65)

for station in stations:

    series = annual_max[station].dropna()

    train = series[
        (series.index >= TRAIN_START) &
        (series.index <= TRAIN_END)
    ]

    test = series[
        series.index == TEST_YEAR
    ]

    # -----------------------------------------------------
    # Check whether enough data exist
    # -----------------------------------------------------

    if len(train) < 3:
        print(
            f"{station}: insufficient training data"
        )
        continue

    if len(test) == 0:
        print(
            f"{station}: no 2024 observation"
        )
        continue

    # Actual 2024 maximum
    actual = float(test.iloc[0])

    # =====================================================
    # MODEL 1
    # NAÏVE PERSISTENCE BASELINE
    #
    # Prediction = previous year's observed maximum
    # =====================================================

    naive_prediction = float(train.iloc[-1])

    # =====================================================
    # MODEL 2
    # PROJECT HEAT-SHIELD PH TREND MODEL
    #
    # Linear trend fitted using 2015–2023.
    # A minimum trend rate of 0.12 °C/year is retained
    # to match the floor used in the existing HEAT-SHIELD
    # PH projection implementation.
    # =====================================================

    x = train.index.to_numpy(dtype=float)
    y = train.to_numpy(dtype=float)

    raw_slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    benchmark_rate = max(
        float(raw_slope),
        0.12
    )

    # HEAT-SHIELD PH prediction for 2024
    heatshield_prediction = (
        float(train.iloc[-1])
        + benchmark_rate
        * (TEST_YEAR - TRAIN_END)
    )

    # =====================================================
    # ERROR CALCULATIONS
    # =====================================================

    naive_error = actual - naive_prediction
    heatshield_error = actual - heatshield_prediction

    naive_abs_error = abs(naive_error)
    heatshield_abs_error = abs(heatshield_error)

    naive_squared_error = naive_error ** 2
    heatshield_squared_error = heatshield_error ** 2

    if actual != 0:

        naive_ape = (
            abs(naive_error / actual)
            * 100
        )

        heatshield_ape = (
            abs(heatshield_error / actual)
            * 100
        )

    else:

        naive_ape = np.nan
        heatshield_ape = np.nan

    # =====================================================
    # STORE RESULTS
    # =====================================================

    results.append({

        "Station": station,

        "Raw_Slope_C_per_Year":
            round(raw_slope, 6),

        "Benchmark_Rate_C_per_Year":
            round(benchmark_rate, 6),

        "Actual_2024_Max_HI_C":
            round(actual, 4),

        "Naive_2024_Prediction_C":
            round(naive_prediction, 4),

        "HeatShield_2024_Prediction_C":
            round(heatshield_prediction, 4),

        "Naive_Absolute_Error_C":
            round(naive_abs_error, 4),

        "HeatShield_Absolute_Error_C":
            round(heatshield_abs_error, 4),

        "Naive_Squared_Error":
            round(naive_squared_error, 6),

        "HeatShield_Squared_Error":
            round(heatshield_squared_error, 6),

        "Naive_MAPE_pct":
            round(naive_ape, 6),

        "HeatShield_MAPE_pct":
            round(heatshield_ape, 6)
    })

    print(
        f"{station:<25}"
        f"Actual={actual:>6.2f}   "
        f"Naive={naive_prediction:>6.2f}   "
        f"HEAT-SHIELD={heatshield_prediction:>6.2f}"
    )

# =========================================================
# 9. CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# 10. CALCULATE OVERALL RMSE
# =========================================================

naive_rmse = np.sqrt(
    results_df[
        "Naive_Squared_Error"
    ].mean()
)

heatshield_rmse = np.sqrt(
    results_df[
        "HeatShield_Squared_Error"
    ].mean()
)

# =========================================================
# 11. CALCULATE OVERALL MAE
# =========================================================

naive_mae = results_df[
    "Naive_Absolute_Error_C"
].mean()

heatshield_mae = results_df[
    "HeatShield_Absolute_Error_C"
].mean()

# =========================================================
# 12. CALCULATE OVERALL MAPE
# =========================================================

naive_mape = results_df[
    "Naive_MAPE_pct"
].mean()

heatshield_mape = results_df[
    "HeatShield_MAPE_pct"
].mean()

# =========================================================
# 13. CALCULATE PERCENTAGE IMPROVEMENT
# =========================================================

if naive_rmse != 0:

    rmse_improvement = (
        (naive_rmse - heatshield_rmse)
        / naive_rmse
    ) * 100

else:
    rmse_improvement = np.nan


if naive_mae != 0:

    mae_improvement = (
        (naive_mae - heatshield_mae)
        / naive_mae
    ) * 100

else:
    mae_improvement = np.nan


if naive_mape != 0:

    mape_improvement = (
        (naive_mape - heatshield_mape)
        / naive_mape
    ) * 100

else:
    mape_improvement = np.nan

# =========================================================
# 14. COUNT STATION-LEVEL WINS
# =========================================================

heatshield_rmse_wins = (
    results_df["HeatShield_Squared_Error"]
    <
    results_df["Naive_Squared_Error"]
).sum()

heatshield_mae_wins = (
    results_df["HeatShield_Absolute_Error_C"]
    <
    results_df["Naive_Absolute_Error_C"]
).sum()

heatshield_mape_wins = (
    results_df["HeatShield_MAPE_pct"]
    <
    results_df["Naive_MAPE_pct"]
).sum()

total_stations = len(results_df)

# =========================================================
# 15. PRINT OVERALL RESULTS
# =========================================================

print("\n")
print("=" * 65)
print("OVERALL BENCHMARK RESULTS")
print("=" * 65)

print("\nNAÏVE PERSISTENCE BASELINE")
print("-" * 40)
print(f"RMSE : {naive_rmse:.4f} °C")
print(f"MAE  : {naive_mae:.4f} °C")
print(f"MAPE : {naive_mape:.4f} %")

print("\nPROJECT HEAT-SHIELD PH")
print("-" * 40)
print(f"RMSE : {heatshield_rmse:.4f} °C")
print(f"MAE  : {heatshield_mae:.4f} °C")
print(f"MAPE : {heatshield_mape:.4f} %")

# =========================================================
# 16. MODEL IMPROVEMENT
# =========================================================

print("\n")
print("=" * 65)
print("HEAT-SHIELD PH IMPROVEMENT OVER NAÏVE BASELINE")
print("=" * 65)

print(
    f"RMSE improvement : "
    f"{rmse_improvement:.2f} %"
)

print(
    f"MAE improvement  : "
    f"{mae_improvement:.2f} %"
)

print(
    f"MAPE improvement : "
    f"{mape_improvement:.2f} %"
)

# =========================================================
# 17. STATION-LEVEL WINS
# =========================================================

print("\n")
print("=" * 65)
print("STATION-LEVEL PERFORMANCE")
print("=" * 65)

print(
    f"HEAT-SHIELD PH lower RMSE: "
    f"{heatshield_rmse_wins}/{total_stations} stations"
)

print(
    f"HEAT-SHIELD PH lower MAE : "
    f"{heatshield_mae_wins}/{total_stations} stations"
)

print(
    f"HEAT-SHIELD PH lower MAPE: "
    f"{heatshield_mape_wins}/{total_stations} stations"
)

# =========================================================
# 18. CREATE SUMMARY TABLE
# =========================================================

summary_df = pd.DataFrame({

    "Model": [
        "Naive Persistence Baseline",
        "PROJECT HEAT-SHIELD PH"
    ],

    "RMSE_C": [
        naive_rmse,
        heatshield_rmse
    ],

    "MAE_C": [
        naive_mae,
        heatshield_mae
    ],

    "MAPE_pct": [
        naive_mape,
        heatshield_mape
    ]
})

# =========================================================
# 19. ADD IMPROVEMENT SUMMARY
# =========================================================

improvement_df = pd.DataFrame({

    "Metric": [
        "RMSE",
        "MAE",
        "MAPE"
    ],

    "Baseline_Value": [
        naive_rmse,
        naive_mae,
        naive_mape
    ],

    "HEAT_SHIELD_Value": [
        heatshield_rmse,
        heatshield_mae,
        heatshield_mape
    ],

    "Improvement_pct": [
        rmse_improvement,
        mae_improvement,
        mape_improvement
    ]
})

# =========================================================
# 20. SAVE CSV FILES
# =========================================================

results_df.to_csv(
    "HEAT_SHIELD_MODEL_BENCHMARK.csv",
    index=False
)

summary_df.to_csv(
    "HEAT_SHIELD_BENCHMARK_SUMMARY.csv",
    index=False
)

improvement_df.to_csv(
    "HEAT_SHIELD_BENCHMARK_IMPROVEMENT.csv",
    index=False
)

# =========================================================
# 21. FINAL OUTPUT
# =========================================================

print("\n")
print("=" * 65)
print("FILES SUCCESSFULLY SAVED")
print("=" * 65)

print(
    "1. HEAT_SHIELD_MODEL_BENCHMARK.csv"
)

print(
    "2. HEAT_SHIELD_BENCHMARK_SUMMARY.csv"
)

print(
    "3. HEAT_SHIELD_BENCHMARK_IMPROVEMENT.csv"
)

print("=" * 65)

print("\nBenchmark comparison completed successfully.")

input("\nPress ENTER to close...")