"""
03_heatwave_projections.py — PROJECT HEAT-SHIELD PH
Non-stationary extreme-value trend projections through 2030.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "clean_daily_heat_index.csv"
OUTPUT_PLOT = BASE_DIR / "projected_heat_index_2025_2030.png"
OUTPUT_CSV = BASE_DIR / "heat_index_projections.csv"

BASE_YEAR = 2015
PROJ_END = 2030
PHYSICAL_CAP = 65.0

HIGH_RISK_STATIONS = [
    "Legazpi City",
    "Catarman",
    "San Jose",
    "Tuguegarao City",
    "Davao City",
    "Science Garden QC",
    "Baguio City",
]

ALL_STATIONS = [
    "Baguio City", "Basco Radar", "Catarman", "Davao City",
    "Dumaguete City", "General Santos City", "Laoag City", "Legazpi City",
    "Mactan Int Airport", "Malaybalay", "Puerto Princesa City", "San Jose",
    "Science Garden QC", "Surigao City", "Tacloban City", "Tuguegarao City",
    "Zamboanga City",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean data not found: {path}. Run 01_data_cleaning.py first."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def annual_maxima(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual maximum heat index per station."""
    return (
        df.groupby(["Year", "Station"])["Heat_Index"]
        .max()
        .reset_index()
        .rename(columns={"Heat_Index": "Annual_Max_HI"})
    )


def fit_and_project(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Fit linear trend on shifted time coordinate X = Year - 2015
    and project annual maxima through 2030.
    """
    records = []
    all_years = list(range(annual["Year"].min(), PROJ_END + 1))

    for station in ALL_STATIONS:
        sub = annual[annual["Station"] == station].dropna()
        if len(sub) < 3:
            continue

        x = (sub["Year"] - BASE_YEAR).values.reshape(-1, 1)
        y = sub["Annual_Max_HI"].values
        model = LinearRegression()
        model.fit(x, y)

        for year in all_years:
            x_pred = np.array([[year - BASE_YEAR]])
            raw = model.predict(x_pred)[0]
            capped = min(raw, PHYSICAL_CAP)
            records.append(
                {
                    "Station": station,
                    "Year": year,
                    "Projected_Max_HI": round(capped, 2),
                    "Slope_per_year": round(model.coef_[0], 4),
                    "Is_Projected": year > sub["Year"].max(),
                }
            )

    return pd.DataFrame(records)


def plot_projections(annual: pd.DataFrame, projections: pd.DataFrame) -> None:
    """Compare historical vs projected trends for high-risk stations."""
    n = len(HIGH_RISK_STATIONS)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, station in enumerate(HIGH_RISK_STATIONS):
        ax = axes[i]
        hist = annual[annual["Station"] == station]
        proj = projections[projections["Station"] == station]

        ax.scatter(
            hist["Year"], hist["Annual_Max_HI"],
            color="#2166ac", s=40, zorder=3, label="Historical Annual Max",
        )
        ax.plot(
            proj["Year"], proj["Projected_Max_HI"],
            color="#b2182b", linewidth=2, linestyle="--", label="Trend / Projection",
        )
        ax.axhline(PHYSICAL_CAP, color="gray", linestyle=":", alpha=0.6,
                   label=f"Physiological cap ({PHYSICAL_CAP}°C)")
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Max Heat Index (°C)")
        ax.set_xlim(2014, PROJ_END + 1)
        ax.legend(fontsize=7, loc="upper left")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "PROJECT HEAT-SHIELD PH — Historical vs Projected Peak Heat Index (2030)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOT}")


def main() -> None:
    print("=" * 72)
    print("PROJECT HEAT-SHIELD PH — Heatwave Projections")
    print("=" * 72)
    try:
        df = load_data(INPUT_FILE)
        annual = annual_maxima(df)
        projections = fit_and_project(annual)
        projections.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")

        plot_projections(annual, projections)

        proj_2030 = projections[
            (projections["Year"] == 2030) & projections["Is_Projected"]
        ][["Station", "Projected_Max_HI", "Slope_per_year"]]
        print("\n--- 2030 Projected Peak Heat Index ---")
        print(proj_2030.to_string(index=False))

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
