"""
02_spatiotemporal_heatmaps.py — PROJECT HEAT-SHIELD PH
Generate spatiotemporal heat-index heatmaps from cleaned data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "clean_daily_heat_index.csv"
OUTPUT_ANNUAL = BASE_DIR / "spatiotemporal_heat_index_heatmap.png"
OUTPUT_MONTHLY = BASE_DIR / "monthly_heatwave_seasonality_heatmap.png"

STATIONS = [
    "Baguio City", "Basco Radar", "Catarman", "Davao City",
    "Dumaguete City", "General Santos City", "Laoag City", "Legazpi City",
    "Mactan Int Airport", "Malaybalay", "Puerto Princesa City", "San Jose",
    "Science Garden QC", "Surigao City", "Tacloban City", "Tuguegarao City",
    "Zamboanga City",
]

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean data not found: {path}. Run 01_data_cleaning.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def annual_mean_heatmap(df: pd.DataFrame) -> None:
    """Annual mean daily max heat index across all stations (2015–2024)."""
    annual = (
        df.groupby(["Year", "Station"])["Heat_Index"]
        .mean()
        .reset_index()
        .pivot(index="Station", columns="Year", values="Heat_Index")
    )
    station_order = [s for s in STATIONS if s in annual.index]
    annual = annual.reindex(station_order)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        annual,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Mean Daily Max Heat Index (°C)"},
        ax=ax,
    )
    ax.set_title(
        "Annual Mean Daily Maximum Heat Index — 17 PAGASA Stations (2015–2024)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    plt.tight_layout()
    fig.savefig(OUTPUT_ANNUAL, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_ANNUAL}")


def monthly_seasonality_heatmap(df: pd.DataFrame) -> None:
    """Monthly average heat index per station across all 12 months."""
    monthly = (
        df.groupby(["Month", "Station"])["Heat_Index"]
        .mean()
        .reset_index()
        .pivot(index="Station", columns="Month", values="Heat_Index")
    )
    station_order = [s for s in STATIONS if s in monthly.index]
    monthly = monthly.reindex(station_order)
    monthly.columns = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        monthly,
        annot=True,
        fmt=".1f",
        cmap="inferno",
        linewidths=0.5,
        cbar_kws={"label": "Mean Heat Index (°C)"},
        ax=ax,
    )
    ax.set_title(
        "Monthly Heatwave Seasonality — Average Heat Index by Station",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Station")
    plt.tight_layout()
    fig.savefig(OUTPUT_MONTHLY, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_MONTHLY}")


def main() -> None:
    print("=" * 72)
    print("PROJECT HEAT-SHIELD PH — Spatiotemporal Heatmaps")
    print("=" * 72)
    try:
        df = load_data(INPUT_FILE)
        annual_mean_heatmap(df)
        monthly_seasonality_heatmap(df)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
