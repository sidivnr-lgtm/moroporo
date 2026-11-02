"""
04_climate_impacts.py — PROJECT HEAT-SHIELD PH
Compute 2030 climate impact projections per station.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "clean_daily_heat_index.csv"
PROJ_FILE = BASE_DIR / "heat_index_projections.csv"
OUTPUT_CSV = BASE_DIR / "2030_Climate_Impact_Projections.csv"
OUTPUT_PLOT = BASE_DIR / "climate_impact_projections_bar.png"

ACADEMIC_MONTHS = {6, 7, 8, 9, 10, 11, 12, 1, 2, 3}
DANGER_HI = 42.0
LABOR_HI_REF = 38.0
ER_HI_REF = 37.0


def outdoor_labor_loss(hi: float) -> float:
    """Outdoor Labor Capacity Loss (%)."""
    return 100.0 * (1.0 - 1.0 / (1.0 + np.exp(0.25 * (hi - LABOR_HI_REF))))


def hospital_er_surge(hi: float) -> float:
    """Hospital ER Surge Risk Index (%)."""
    return 100.0 * (np.exp(0.08 * (hi - ER_HI_REF)) - 1.0)


def estimate_suspension_days(df: pd.DataFrame, station: str, target_hi: float) -> float:
    """
    Estimate annual school suspension days (HI >= 42°C) during academic months,
    scaled to 2030 using recent-year frequency as baseline.
    """
    sub = df[(df["Station"] == station) & (df["Month"].isin(ACADEMIC_MONTHS))]
    if sub.empty:
        return 0.0
    recent = sub[sub["Year"] >= 2020]
    if recent.empty:
        recent = sub
    days_per_year = recent.groupby("Year").apply(
        lambda g: (g["Heat_Index"] >= DANGER_HI).sum()
    )
    return round(days_per_year.mean(), 1)


def load_projected_hi_2030() -> pd.DataFrame:
    if PROJ_FILE.exists():
        proj = pd.read_csv(PROJ_FILE)
        return proj[proj["Year"] == 2030][["Station", "Projected_Max_HI"]]
    return pd.DataFrame()


def compute_impacts(df: pd.DataFrame) -> pd.DataFrame:
    """Build 2030 impact table for all stations."""
    proj_2030 = load_projected_hi_2030()
    stations = sorted(df["Station"].unique())
    rows = []

    for station in stations:
        if not proj_2030.empty and station in proj_2030["Station"].values:
            hi_2030 = proj_2030.loc[
                proj_2030["Station"] == station, "Projected_Max_HI"
            ].iloc[0]
        else:
            hi_2030 = df[df["Station"] == station]["Heat_Index"].max()

        rows.append(
            {
                "Station": station,
                "Projected_Max_HI_2030_C": round(hi_2030, 1),
                "Outdoor_Labor_Capacity_Loss_Pct": round(
                    outdoor_labor_loss(hi_2030), 1
                ),
                "Projected_Annual_School_Suspension_Days": estimate_suspension_days(
                    df, station, hi_2030
                ),
                "Hospital_ER_Surge_Risk_Index_Pct": round(
                    max(hospital_er_surge(hi_2030), 0.0), 1
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_impacts(impacts: pd.DataFrame) -> None:
    """Bar chart comparing key 2030 impact metrics across stations."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    metrics = [
        ("Projected_Max_HI_2030_C", "2030 Projected Max HI (°C)", "#d73027"),
        ("Outdoor_Labor_Capacity_Loss_Pct", "Outdoor Labor Capacity Loss (%)", "#fc8d59"),
        (
            "Projected_Annual_School_Suspension_Days",
            "Projected School Suspension Days / yr",
            "#fee090",
        ),
        ("Hospital_ER_Surge_Risk_Index_Pct", "Hospital ER Surge Risk (%)", "#91bfdb"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        sorted_df = impacts.sort_values(col, ascending=True)
        ax.barh(sorted_df["Station"], sorted_df[col], color=color, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(title.split("(")[-1].rstrip(")") if "(" in title else "")

    fig.suptitle(
        "PROJECT HEAT-SHIELD PH — 2030 Climate Impact Projections by Station",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOT}")


def main() -> None:
    print("=" * 72)
    print("PROJECT HEAT-SHIELD PH — Climate Impact Projections")
    print("=" * 72)
    try:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Clean data not found: {INPUT_FILE}. Run 01_data_cleaning.py first."
            )
        df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
        impacts = compute_impacts(df)
        impacts.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")
        print("\n--- 2030 Climate Impact Projections ---")
        print(impacts.to_string(index=False))

        plot_impacts(impacts)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
