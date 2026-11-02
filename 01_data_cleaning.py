"""
01_data_cleaning.py — PROJECT HEAT-SHIELD PH
Load, clean, and merge PAGASA heat-index and daily station records.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HEAT_INDEX_FILE = BASE_DIR / "Daily Heat Index Data.csv"
OUTPUT_FILE = BASE_DIR / "clean_daily_heat_index.csv"

STATIONS = [
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
    "Zamboanga City",
]

DAILY_FILE_MAP = {
    "Baguio City": "Baguio Daily Data.csv",
    "Basco Radar": "Basco Radar Daily Data.csv",
    "Laoag City": "Laoag Daily Data.csv",
    "Legazpi City": "Legazpi Daily Data.csv",
    "Mactan Int Airport": "Mactan Daily Data.csv",
    "Puerto Princesa City": "Puerto Princesa Daily Data.csv",
    "Science Garden QC": "Science Garden Daily Data.csv",
    "Tacloban City": "Tacloban Daily Data.csv",
    "Tuguegarao City": "Tuguegarao Daily Data.csv",
    "Catarman": "Catarman Daily Data.csv",
    "Davao City": "Davao City Daily Data.csv",
    "Dumaguete City": "Dumaguete Daily Data.csv",
    "General Santos City": "General Santos Daily Data.csv",
    "Malaybalay": "Malaybalay Daily Data.csv",
    "San Jose": "San Jose Daily Data.csv",
    "Surigao City": "Surigao Daily Data.csv",
    "Zamboanga City": "Zamboanga Daily Data.csv",
}

SENTINEL = -999.0
HI_MIN, HI_MAX = 10.0, 70.0
DANGER_THRESHOLD = 42.0


def _standardize_station_name(col: str) -> str:
    """Map raw heat-index column header to a short station label."""
    name = re.sub(r"\s*Heat\.Index\.Max\.Dly.*", "", col).strip()
    replacements = {
        "Baguio City, Benguet": "Baguio City",
        "Basco (Radar), Batanes": "Basco Radar",
        "Catarman, Northern Samar": "Catarman",
        "Davao City, Davao Del Sur": "Davao City",
        "Dumaguete City, Negros Oriental": "Dumaguete City",
        "General Santos City, South Cotabato": "General Santos City",
        "Laoag City, Ilocos Norte": "Laoag City",
        "Legazpi City, Albay": "Legazpi City",
        "Mactan International Airport, Cebu": "Mactan Int Airport",
        "Malaybalay, Bukidnon": "Malaybalay",
        "Puerto Princesa City, Palawan": "Puerto Princesa City",
        "San Jose, Occidental Mindoro": "San Jose",
        "Science Garden Quezon City, Metro Manila": "Science Garden QC",
        "Surigao City, Surigao Del Norte": "Surigao City",
        "Tacloban City, Leyte": "Tacloban City",
        "Tuguegarao City, Cagayan": "Tuguegarao City",
        "Zamboanga City, Zamboanga Del Sur": "Zamboanga City",
    }
    return replacements.get(name, name)


def load_heat_index(path: Path) -> pd.DataFrame:
    """Load heat-index CSV, drop metadata row, clean values."""
    if not path.exists():
        raise FileNotFoundError(f"Heat index file not found: {path}")

    df = pd.read_csv(path)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Source")].copy()

    date_col = df.columns[0]
    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", errors="coerce")

    station_cols = [c for c in df.columns if c != "Date"]
    rename_map = {_standardize_station_name(c): c for c in station_cols}
    inv_map = {v: k for k, v in rename_map.items()}

    for orig_col in station_cols:
        df[orig_col] = pd.to_numeric(df[orig_col], errors="coerce")
        mask = (
            (df[orig_col] == SENTINEL)
            | (df[orig_col] < HI_MIN)
            | (df[orig_col] > HI_MAX)
        )
        df.loc[mask, orig_col] = np.nan

    df.rename(columns=inv_map, inplace=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df


def load_daily_station(station: str, filename: str) -> pd.DataFrame | None:
    """Load a single daily meteorological station file."""
    path = BASE_DIR / filename
    if not path.exists():
        print(f"  [WARN] Daily file missing for {station}: {filename}")
        return None

    daily = pd.read_csv(path)
    daily.columns = [c.strip().upper() for c in daily.columns]
    daily["Date"] = pd.to_datetime(
        dict(year=daily["YEAR"], month=daily["MONTH"], day=daily["DAY"]),
        errors="coerce",
    )
    daily["Station"] = station
    for col in ("RAINFALL", "TMAX", "TMIN", "RH", "WIND_SPEED", "WIND_DIRECTION"):
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
            daily.loc[daily[col] == SENTINEL, col] = np.nan
    return daily


def melt_heat_index(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide heat-index frame to long format."""
    id_cols = ["Date", "Year", "Month"]
    value_cols = [s for s in STATIONS if s in wide.columns]
    long_df = wide.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="Station",
        value_name="Heat_Index",
    )
    return long_df.sort_values(["Station", "Date"]).reset_index(drop=True)


def merge_daily_records(long_hi: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference and merge available daily station CSVs."""
    daily_frames = []
    for station, filename in DAILY_FILE_MAP.items():
        daily = load_daily_station(station, filename)
        if daily is not None:
            daily_frames.append(daily)

    if not daily_frames:
        return long_hi

    all_daily = pd.concat(daily_frames, ignore_index=True)
    merge_cols = ["Date", "Station"]
    daily_cols = [
        c
        for c in all_daily.columns
        if c not in ("YEAR", "MONTH", "DAY")
    ]
    merged = long_hi.merge(all_daily[daily_cols], on=merge_cols, how="left")
    return merged


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table: danger days, max HI, mean HI per station."""
    summary = (
        long_df.groupby("Station")["Heat_Index"]
        .agg(
            Days_GTE_42C=lambda s: (s >= DANGER_THRESHOLD).sum(),
            Historical_Max_HI_C="max",
            Mean_HI_C="mean",
        )
        .reset_index()
    )
    summary.rename(
        columns={
            "Days_GTE_42C": "Days >= 42°C (PAGASA Danger)",
            "Historical_Max_HI_C": "Historical Max HI (°C)",
            "Mean_HI_C": "Mean HI (°C)",
        },
        inplace=True,
    )
    summary["Historical Max HI (°C)"] = summary["Historical Max HI (°C)"].round(1)
    summary["Mean HI (°C)"] = summary["Mean HI (°C)"].round(1)
    station_order = {s: i for i, s in enumerate(STATIONS)}
    summary["_order"] = summary["Station"].map(station_order)
    return summary.sort_values("_order").drop(columns="_order")


def main() -> None:
    print("=" * 72)
    print("PROJECT HEAT-SHIELD PH — Data Cleaning Pipeline")
    print("=" * 72)

    try:
        wide = load_heat_index(HEAT_INDEX_FILE)
        print(f"Loaded heat index: {len(wide):,} daily records, "
              f"{wide['Date'].min().date()} to {wide['Date'].max().date()}")

        long_df = melt_heat_index(wide)
        print("Merging daily station meteorological records …")
        clean = merge_daily_records(long_df)
        clean.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved: {OUTPUT_FILE}")

        summary = build_summary(clean)
        print("\n--- Station Summary ---")
        print(summary.to_string(index=False))
        summary.to_csv(BASE_DIR / "station_summary.csv", index=False)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
