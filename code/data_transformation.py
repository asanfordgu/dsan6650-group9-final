# %%
import pandas as pd
import numpy as np

def transform_ems_for_rl(ems_df):
    """
    Transform cleaned EMS events into RL-ready features.
    Assumes columns like: Dispatch DtTm, Response DtTm, Priority, lat, lon, etc.
    """

    df = ems_df.copy()

    # --- ensure datetime ---
    if "Dispatch DtTm" in df.columns:
        df["Dispatch DtTm"] = pd.to_datetime(df["Dispatch DtTm"], errors="coerce")
    if "Response DtTm" in df.columns:
        df["Response DtTm"] = pd.to_datetime(df["Response DtTm"], errors="coerce")

    # --- time-based features ---
    df["event_time"] = df["Dispatch DtTm"]
    df["event_hour"] = df["event_time"].dt.hour
    df["event_dow"] = df["event_time"].dt.dayofweek  # 0=Mon

    # --- response time (minutes) ---
    if {"Dispatch DtTm", "Response DtTm"}.issubset(df.columns):
        dt = (df["Response DtTm"] - df["Dispatch DtTm"]).dt.total_seconds() / 60.0
        df["response_time_min"] = dt.replace([np.inf, -np.inf], np.nan)
    else:
        df["response_time_min"] = np.nan

    # --- priority features ---
    # try to convert priority to numeric if possible
    if "Priority" in df.columns:
        df["Priority_num"] = pd.to_numeric(df["Priority"], errors="coerce")
        df["high_priority"] = df["Priority_num"].isin([1, 2])  # adjust if needed
    else:
        df["Priority_num"] = np.nan
        df["high_priority"] = False

    # --- feature selection: keep only what RL needs ---
    keep_cols = [
        "event_id" if "event_id" in df.columns else None,
        "Call Number" if "Call Number" in df.columns else None,
        "Incident Number" if "Incident Number" in df.columns else None,
        "Call Type" if "Call Type" in df.columns else None,
        "Call Type Group" if "Call Type Group" in df.columns else None,
        "Unit Type" if "Unit Type" in df.columns else None,
        "event_time",
        "event_hour",
        "event_dow",
        "lat" if "lat" in df.columns else None,
        "lon" if "lon" in df.columns else None,
        "Priority_num",
        "high_priority",
        "response_time_min",
    ]
    keep_cols = [c for c in keep_cols if c is not None]

    ems_rl = df[keep_cols].reset_index(drop=True)

    return ems_rl


# %%
ems_clean = pd.read_csv("../data/3_clean_dataset/cleaned_emergency_logs.csv")
ems_rl = transform_ems_for_rl(ems_clean)
ems_rl.to_csv('../data/4_transformed_dataset/transformed_emergency_logs.csv')


# %%
import pandas as pd

def transform_pems_metadata_for_rl(meta_df):
    """
    Transform cleaned PeMS station metadata into RL-ready station features.
    """

    df = meta_df.copy()

    # Ensure numeric
    for col in ["Latitude", "Longitude", "Lanes", "District"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic filters (should already be done, but safe)
    if "District" in df.columns:
        df = df[df["District"] == 4]

    if "Type" in df.columns:
        df = df[df["Type"] == "ML"]

    # One-hot encode direction if present
    if "Freeway Direction" in df.columns:
        dir_dummies = pd.get_dummies(df["Freeway Direction"], prefix="dir", drop_first=False)
        df = pd.concat([df, dir_dummies], axis=1)

    # Simple freeway flag example (adjust list as needed)
    if "Freeway" in df.columns:
        df["is_101"] = (df["Freeway"] == df["Freeway"].mode().iloc[0])  # or explicit == 101

    # Feature selection
    keep_cols = [
        "ID",
        "Freeway" if "Freeway" in df.columns else None,
        "Freeway Direction" if "Freeway Direction" in df.columns else None,
        "Latitude",
        "Longitude",
        "Lanes",
        "Type" if "Type" in df.columns else None,
        "City" if "City" in df.columns else None,
    ]
    # include any dir_ one-hot columns
    keep_cols += [c for c in df.columns if c.startswith("dir_")]
    if "is_101" in df.columns:
        keep_cols.append("is_101")

    keep_cols = [c for c in keep_cols if c is not None and c in df.columns]

    meta_rl = df[keep_cols].reset_index(drop=True)

    return meta_rl


# %%
meta_clean = pd.read_csv("../data/3_clean_dataset/cleaned_station_metadata.csv")
meta_rl = transform_pems_metadata_for_rl(meta_clean)
meta_rl.to_csv('../data/4_transformed_dataset/transformed_station_metadata.csv')


# %%
import pandas as pd

def transform_pems_day_for_rl(parquet_path, stations_subset=None):
    """
    Transform one cleaned PeMS 5-min day parquet file into RL-ready traffic features.
    - parquet_path: path to a d04_text_station_5min_YYYY_MM_DD.parquet
    - stations_subset: optional list of station IDs to keep
    """

    df = pd.read_parquet(parquet_path)

    # Ensure necessary types
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    for col in ["Total Flow", "Avg Speed", "Avg Occupancy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional station filter
    if stations_subset is not None and "Station" in df.columns:
        df = df[df["Station"].astype(str).isin([str(s) for s in stations_subset])]

    # Time features
    df["hour"] = df["Timestamp"].dt.hour
    df["dow"] = df["Timestamp"].dt.dayofweek

    # Feature selection
    keep_cols = [
        "Timestamp",
        "Station" if "Station" in df.columns else None,
        "Total Flow" if "Total Flow" in df.columns else None,
        "Avg Speed" if "Avg Speed" in df.columns else None,
        "Avg Occupancy" if "Avg Occupancy" in df.columns else None,
        "hour",
        "dow",
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in df.columns]

    day_rl = df[keep_cols].dropna(subset=["Timestamp"]).reset_index(drop=True)

    return day_rl


# %%
import glob
import os
import pandas as pd  # needed because transform_pems_day_for_rl uses pandas

# make sure this function is defined somewhere above:
# from your previous code:
# def transform_pems_day_for_rl(parquet_path, stations_subset=None): ...

INPUT_DIR = "../data/3_clean_dataset"
OUTPUT_DIR = "../data/4_transformed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parquet_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "d04_text_station_5min_2025_*.parquet"))
)

print("Found parquet files:", len(parquet_files))

bad_files = []

for fp in parquet_files:
    try:
        day_rl = transform_pems_day_for_rl(fp)

        base = os.path.basename(fp)  # e.g. d04_text_station_5min_2025_01_06.parquet
        out_name = base.replace(".parquet", "_rl.parquet")
        out_path = os.path.join(OUTPUT_DIR, out_name)

        day_rl.to_parquet(out_path, index=False)
        print("saved:", out_path)

    except Exception as e:
        bad_files.append((fp, repr(e)))
        print("skipping:", fp, "->", repr(e))

print("\nDone. Failed files:", len(bad_files))




