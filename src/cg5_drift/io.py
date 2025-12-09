# -*- coding: utf-8 -*-
"""
Input/output utilities for CG-5 base-station time series.
"""

import os
import pandas as pd
import numpy as np


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CG-5 base-station CSV file and return a cleaned DataFrame.

    Expected columns (semicolon-separated):
        DATE, TIME, GRAV*, SD

    Returns a DataFrame with:
        DATETIME   : pandas.Timestamp
        g          : gravity (mGal)
        sigma      : internal standard deviation (mGal)
        source_file: original filename
    """
    df = pd.read_csv(filepath, sep=";")
    df.columns = [c.strip().upper() for c in df.columns]

    if "SD" not in df.columns:
        raise ValueError(f"SD column not found in {filepath}")

    # Identify gravity column (any column containing 'GRAV')
    g_col = next((c for c in df.columns if "GRAV" in c), "GRAV")

    # Robust datetime parsing (two common CG-5 formats)
    try:
        dt = pd.to_datetime(
            df["DATE"] + " " + df["TIME"],
            format="%d/%m/%Y %H:%M:%S",
            errors="raise",
        )
    except Exception:
        dt = pd.to_datetime(
            df["DATE"] + " " + df["TIME"],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    out = pd.DataFrame(
        {
            "DATETIME": dt,
            "g": pd.to_numeric(df[g_col], errors="coerce"),
            "sigma": pd.to_numeric(df["SD"], errors="coerce"),
        }
    )

    out = (
        out.dropna(subset=["DATETIME", "g"])
        .sort_values("DATETIME")
        .reset_index(drop=True)
    )
    out["source_file"] = os.path.basename(filepath)
    return out


def to_seconds(x_dt: np.ndarray, x0) -> np.ndarray:
    """Convert datetime array to seconds from reference datetime x0."""
    return (x_dt - x0) / np.timedelta64(1, "s")


def site_label_from_filename(fname: str) -> str:
    """Extract site label from CSV filename (without extension)."""
    return os.path.splitext(os.path.basename(fname))[0]
