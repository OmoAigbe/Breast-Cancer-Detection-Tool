#!/usr/bin/env python3
# """
# scripts/7_extract_histogram_based_features.py
#
# Extract histogram-based features (mean, variance, std, skewness) from images under:
#  - data/sample/0/ and data/sample/1/  (flat class folders)
# or
#  - data/sample/<patient_id>/<class>/*  (nested patient folders)
#
# Saves results to data/features/Histogram_Features.csv
# Run from repo root:
#     python scripts/7_extract_histogram_based_features.py
# """

from pathlib import Path
import sys
import pandas as pd
from functools import reduce

# CONFIG
REPO_ROOT = Path.cwd()
FEATURE_DIR = REPO_ROOT / "data" / "features"
OUTPUT_CSV = FEATURE_DIR / "Merged_Features.csv"
OUTPUT_XLSX = FEATURE_DIR / "Merged_Features.xlsx"

FILES = {
    "rgb": FEATURE_DIR / "RGB_Features.csv",
    "glcm": FEATURE_DIR / "GLCM_Features.csv",
    "hsv": FEATURE_DIR / "HSV_Features.csv",
    "hist": FEATURE_DIR / "Histogram_Features.csv",
}

KEYS = ["patient_id", "path", "target"]

def load_df(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Expected feature file for {name} not found: {path}")
    df = pd.read_csv(path)
    if not set(KEYS).issubset(df.columns):
        raise ValueError(f"File {path} missing required key columns {KEYS}. Found columns: {list(df.columns)}")
    return df

def main():
    # Ensure feature dir exists
    if not FEATURE_DIR.exists():
        print(f"Feature directory not found: {FEATURE_DIR}", file=sys.stderr)
        sys.exit(1)

    # Load dataframes with a deterministic order
    dfs = []
    col_prefixes = []
    for name, path in FILES.items():
        print(f"Loading {name} features from {path} ...")
        df = load_df(path, name)
        dfs.append(df)
        col_prefixes.append(name)

    # To avoid collisions, drop duplicated key columns in right-hand tables during merge using suffixes.
    # Build a sequence of suffix pairs for merges: ('', '_glcm'), ('', '_hsv'), ...
    def merge_with_suffix(left, right, right_name):
        # choose suffix for right side only
        suffix = f"_{right_name}"
        # perform merge; left suffix blank keeps left columns intact
        return pd.merge(left, right, on=KEYS, how="inner", suffixes=("", suffix))

    # Start from the first df and merge others in order
    merged = dfs[0]
    merged_name = list(FILES.keys())[0]
    for idx, right in enumerate(dfs[1:], start=1):
        right_name = list(FILES.keys())[idx]
        print(f"Merging {right_name} features ...")
        merged = merge_with_suffix(merged, right, right_name)

    # Optional: reorder columns so keys come first
    other_cols = [c for c in merged.columns if c not in KEYS]
    merged = merged[KEYS + other_cols]

    # Persist
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged CSV to {OUTPUT_CSV}")

    try:
        merged.to_excel(OUTPUT_XLSX, index=False)
        print(f"Saved merged Excel to {OUTPUT_XLSX}")
    except Exception as e:
        print(f"Warning: failed to write Excel file: {e}", file=sys.stderr)

    print("Merged shape:", merged.shape)
    print("Columns:", merged.columns.tolist())

if __name__ == "__main__":
    main()