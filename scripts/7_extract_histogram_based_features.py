#!/usr/bin/env python3
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
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from scipy.stats import skew

# CONFIG
REPO_ROOT = Path.cwd()
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
OUTPUT_DIR = REPO_ROOT / "data" / "features"
OUTPUT_CSV = OUTPUT_DIR / "Histogram_Features.csv"
TARGET_CLASSES = ("0", "1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def gather_paths(sample_dir: Path):
    rows = []
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Case A: flat layout data/sample/0/* , data/sample/1/*
    if any((sample_dir / cls).exists() for cls in TARGET_CLASSES):
        for cls in TARGET_CLASSES:
            cls_dir = sample_dir / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file():
                    rows.append({"patient_id": None, "path": p, "target": int(cls)})
        return rows

    # Case B: nested patient folders sample/<patient_id>/<class>/*
    for patient in sorted(sample_dir.iterdir()):
        if not patient.is_dir():
            continue
        pid = patient.name
        for cls in TARGET_CLASSES:
            cls_dir = patient / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file():
                    rows.append({"patient_id": pid, "path": p, "target": int(cls)})
    return rows

def extract_histogram_features_from_bgr(img_bgr: np.ndarray):
    """
    Accepts image as returned by cv2.imread (BGR),
    returns mean, variance, std_dev, skewness of grayscale intensities.
    """
    if img_bgr is None:
        return (np.nan, np.nan, np.nan, np.nan)
    # Convert to grayscale directly from BGR
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(np.mean(gray))
    variance = float(np.var(gray))
    std_dev = float(np.std(gray))
    # compute skew on flattened vector; guard against constant images
    flat = gray.flatten()
    try:
        skewness = float(skew(flat))
    except Exception:
        skewness = float("nan")
    return mean, variance, std_dev, skewness

def main():
    rows = gather_paths(SAMPLE_DIR)
    if not rows:
        print(f"No images found under {SAMPLE_DIR}. Exiting.", file=sys.stderr)
        sys.exit(1)

    records = []
    for r in tqdm(rows, desc="Processing images"):
        img_path = r["path"]
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError("cv2.imread returned None (unreadable file)")
            mean, variance, std_dev, skewness = extract_histogram_features_from_bgr(img_bgr)
        except Exception as e:
            print(f"Warning: failed to process {img_path}: {e}", file=sys.stderr)
            mean = variance = std_dev = skewness = float("nan")

        records.append({
            "patient_id": r["patient_id"],
            "path": str(img_path),
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "skewness": skewness,
            "target": r["target"]
        })

    df_hist = pd.DataFrame.from_records(records, columns=["patient_id", "path", "mean", "variance", "std_dev", "skewness", "target"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_hist.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved histogram features to {OUTPUT_CSV}")
    print(df_hist.head())

if __name__ == "__main__":
    main()