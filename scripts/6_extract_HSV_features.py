#!/usr/bin/env python3
#!/usr/bin/env python3
# """
# scripts/6_extract_HSV_features.py
#
# Extract mean HSV features from images under data/sample and write CSV to data/features/HSV_Features.csv.
#
# Run from repo root:
#     python scripts/6_extract_HSV_features.py
# """

from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# CONFIG
REPO_ROOT = Path.cwd()                # run script from repo root
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
OUTPUT_DIR = REPO_ROOT / "data" / "features"
OUTPUT_CSV = OUTPUT_DIR / "HSV_Features.csv"
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

def extract_hsv_mean_from_bgr(img_bgr: np.ndarray):
    """
    Accepts image as returned by cv2.imread (BGR),
    returns mean_h, mean_s, mean_v (as floats).
    Note: OpenCV H ranges 0..179 for 8-bit images.
    """
    if img_bgr is None:
        return (np.nan, np.nan, np.nan)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # BGR -> HSV
    # convert to float and compute means
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    return h_mean, s_mean, v_mean

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
            h_mean, s_mean, v_mean = extract_hsv_mean_from_bgr(img_bgr)
        except Exception as e:
            print(f"Warning: failed to process {img_path}: {e}", file=sys.stderr)
            h_mean = s_mean = v_mean = float("nan")

        records.append({
            "patient_id": r["patient_id"],
            "path": str(img_path),
            "hue": h_mean,
            "saturation": s_mean,
            "value": v_mean,
            "target": r["target"]
        })

    df_hsv = pd.DataFrame.from_records(records, columns=["patient_id", "path", "hue", "saturation", "value", "target"])
    df_hsv.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved HSV features to {OUTPUT_CSV}")
    print(df_hsv.head())

if __name__ == "__main__":
    main()