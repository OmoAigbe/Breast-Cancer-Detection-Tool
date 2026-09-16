#!/usr/bin/env python3
#"""
#scripts/extract_rgb_features.py

#Extract simple RGB summary features (mean R,G,B and average color) from images found under:
 #- data/sample/0/ and data/sample/1/  (flat class folders)
#or
 #- data/sample/<patient_id>/<class>/*  (nested patient folders)

#Saves results to data/features/RGB_Features.csv
#"""

import sys
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# CONFIG
REPO_ROOT = Path.cwd()  # run from repo root
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
OUTPUT_DIR = REPO_ROOT / "data" / "features"
OUTPUT_CSV = OUTPUT_DIR / "RGB_Features.csv"
TARGET_CLASSES = ("0", "1")  # folder names for classes
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def gather_paths(sample_dir: Path):
    """Return list of dicts: {'patient_id', 'path', 'target'}"""
    rows = []
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Case A: flat layout data/sample/0/* , data/sample/1/*
    found_flat = any((sample_dir / cls).exists() for cls in TARGET_CLASSES)
    if found_flat:
        for cls in TARGET_CLASSES:
            cls_dir = sample_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.is_file():
                    rows.append({"patient_id": None, "path": p, "target": int(cls)})
        return rows

    # Case B: nested patient folders sample/<patient_id>/<class>/*
    for patient in sample_dir.iterdir():
        if not patient.is_dir():
            continue
        pid = patient.name
        for cls in TARGET_CLASSES:
            cls_dir = patient / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.is_file():
                    rows.append({"patient_id": pid, "path": p, "target": int(cls)})
    return rows

def extract_rgb_features_from_image(img_bgr: np.ndarray):
    """Accepts image read by cv2 (BGR), returns mean_r, mean_g, mean_b, color_ratio"""
    if img_bgr is None:
        return (np.nan, np.nan, np.nan, np.nan)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Ensure float computation
    img_rgb = img_rgb.astype(np.float32)
    # Compute channel means
    mean_r = float(np.mean(img_rgb[:, :, 0]))
    mean_g = float(np.mean(img_rgb[:, :, 1]))
    mean_b = float(np.mean(img_rgb[:, :, 2]))
    color_ratio = float((mean_r + mean_g + mean_b) / 3.0)
    return mean_r, mean_g, mean_b, color_ratio

def main():
    rows = gather_paths(SAMPLE_DIR)
    if not rows:
        print(f"No images found under {SAMPLE_DIR}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build dataframe by iterating once and collecting records in a list (fast)
    records = []
    for r in tqdm(rows, desc="Processing images"):
        image_path = r["path"]
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("cv2.imread returned None")
            mean_r, mean_g, mean_b, color_ratio = extract_rgb_features_from_image(img)
        except Exception as e:
            # On read error, store NaNs and continue
            mean_r = mean_g = mean_b = color_ratio = float("nan")
            print(f"Warning: failed to read/process {image_path}: {e}", file=sys.stderr)

        records.append({
            "patient_id": r["patient_id"],
            "path": str(image_path),
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "color_ratio": color_ratio,
            "target": r["target"]
        })

    df_rgb = pd.DataFrame.from_records(records, columns=["patient_id", "path", "mean_r", "mean_g", "mean_b", "color_ratio", "target"])
    df_rgb.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved RGB features to {OUTPUT_CSV}")
    print(df_rgb.head())

if __name__ == "__main__":
    main()