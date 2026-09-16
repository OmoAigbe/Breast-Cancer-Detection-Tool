#!/usr/bin/env python3
#"""
#scripts/extract_glcm_features.py

#Extract GLCM features (contrast, correlation, energy, homogeneity) from images
#found under data/sample and save results to data/features/GLCM_Features.csv.

#Run from repo root:
#    python scripts/extract_glcm_features.py
#"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from skimage.feature import graycomatrix, graycoprops

# CONFIG
REPO_ROOT = Path.cwd()
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
OUTPUT_DIR = REPO_ROOT / "data" / "features"
OUTPUT_CSV = OUTPUT_DIR / "GLCM_Features.csv"
TARGET_CLASSES = ("0", "1")

NUM_LEVELS = 256
DISTANCES = [1]
ANGLES = [0]  

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def gather_paths(sample_dir: Path):
    """Scan the sample directory for flat layouts or nested patient folders."""
    rows = []
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Case A: flat layout data/sample/0/* , data/sample/1/*
    if any((sample_dir / cls).exists() for cls in TARGET_CLASSES):
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

def extract_glcm_props(img_gray: np.ndarray, distances, angles, levels):   
    """Extract GLCM properties using 256 grey levels."""
    # Ensure correct spelling of graycomatrix and graycoprops
    glcm = graycomatrix(img_gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast').mean())
    correlation = float(graycoprops(glcm, 'correlation').mean())
    energy = float(graycoprops(glcm, 'energy').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    return contrast, correlation, energy, homogeneity

def main():
    rows = gather_paths(SAMPLE_DIR)
    if not rows:
        print(f"No images found under {SAMPLE_DIR}. Exiting.", file=sys.stderr)
        sys.exit(1)

    records = []
    for r in tqdm(rows, desc="Processing images"):
        img_path = r["path"]
        try:
            # cv2.imread returns BGR or None
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError("cv2.imread returned None (file missing or unreadable)")
            
            # Convert to grayscale (BGR -> GRAY)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Compute glcm props directly from the raw 256 grayscale image
            contrast, correlation, energy, homogeneity = extract_glcm_props(img_gray, DISTANCES, ANGLES, NUM_LEVELS)
            
        except Exception as e:
            print(f"Warning: failed to process {img_path}: {e}", file=sys.stderr)
            contrast = correlation = energy = homogeneity = float("nan")

        records.append({
            "patient_id": r["patient_id"],
            "path": str(img_path),
            "contrast": contrast,
            "correlation": correlation,
            "energy": energy,
            "homogeneity": homogeneity,
            "target": r["target"]
        })

    # Build DataFrame rapidly from the accumulated records list
    df_glcm = pd.DataFrame.from_records(records, columns=["patient_id", "path", "contrast", "correlation", "energy", "homogeneity", "target"])
    df_glcm.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved GLCM features to {OUTPUT_CSV}")
    print(df_glcm.head())

if __name__ == "__main__":
    main()
