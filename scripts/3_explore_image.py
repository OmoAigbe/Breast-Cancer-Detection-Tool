#!/usr/bin/env python3
#Load a small sample from data/sample/0 and data/sample/1, pick 5 random images
#from each class and display them side-by-side.
#Adjust SAMPLE_DIR if your sample is located elsewhere

import os
import random
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
import pandas as pd


# CONFIG
REPO_ROOT = Path.cwd()               # run from repo root
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
TARGET_CLASSES = ["0", "1"]          # class subfolders
SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def gather_image_paths(sample_dir):
    """Return list of dicts: {'target': int, 'path': Path, 'patient_id': str}"""
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    rows = []
    for patient_or_file in sample_dir.iterdir():
        # If structure is sample/<target>/<files> OR sample/<target> directly
        if patient_or_file.is_dir() and patient_or_file.name in TARGET_CLASSES:
            # case: sample/0/*.png
            target = int(patient_or_file.name)
            for f in patient_or_file.iterdir():
                if f.is_file():
                    rows.append({"patient_id": None, "path": f, "target": target})
        else:
            # try sample/<patient_id>/<class> structure
            # sample/<patient_id>/<class>/*  (as in original layout)
            # iterate patient folders
            break

    # If rows empty, try patient_id/class layout
    if not rows:
        # look for patient directories
        for patient in sample_dir.iterdir():
            if not patient.is_dir():
                continue
            pid = patient.name
            for c in TARGET_CLASSES:
                class_dir = patient / c
                if class_dir.exists() and class_dir.is_dir():
                    for f in class_dir.iterdir():
                        if f.is_file():
                            rows.append({"patient_id": pid, "path": f, "target": int(c)})
    return rows

def build_dataframe(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found in sample directory.")
    return df

def sample_and_plot(df, n_per_class=SAMPLES_PER_CLASS):
    # sample per class
    sampled = pd.concat([
        df[df["target"] == 0].sample(n=n_per_class, random_state=RANDOM_SEED),
        df[df["target"] == 1].sample(n=n_per_class, random_state=RANDOM_SEED)
    ]).reset_index(drop=True)

    # prepare grid
    fig, axes = plt.subplots(n_per_class, 2, figsize=(8, 4 * n_per_class))
    if n_per_class == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_per_class):
        # negative (target 0)
        neg_row = sampled[sampled["target"] == 0].iloc[i]
        img_neg = imread(str(neg_row["path"]))
        axes[i, 0].imshow(img_neg)
        axes[i, 0].set_title(f"IDC Negative #{i+1}")
        axes[i, 0].axis("off")

        # positive (target 1)
        pos_row = sampled[sampled["target"] == 1].iloc[i]
        img_pos = imread(str(pos_row["path"]))
        axes[i, 1].imshow(img_pos)
        axes[i, 1].set_title(f"IDC Positive #{i+1}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    print("Looking for images in:", SAMPLE_DIR)
    rows = gather_image_paths(SAMPLE_DIR)
    df = build_dataframe(rows)
    print("Found counts per class:\n", df["target"].value_counts().to_dict())
    sample_and_plot(df)

if __name__ == "__main__":
    main()
