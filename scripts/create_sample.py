#!/usr/bin/env python3
import os
import random
import shutil
from collections import defaultdict

FULL_DATA_DIR = os.path.join("data", "full")  
SAMPLE_DIR = os.path.join("data", "sample")
SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42
TARGET_CLASSES = ["0", "1"]

random.seed(RANDOM_SEED)

def gather_image_paths(full_dir):
    if not os.path.isdir(full_dir):
        raise FileNotFoundError(f"FULL_DATA_DIR not found: {full_dir}")
    rows = []
    patient_folders = [d for d in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, d))]
    for patient_id in patient_folders:
        patient_path = os.path.join(full_dir, patient_id)
        for c in TARGET_CLASSES:
            class_path = os.path.join(patient_path, c)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    fpath = os.path.join(class_path, fname)
                    if os.path.isfile(fpath):
                        rows.append({"patient_id": patient_id, "path": fpath, "target": int(c)})
    return rows

def create_sample(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["target"]].append(r["path"])

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for target, paths in groups.items():
        target_dir = os.path.join(SAMPLE_DIR, str(target))
        os.makedirs(target_dir, exist_ok=True)
        k = min(SAMPLES_PER_CLASS, len(paths))
        sampled = random.sample(paths, k)
        for src in sampled:
            basename = os.path.basename(src)
            dst = os.path.join(target_dir, basename)
            if os.path.exists(dst):
                base, ext = os.path.splitext(basename)
                dst = os.path.join(target_dir, f"{base}_{os.path.basename(os.path.dirname(os.path.dirname(src)))}{ext}")
            shutil.copy2(src, dst)
        print(f"Copied {k} files for target {target} to {target_dir}")

def main():
    print("Gathering images from:", FULL_DATA_DIR)
    rows = gather_image_paths(FULL_DATA_DIR)
    counts = defaultdict(int)
    for r in rows:
        counts[r["target"]] += 1
    print("Available per-class counts:", dict(counts))
    create_sample(rows)
    print("Sample creation complete. Sample located at:", SAMPLE_DIR)

if __name__ == "__main__":
    main()