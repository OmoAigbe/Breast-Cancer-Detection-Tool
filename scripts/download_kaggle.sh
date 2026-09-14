#!/usr/bin/env bash

set -euo pipefail

SLUG="${1:-}"

if [ -z "$SLUG" ]; then
  read -p "Enter Kaggle dataset slug (owner/dataset-name): " SLUG
fi

if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found. Place your kaggle.json there and run 'chmod 600 ~/.kaggle/kaggle.json'"
  exit 1
fi

DEST_DIR="data/full"
mkdir -p "$DEST_DIR"

echo "Downloading Kaggle dataset '${SLUG}' into ${DEST_DIR} ..."
kaggle datasets download -d "$SLUG" -p "$DEST_DIR" --unzip

echo "Download complete. Files placed under ${DEST_DIR}."