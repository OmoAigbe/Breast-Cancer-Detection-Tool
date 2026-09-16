# Breast Cancer Detection Tool

A machine learning project for identifying Invasive Ductal Carcinoma (IDC) from histopathology images. The repository combines image feature extraction, dataset assembly, and model training using both a decision tree and a neural network classifier.

This project is based on the Breast Histopathology Images dataset from Kaggle and follows the workflow described in the project paper included in the repository.

## Overview

Breast cancer remains one of the most common cancers in women worldwide, and IDC is the most common subtype. Manual pathology review can be subjective and time-consuming, so the goal of this project is to support diagnosis with a reproducible image-based classification pipeline.

The project extracts image-derived features such as:

- RGB values
- HSV values
- GLCM texture features
- Histogram statistics

These features are then combined into a single training dataset and used to train:

- a Decision Tree classifier
- a Multi-Layer Perceptron (HP Neural) classifier

## Project goals

- Build a reliable IDC prediction model from histology images
- Extract medically relevant visual features from stained tissue patches
- Compare the performance of multiple ML classifiers
- Provide a reusable workflow for image feature extraction and classification

## Dataset

The project uses the publicly available Breast Histopathology Images dataset from Kaggle.

The dataset contains histopathology image patches labeled as:

- 0: non-IDC / negative
- 1: IDC / positive

The repository includes code that samples a subset of the data, making it easier to work with a smaller and more manageable dataset.

## Workflow summary

1. Download or prepare the image dataset
2. Create a smaller sample subset
3. Extract RGB features
4. Extract GLCM features
5. Extract HSV features
6. Extract histogram-based features
7. Merge all feature tables into one dataset
8. Train and evaluate the models

## Repository structure

```text
.
├── README.md
├── feature_extraction.ipynb
├── LICENSE
├── paper_entry/
│   └── Team Machine Learning Dynamite 2.pdf
├── models/
│   ├── decision_tree.py
│   ├── HP_neural.py
│   ├── output/
│   └── output_mlp/
├── scripts/
│   ├── 1_download_kaggle.sh
│   ├── 2_create_sample.py
│   ├── 3_explore_image.py
│   ├── 4_extract_rgb_features.py
│   ├── 5_extracting_GLCM_features.py
│   ├── 6_extract_HSV_features.py
│   ├── 7_extract_histogram_based_features.py
│   └── 8_merge_features_dataset.py
└── data/
    ├── sample/
    └── features/
```

## Feature extraction

The image processing pipeline computes the following feature types:

### RGB features

- mean_r
- mean_g
- mean_b
- color_ratio

### GLCM texture features

- contrast
- correlation
- energy
- homogeneity

### HSV features

- hue
- saturation
- value

### Histogram features

- mean
- variance
- std_dev
- skewness

These are extracted from each histology patch and saved as CSV feature files in the data/features directory.

## Model training

The project trains two models on the merged feature dataset.

### Decision Tree

The script in models/decision_tree.py:

- loads the merged CSV dataset
- prepares X and y variables
- imputes missing values
- standardizes features
- builds a decision tree classifier
- performs train/test evaluation
- saves the best model to models/output/decision_tree_best.joblib

### HP Neural model

The script in models/HP_neural.py:

- loads the merged CSV dataset
- runs a grid search over neural network hyperparameters
- evaluates model performance using ROC-AUC and classification metrics
- saves the best model to models/output_mlp/mlp_best.joblib

## Expected results

The project paper reports the following outcomes:

- Decision Tree accuracy: about 85%
- Decision Tree false positive rate: 16.7%
- Decision Tree false negative rate: 17.5%
- HP Neural accuracy: about 89%
- HP Neural false positive rate: 16.1%
- HP Neural false negative rate: 18.2%

These results suggest that RGB and texture-based features can be useful for automated IDC detection and can form a strong basis for future model improvement.

## Setup

### Requirements

Python 3.9+ is recommended.

Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib opencv-python scikit-image tqdm scipy joblib
```

## Running the project

### 1. Create a sample subset of the dataset

```bash
python scripts/2_create_sample.py
```

This creates a smaller dataset under data/sample with class folders 0 and 1.

### 2. Extract image features

Run the feature extraction scripts in order:

```bash
python scripts/4_extract_rgb_features.py
python scripts/5_extracting_GLCM_features.py
python scripts/6_extract_HSV_features.py
python scripts/7_extract_histogram_based_features.py
```

### 3. Merge feature files

```bash
python scripts/8_merge_features_dataset.py
```

This creates a combined file such as:

- data/features/Merged_Features.csv

### 4. Train the models

Decision Tree:

```bash
python models/decision_tree.py
```

HP Neural model:

```bash
python models/HP_neural.py
```

## Notes on data layout

The feature scripts support both common layout patterns:

- flat folders: data/sample/0/... and data/sample/1/...
- nested patient folders: data/sample/<patient_id>/0/... and data/sample/<patient_id>/1/...

This makes the pipeline flexible for different dataset arrangements.

## Credits

This project was developed as part of the Curiosity Cup 2025 and the team is listed as:

- Omoegbemwen Aigbe (ME)
- Chi Wei Bernadette Chan
- Richard Malatesta
- Yazid Shuaibu

## License

This project is distributed under the MIT License. See the LICENSE file for details.

## Conclusion

This repository demonstrates a practical image-analysis ML workflow for IDC detection using handcrafted features from pathology images. While the models are relatively lightweight and interpretable, they show strong potential for clinical decision support when combined with more advanced feature engineering or deep learning approaches.
