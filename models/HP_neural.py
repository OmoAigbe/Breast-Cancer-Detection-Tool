#!/usr/bin/env python3


import argparse
from pathlib import Path
import joblib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

def load_features(path: Path):
    df = pd.read_csv(path)
    required = {"patient_id", "path", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: need {required}")
    return df

def prepare_X_y(df: pd.DataFrame):
    X = df.drop(columns=["patient_id", "path", "target"])
    y = df["target"].astype(int)
    # Keep numeric only
    X = X.select_dtypes(include=[np.number])
    return X, y

def build_pipeline(random_state: int):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(random_state=random_state, max_iter=500))
    ])
    return pipeline

def plot_and_save_roc(y_train, y_train_proba, y_test, y_test_proba, out_dir: Path, prefix="mlp"):
    # ROC AUC
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_proba)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_test_proba)
    auc_tr = roc_auc_score(y_train, y_train_proba)
    auc_te = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_tr:.3f})", color="blue")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - Train")
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(fpr_te, tpr_te, label=f"Test ROC (AUC={auc_te:.3f})", color="orange")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - Test")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc.png")
    plt.close()

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(Path(args.features))
    X, y = prepare_X_y(df)
    print("Feature matrix shape:", X.shape, "Positive class fraction:", y.mean())

    # train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_seed
    )

    pipeline = build_pipeline(args.random_seed)

    # Grid search — tune hidden_layer_sizes and alpha and learning_rate_init
    param_grid = {
        "clf__hidden_layer_sizes": [(50,), (100,), (100,50)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 1e-4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
    gs = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=args.n_jobs, verbose=2)
    print("Starting GridSearchCV ...")
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    # Evaluate on test set
    y_test_pred = best.predict(X_test)
    y_test_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
    print("Test classification report:")
    print(classification_report(y_test, y_test_pred))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    if y_test_proba is not None:
        print("Test ROC AUC:", roc_auc_score(y_test, y_test_proba))

    # Evaluate on training set too (to monitor overfitting)
    y_train_pred = best.predict(X_train)
    y_train_proba = best.predict_proba(X_train)[:, 1] if hasattr(best, "predict_proba") else None
    print("Train classification report (subset):")
    print(classification_report(y_train, y_train_pred))
    if y_train_proba is not None:
        print("Train ROC AUC:", roc_auc_score(y_train, y_train_proba))

    # Confusion matrix and ROC plots saved
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Test")
    plt.savefig(out_dir / "confusion_matrix_test.png")
    plt.close()

    if y_train_proba is not None and y_test_proba is not None:
        plot_and_save_roc(y_train, y_train_proba, y_test, y_test_proba, out_dir, prefix="mlp")

    # Save best model
    joblib.dump(gs.best_estimator_, out_dir / "mlp_best.joblib")
    print("Saved best model to", out_dir / "mlp_best.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features/Merged_Features.csv")
    parser.add_argument("--out-dir", type=str, default="models/output_mlp")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    main(args)