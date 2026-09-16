#!/usr/bin/env python3

import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

def load_features(path: Path):
    df = pd.read_csv(path)
    # basic checks
    required = {"patient_id", "path", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}: need {required}")
    return df

def prepare_X_y(df: pd.DataFrame):
    X = df.drop(columns=["patient_id", "path", "target"])
    y = df["target"].astype(int)
    # keep numeric columns only (drop anything non-numeric)
    X = X.select_dtypes(include=[np.number])
    return X, y

def build_pipeline():
    # For tree models scaling is optional, but keep pipeline so we can swap models later.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=0))
    ])
    return pipeline

def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(Path(args.features))
    X, y = prepare_X_y(df)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_seed
    )

    pipeline = build_pipeline()

    # quick baseline: default decision tree
    pipeline.set_params(clf__random_state=args.random_seed)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # metrics
    print("Classification report (test):")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC (test): {roc_auc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Decision Tree - Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test)")
        plt.legend(loc="lower right")
        plt.savefig(out_dir / "roc_curve.png")
        plt.close()

    # Cross-validation score (stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-validated ROC AUC (5-fold): mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")

    # Optional: quick grid search for tree depth
    param_grid = {
        "clf__max_depth": [3, 5, 7, 9, None],
        "clf__min_samples_leaf": [1, 2, 5, 10]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=args.n_jobs, verbose=1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    best = gs.best_estimator_
    # evaluate best
    y_pred_best = best.predict(X_test)
    y_proba_best = best.predict_proba(X_test)[:, 1]
    print("Classification report (best):")
    print(classification_report(y_test, y_pred_best))
    print(f"ROC AUC (best test): {roc_auc_score(y_test, y_proba_best):.4f}")

    # persist best model
    joblib.dump(best, out_dir / "decision_tree_best.joblib")
    print("Saved best model to:", out_dir / "decision_tree_best.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Tree on merged features")
    parser.add_argument("--features", type=str, default="data/features/Merged_Features.csv")
    parser.add_argument("--out-dir", type=str, default="models/output")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    run(args)