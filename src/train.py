
#!/usr/bin/env python3
"""
Baseline sentiment classifier (TF-IDF + LinearSVC) with simple explainability hooks.

Why LinearSVC? Fast, strong baseline for sparse TF-IDF features.
Why char n-grams? Robust on misspellings/subwords (works well on IMDB).
"""

from __future__ import annotations
import argparse, os, json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

# ---------- data loading ----------

def try_load_imdb() -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[None, None]:
    """Load the IMDB dataset via `datasets`. Returns (train_df, test_df) or (None, None) if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("imdb")
        return pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])
    except Exception as e:
        print("[WARN] Could not load IMDB via datasets:", e)
        return None, None

def load_csv(csv_path: str, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CSV with columns: text, label (0/1 or neg/pos). Returns stratified train/test."""
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns (label=0/1 or 'pos'/'neg').")

    df = df[["text", "label"]].dropna()
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
        .replace({"pos": 1, "positive": 1, "neg": 0, "negative": 0})
    )
    df = df[df["label"].isin(["0", "1", 0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if len(df) < 3 or df["label"].nunique() < 2:
        raise ValueError("Not enough clean, labeled rows across both classes after filtering.")
    print("[INFO] Label counts:", df["label"].value_counts().to_dict())

    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# ---------- text prep ----------

def basic_clean(s: str) -> str:
    """Very small normaliser: lowercase, collapse whitespace, strip."""
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("\n", " ").replace("\r", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()

# ---------- model ----------

def build_pipeline(
    analyzer: str = "char",
    ngram_min: int = 3,
    ngram_max: int = 5,
    max_features: int = 30000,
) -> Pipeline:
    """TF-IDF + LinearSVC (calibrated). Defaults to char 3–5 grams for IMDB."""
    vectorizer = TfidfVectorizer(
        preprocessor=basic_clean,
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        stop_words="english" if analyzer == "word" else None,
        sublinear_tf=True,
        max_features=max_features,
        min_df=2 if analyzer == "char" else 1,
    )
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=3)  # for predict_proba
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])

# ---------- viz/metrics ----------

def plot_confusion(y_true, y_pred, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["neg (0)", "pos (1)"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["neg (0)", "pos (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_metrics(y_true, y_pred, out_json: str) -> None:
    report = classification_report(y_true, y_pred, target_names=["neg (0)", "pos (1)"], output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": report,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Train a simple sentiment model (TF-IDF + LinearSVC).")
    p.add_argument("--dataset", choices=["imdb", "csv"], required=True, help="Which dataset to use.")
    p.add_argument("--csv_path", default="sample_data/sample_sentiment.csv", help="CSV path if --dataset csv.")
    p.add_argument("--analyzer", choices=["char", "word"], default="char", help="Feature units for TF-IDF.")
    p.add_argument("--ngram_min", type=int, default=3, help="Lower bound for n-gram length.")
    p.add_argument("--ngram_max", type=int, default=5, help="Upper bound for n-gram length.")
    p.add_argument("--max_features", type=int, default=30000, help="Max TF-IDF features.")
    p.add_argument("--model_dir", default="models", help="Where to save model.")
    p.add_argument("--metrics_dir", default="metrics", help="Where to save metrics/plots.")
    args = p.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # Load data
    if args.dataset == "imdb":
        train_df, test_df = try_load_imdb()
        if train_df is None:
            print("[INFO] Falling back to CSV...")
            train_df, test_df = load_csv(args.csv_path)
    else:
        train_df, test_df = load_csv(args.csv_path)

    print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Build & fit
    pipe = build_pipeline(
        analyzer=args.analyzer,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )
    pipe.fit(train_df["text"].tolist(), train_df["label"].tolist())

    # Evaluate
    y_pred = pipe.predict(test_df["text"].tolist())
    y_true = test_df["label"].tolist()

    # Save
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipe, model_path); print(f"[INFO] Saved model to {model_path}")

    cm_png = os.path.join(args.metrics_dir, "confusion_matrix.png")
    plot_confusion(y_true, y_pred, cm_png); print(f"[INFO] Saved confusion matrix to {cm_png}")

    metrics_json = os.path.join(args.metrics_dir, "metrics.json")
    save_metrics(y_true, y_pred, metrics_json); print(f"[INFO] Saved metrics to {metrics_json}")

    # brief console
    with open(metrics_json) as f:
        preview = f.read()
    print("[RESULTS] Sample metrics:")
    print(preview[:800], "...")

if __name__ == "__main__":
    main()

