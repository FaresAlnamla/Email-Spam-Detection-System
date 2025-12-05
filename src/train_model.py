# src/train_logreg.py
from __future__ import annotations
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

"""
Training script for the Logistic Regression spam detector.

Usage (example):
    python -m src.train_logreg --data data/raw/spam.csv --text-col text --label-col label
"""

from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def try_load_dataframe(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Load and clean the dataset.

    It first tries to use `src.preprocess.load_and_clean`. If anything fails,
    it falls back to a minimal read using pandas.
    """
    try:
        from src.preprocess import load_and_clean

        df = load_and_clean(csv_path, text_col=text_col, label_col=label_col)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to use load_and_clean: {exc}")
        print("[WARN] Falling back to raw pandas.read_csv(...)")

        df = pd.read_csv(csv_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"Columns not found. Expected text_col='{text_col}', "
                f"label_col='{label_col}'. Available: {list(df.columns)}"
            )

        df = df[[text_col, label_col]].dropna()
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)

    return df


def build_pipeline() -> Pipeline:
    """
    Build a TF–IDF + Linear SVM (calibrated) pipeline.

    - Word n-grams 1–2: تمثيل سياقي معقول.
    - LinearSVC غالباً يعطي فاصل أفضل بين spam/ham من LR في نصوص قصيرة.
    - CalibratedClassifierCV عشان نحصل على احتمالات (predict_proba).
    """
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        lowercase=True,
        sublinear_tf=True,
    )

    base = LinearSVC(

        class_weight="balanced",
        dual="auto",   # أو dual=False، الاثنين شغّالين على الأغلب
    )

    clf = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=3,
    )

    return Pipeline([
        ("tfidf", vec),
        ("clf", clf),
    ])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    ap = argparse.ArgumentParser(description="Train Logistic Regression spam model")
    ap.add_argument(
        "--data",
        required=True,
        help="CSV file with text/label columns",
    )
    ap.add_argument(
        "--text-col",
        default="text",
        help="Name of the text column (default: text)",
    )
    ap.add_argument(
        "--label-col",
        default="label",
        help="Name of the label column (default: label)",
    )
    ap.add_argument(
        "--out",
        default="models/bundle_lr.joblib",
        help="Output path for the model bundle (default: models/bundle_lr.joblib)",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test set ratio (default: 0.20)",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    df = try_load_dataframe(args.data, args.text_col, args.label_col)
    X = df[args.text_col].values
    y = df[args.label_col].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    # Quick, readable evaluation report
    y_pred = pipe.predict(X_te)
    print("\n=== Evaluation on hold-out set ===")
    print(classification_report(y_te, y_pred, digits=3))

    # Save bundle compatible with API + CLI (vectorizer + model + meta)
    Path("models").mkdir(exist_ok=True)

    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    bundle = {
        "vectorizer": vec,
        "model": clf,
        "meta": {
            "model_type": "linear_svm",
            "features": "tfidf_word_1-2",
            "calibrated": True,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "version": "svm_sprint1",
        },
    }

    joblib.dump(bundle, args.out)
    print(f"[OK] saved \u2192 {args.out}")


if __name__ == "__main__":
    main()
