# src/naive_bayes.py
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score

def main(data_path: str, model_path: str, report_path: str, test_size: float = 0.2, seed: int = 42):
    print(f"📥 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    # Expect columns exactly: label, text
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print("🧱 Building TF–IDF + ComplementNB...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = ComplementNB()
    model.fit(X_train_vec, y_train)

    # quick eval on the holdout
    X_test_vec = vectorizer.transform(X_test)
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, pos_label="spam")

    print(f"✅ Trained baseline NB. Acc={acc:.4f} | F1={f1:.4f}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "meta": {
            "algorithm": "ComplementNB",
            "vocab_size": len(vectorizer.vocabulary_),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy_holdout": float(acc),
            "f1_holdout": float(f1),
        },
    }
    joblib.dump(bundle, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"✅ Trained baseline NB.\n")
        f.write(f"Vocab={len(vectorizer.vocabulary_)}\n")
        f.write(f"Acc={acc:.4f} | F1={f1:.4f}\n")

    print(f"📦 Saved model -> {model_path}")
    print(f"📝 Saved report -> {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/spam.csv")
    ap.add_argument("--model", default="models/baseline_nb.joblib")
    ap.add_argument("--report", default="reports/baseline_nb.txt")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.data, args.model, args.report, args.test_size, args.seed)
