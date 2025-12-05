# src/evaluate.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

def _plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        ylabel='True label', xlabel='Predicted label',
        title='Confusion Matrix'
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_roc(y_true_bin, scores, out_path):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc = roc_auc_score(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc

def _plot_pr(y_true_bin, scores, out_path):
    precisions, recalls, _ = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recalls, precisions, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ap

def evaluate(model_path: str, data_path: str, report_path: str, test_size: float = 0.2, seed: int = 42):
    print(f"🔍 Loading model from {model_path}")
    obj = joblib.load(model_path)

    if not isinstance(obj, dict) or "vectorizer" not in obj or "model" not in obj:
        raise ValueError("Model file must be a dict with keys: vectorizer, model (train with src/naive_bayes.py).")

    vectorizer = obj["vectorizer"]
    model = obj["model"]

    # Optional: show meta
    if "meta" in obj:
        print("ℹ️ Model metadata:")
        for k, v in obj["meta"].items():
            print(f"   {k}: {v}")

    print(f"📥 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Vectorize test and predict
    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    # Probability/scores (for curves & ranking misclassifications)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_vec)
        # find spam column
        spam_idx = np.where(model.classes_.astype(str) == "spam")[0]
        spam_idx = spam_idx[0] if len(spam_idx) else (1 if proba.shape[1] == 2 else 0)
        scores = proba[:, spam_idx]
    elif hasattr(model, "decision_function"):
        dfun = model.decision_function(X_test_vec)
        scores = dfun if dfun.ndim == 1 else dfun[:, -1]
    else:
        scores = None

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label="spam")
    rec = recall_score(y_test, preds, pos_label="spam")
    f1  = f1_score(y_test, preds, pos_label="spam")
    print(f"\n✅ Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}\n")

    # Paths
    rp = Path(report_path); rp.parent.mkdir(parents=True, exist_ok=True)
    cm_path  = str(rp).replace(".md", "_confusion.png")
    roc_path = str(rp).replace(".md", "_roc.png")
    pr_path  = str(rp).replace(".md", "_pr.png")
    mis_path = str(rp).replace(".md", "_misclassified.csv")
    clf_path = str(rp).replace(".md", "_classification_report.txt")

    # Confusion matrix
    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, preds, labels=labels)
    _plot_confusion_matrix(cm, labels, cm_path)

    # Misclassified
    mis_mask = preds != y_test
    pd.DataFrame({
        "text": X_test[mis_mask],
        "true": y_test[mis_mask],
        "pred": preds[mis_mask],
        "score_or_proba": (scores[mis_mask] if scores is not None else np.nan)
    }).to_csv(mis_path, index=False)

    # Classification report
    with open(clf_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds, digits=4))

    # Curves
    auc, ap = None, None
    if scores is not None:
        y_true_bin = (y_test.str.lower() == "spam").astype(int).values
        auc = _plot_roc(y_true_bin, scores, roc_path)
        ap  = _plot_pr(y_true_bin, scores, pr_path)

    # Markdown summary
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Naive Bayes Evaluation\n\n")
        f.write(f"**Accuracy:** {acc:.4f}  \n")
        f.write(f"**Precision (spam):** {prec:.4f}  \n")
        f.write(f"**Recall (spam):** {rec:.4f}  \n")
        f.write(f"**F1 (spam):** {f1:.4f}  \n\n")
        f.write(f"![Confusion Matrix]({Path(cm_path).name})\n\n")
        if auc is not None and ap is not None:
            f.write(f"**AUC-ROC:** {auc:.4f}  \n")
            f.write(f"**Average Precision (PR AUC):** {ap:.4f}  \n\n")
            f.write(f"![ROC Curve]({Path(roc_path).name})\n\n")
            f.write(f"![Precision–Recall Curve]({Path(pr_path).name})\n\n")
        f.write("**Artifacts:**\n\n")
        f.write(f"- `{Path(mis_path).name}` — misclassified samples with scores\n")
        f.write(f"- `{Path(clf_path).name}` — full classification report\n")

    print(f"📊 Report saved -> {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default="reports/baseline_nb_eval.md")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    evaluate(args.model, args.data, args.report, args.test_size, args.seed)
