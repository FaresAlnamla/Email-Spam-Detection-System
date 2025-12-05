# src/batch_predict.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "ISO-8859-1"]

def _detect_text_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    cols_lower = {c.lower(): c for c in df.columns}
    for k in ["text", "message", "sms", "content", "body"]:
        if k in cols_lower:
            return cols_lower[k]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError(
        f"Could not detect a text column. Pass --text-col explicitly. Columns={list(df.columns)}"
    )

def _iter_csv(path: str, chunksize: int):
    last_err = None
    for enc in FALLBACK_ENCODINGS:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, chunksize=chunksize)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("Failed to open CSV with fallback encodings")

def _write_chunk(out_path: str, df: pd.DataFrame, wrote_header: bool) -> bool:
    df.to_csv(out_path, mode="a", index=False, header=not wrote_header, encoding="utf-8")
    return True

def main(model_path: str, input_path: str, output_path: str, text_col: str | None,
        chunksize: int, threshold: float | None):
    # 1) Load model bundle
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "vectorizer" not in obj or "model" not in obj:
        raise ValueError("Model must be a dict with keys: vectorizer, model (train with src/naive_bayes.py).")
    vectorizer = obj["vectorizer"]
    model = obj["model"]

    # 2) Prepare output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # truncate file if exists
    open(output_path, "w", encoding="utf-8").close()
    wrote_header = False

    # 3) Stream input
    reader = _iter_csv(input_path, chunksize)
    for i, chunk in enumerate(reader):
        tcol = _detect_text_col(chunk, text_col) if i == 0 else tcol

        texts = chunk[tcol].astype(str).tolist()
        X = vectorizer.transform(texts)

        pred = model.predict(X)

        # Proba for spam (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            spam_idx = np.where(model.classes_.astype(str) == "spam")[0]
            spam_idx = spam_idx[0] if len(spam_idx) else (1 if proba.shape[1] == 2 else 0)
            proba_spam = proba[:, spam_idx]
        elif hasattr(model, "decision_function"):
            dfun = model.decision_function(X)
            if dfun.ndim == 1:
                proba_spam = 1 / (1 + np.exp(-dfun))
            else:
                scores = dfun[:, -1]
                proba_spam = 1 / (1 + np.exp(-scores))
        else:
            proba_spam = np.full(shape=(len(texts),), fill_value=np.nan, dtype=float)

        out_chunk = chunk.copy()
        out_chunk["pred"] = pred
        out_chunk["proba_spam"] = proba_spam

        if threshold is not None:
            thr = float(threshold)
            out_chunk["pred_thresholded"] = np.where(out_chunk["proba_spam"] >= thr, "spam", "ham")

        wrote_header = _write_chunk(output_path, out_chunk, wrote_header)

    print(f"✅ Done. Wrote: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch score a CSV file with spam detector.")
    ap.add_argument("--model", required=True, help="Path to joblib bundle (vectorizer + model).")
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--text-col", default=None, help="Name of text column (default: auto-detect).")
    ap.add_argument("--chunksize", type=int, default=5000, help="Rows per chunk to process.")
    ap.add_argument("--threshold", type=float, default=None, help="Optional proba threshold for 'pred_thresholded'.")
    args = ap.parse_args()
    main(args.model, args.input, args.output, args.text_col, args.chunksize, args.threshold)
