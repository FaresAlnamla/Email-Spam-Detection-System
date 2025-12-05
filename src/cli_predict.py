# src/cli_predict.py
import argparse
import os
import joblib

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/bundle_lr.joblib")

def load_bundle(model_path: str):
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "vectorizer" not in obj or "model" not in obj:
        raise ValueError("Model file must be a dict with keys: vectorizer, model.")
    return obj["vectorizer"], obj["model"], obj.get("meta", {})

def predict_text(model_path: str, text: str):
    vec, clf, meta = load_bundle(model_path)
    X = vec.transform([text])
    pred = clf.predict(X)[0]

    prob = None
    if hasattr(clf, "predict_proba"):
        prob = float(max(clf.predict_proba(X)[0]))
    return str(pred), prob, meta

def main():
    ap = argparse.ArgumentParser(description="CLI spam detector")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model bundle (default: {DEFAULT_MODEL_PATH})"
    )
    ap.add_argument(
        "--text",
        required=True,
        help="Message to classify"
    )
    args = ap.parse_args()

    pred, prob, meta = predict_text(args.model, args.text)

    print("\n=== Spam Detector (CLI) ===")
    print(f"Model meta: {meta}")
    print(f"Text       : {args.text}")
    if prob is not None:
        print(f"Prediction : {pred} (prob={prob:.3f})")
    else:
        print(f"Prediction : {pred}")

if __name__ == "__main__":
    main()
