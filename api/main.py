# api/main.py
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from io import StringIO, BytesIO

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .schemas import (
    PredictIn,
    PredictOut,
    BatchPredictIn,
    BatchPredictOut,
)

# ---------------------------
# Settings (env-driven)
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/bundle_svm.joblib")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
TRUSTED_HOSTS = [h.strip() for h in os.getenv("TRUSTED_HOSTS", "*").split(",")]

# ---------------------------
# Profiles (single source of truth)
# ---------------------------

# تعريف الفئات الرسمية اللي الواجهة لازم تستخدم الـ key تبعها
PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "default": {
        "label": "Default (balanced)",
        "threshold": 0.55,
        "description": "General-purpose profile with balanced spam capture vs. false positives.",
    },
    "telco": {
        "label": "Telecom / SMS filtering",
        "threshold": 0.55,
        "description": "Suitable for telecoms: slightly conservative to avoid blocking real OTPs / alerts.",
    },
    "bank": {
        "label": "Bank / Financial (very strict)",
        "threshold": 0.65,
        "description": "Very strict: only mark as spam when highly confident. Protects legitimate financial messages.",
    },
    "marketing": {
        "label": "Email marketing / newsletters (aggressive)",
        "threshold": 0.45,  # كانت 0.40
        "description": "Aggressive: catch promotional & marketing spam even if it risks some false positives.",
    },
    "aggressive": {
        "label": "Aggressive (max spam capture)",
        "threshold": 0.45,  # كانت 0.40
        "description": "Maximize spam capture. Use when you prefer to over-block rather than miss spam.",
    },
    "balanced": {
        "label": "Balanced (general use)",
        "threshold": 0.55,  # أهم تعديل: نفس default
        "description": "Balanced behavior similar to 'default', suitable for most use cases.",
    },
    "conservative": {
        "label": "Conservative (protect REAL messages)",
        "threshold": 0.60,
        "description": "More conservative: only flag spam when probability is high.",
    },
}

# thresholds جاهزة من التعريفات
PROFILE_THRESHOLDS: Dict[str, float] = {
    key: value["threshold"] for key, value in PROFILE_DEFINITIONS.items()
}

# aliases عشان تظل المفاتيح القديمة شغالة لو استُخدمت
PROFILE_ALIASES: Dict[str, str] = {
    "financial": "bank",
    "telecom": "telco",
    "email_marketing": "marketing",
    "newsletter": "marketing",
}

# لليبلز القديمة لو بدك تستخدمها في مكان ما
PROFILE_OPTIONS = {v["label"]: k for k, v in PROFILE_DEFINITIONS.items()}

# System Type (defines the default filtering policy)
SYSTEM_PROFILE = os.getenv("SYSTEM_PROFILE", "default").lower()


def compute_spam_threshold() -> float:
    """
    Decide spam threshold based on:
    1) SPAM_THRESHOLD env if موجود وصحيح
    2) أو SYSTEM_PROFILE من التعريفات
    """
    env_override = os.getenv("SPAM_THRESHOLD")
    if env_override is not None:
        try:
            return float(env_override)
        except ValueError:
            logger = logging.getLogger("spam-detector.api")
            logger.warning("Invalid SPAM_THRESHOLD value: %s", env_override)

    profile_key = PROFILE_ALIASES.get(SYSTEM_PROFILE, SYSTEM_PROFILE)
    return PROFILE_THRESHOLDS.get(profile_key, PROFILE_THRESHOLDS["default"])


# القيمة الأساسية لو ما نُمرِّر profile في الريكوست
SPAM_THRESHOLD = compute_spam_threshold()


def resolve_threshold(profile: str | None) -> float:
    """
    Resolve the spam threshold based on an optional profile name.
    If no valid profile is provided, fall back to the global SPAM_THRESHOLD.
    """
    if profile:
        key = profile.lower()
        key = PROFILE_ALIASES.get(key, key)
        if key in PROFILE_THRESHOLDS:
            return PROFILE_THRESHOLDS[key]
    return SPAM_THRESHOLD


# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("spam-detector.api")


# ---------------------------
# Utilities
# ---------------------------
def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    """
    Compute SHA256 hash of a file.

    Used for debugging to ensure we know exactly which model file is loaded.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_bundle(path: str) -> Dict[str, Any]:
    """
    Load a model bundle from disk.

    Expected format:
        {'vectorizer': ..., 'model': ..., 'meta': {...}}
    """
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "vectorizer" not in obj or "model" not in obj:
        raise RuntimeError("Invalid model bundle: expected {'vectorizer', 'model'}.")
    return obj


# ---------------------------
# App & Middlewares
# ---------------------------
app = FastAPI(
    title="Spam Detector API",
    version="1.0.0",
    description="Character-level TF-IDF + Logistic Regression spam classifier",
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Simple middleware to attach a request ID to each request."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or f"req_{int(time.time() * 1000)}"
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS (tighten ALLOW_ORIGINS in production, e.g. "https://your.site")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",")],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# (Optional) TrustedHostMiddleware — only if you deploy behind fixed hostnames
# from starlette.middleware.trustedhost import TrustedHostMiddleware
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

# ---------------------------
# Load model once at startup
# ---------------------------
MODEL_BUNDLE: Dict[str, Any] | None = None
MODEL_META: Dict[str, Any] = {}
MODEL_FILE_INFO: Dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    global MODEL_BUNDLE, MODEL_META, MODEL_FILE_INFO

    logger.info("Loading model bundle: %s", MODEL_PATH)
    t0 = time.time()
    bundle = load_bundle(MODEL_PATH)
    dt = time.time() - t0

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    meta = bundle.get("meta", {})

    # Capture file info
    p = Path(MODEL_PATH)
    MODEL_FILE_INFO = {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "mtime": p.stat().st_mtime if p.exists() else None,
        "sha256": file_sha256(str(p)) if p.exists() else None,
        "load_seconds": round(dt, 4),
    }

    MODEL_META = {
        "vectorizer": vectorizer.__class__.__name__,
        "classifier": model.__class__.__name__,
        "bundle_meta": meta,
    }
    MODEL_BUNDLE = bundle

    logger.info(
        "Model loaded in %.3fs | vec=%s | clf=%s",
        dt,
        MODEL_META["vectorizer"],
        MODEL_META["classifier"],
    )


# ---------------------------
# Error handling
# ---------------------------
@app.exception_handler(Exception)
async def _unhandled_ex(request: Request, exc: Exception):
    """Global fallback error handler to avoid leaking internal traces."""
    rid = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled error | request_id=%s", rid)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": rid},
    )


# ---------------------------
# Routes
# ---------------------------

@app.get("/profiles")
def list_profiles() -> Dict[str, Any]:
    """
    إرجاع كل الفئات المتوفّرة للواجهة.

    هذا الإندبوينت مفيد جداً للـ UI:
    - يعرض اللابل
    - الثريشولد
    - الوصف
    - والبروفايل الافتراضي الحالي
    """
    return {
        "system_profile": SYSTEM_PROFILE,
        "default_threshold": SPAM_THRESHOLD,
        "profiles": [
            {
                "key": key,
                "label": data["label"],
                "threshold": data["threshold"],
                "description": data["description"],
            }
            for key, data in PROFILE_DEFINITIONS.items()
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Health-check endpoint (ممكن تستخدمه داخلياً فقط، والـ UI انت شلّه من الواجهة).

    Returns:
    - whether the model is loaded
    - model file info (size, hash, etc.)
    - basic model metadata
    - current config (thresholds, profiles, limits)
    """
    ok = MODEL_BUNDLE is not None
    return {
        "ok": ok,
        "model_loaded": ok,
        "model_file": MODEL_FILE_INFO,
        "model_meta": MODEL_META,
        "config": {
            "system_profile": SYSTEM_PROFILE,
            "spam_threshold": SPAM_THRESHOLD,
            "max_batch": MAX_BATCH,
        },
    }


@app.post("/predict", response_model=PredictOut)
def predict(
    payload: PredictIn,
    request: Request,
    profile: str | None = Query(
        default=None,
        description="Optional classification profile key (e.g. 'default', 'telco', 'bank').",
    ),
) -> PredictOut:
    """
    Classify a single message as spam or ham, with optional spam probability.

    You can override the threshold per-request via the `profile` query param.
    """
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty 'text'")

    vec = MODEL_BUNDLE["vectorizer"]
    clf = MODEL_BUNDLE["model"]

    X = vec.transform([text])

    threshold = resolve_threshold(profile)
    proba_spam: float | None = None

    if hasattr(clf, "predict_proba"):
        import numpy as np

        proba = clf.predict_proba(X)
        spam_idx = np.where(clf.classes_.astype(str) == "spam")[0]
        spam_idx = spam_idx[0] if len(spam_idx) else (1 if proba.shape[1] == 2 else 0)
        proba_spam = float(proba[:, spam_idx][0])

        # Threshold-based label
        label = "spam" if proba_spam >= threshold else "ham"
    else:
        # Fallback if predict_proba is not available
        pred = clf.predict(X)[0]
        label = str(pred)

    return PredictOut(
        text=text,
        pred=label,
        proba_spam=proba_spam,
        request_id=getattr(request.state, "request_id", None),
    )


@app.post("/batch", response_model=BatchPredictOut)
def batch(
    payload: BatchPredictIn,
    request: Request,
    profile: str | None = Query(
        default=None,
        description="Optional classification profile key (e.g. 'default', 'telco', 'bank').",
    ),
) -> BatchPredictOut:
    """
    Classify a batch of messages in a single request.

    Enforces:
    - non-empty list of texts
    - upper limit on batch size via MAX_BATCH env var
    - optional per-request profile to choose threshold
    """
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not payload.texts:
        raise HTTPException(status_code=422, detail="No texts provided")
    if len(payload.texts) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Batch too large (>{MAX_BATCH})")

    vec = MODEL_BUNDLE["vectorizer"]
    clf = MODEL_BUNDLE["model"]

    texts: List[str] = [str(t or "").strip() for t in payload.texts]
    X = vec.transform(texts)

    # We work entirely based on probabilities, not on clf.predict
    preds: List[str] = ["ham"] * len(texts)
    probs: List[float | None] = [None] * len(texts)

    threshold = resolve_threshold(profile)

    if hasattr(clf, "predict_proba"):
        import numpy as np

        proba = clf.predict_proba(X)
        spam_idx = (clf.classes_.astype(str) == "spam").nonzero()[0]
        spam_idx = spam_idx[0] if len(spam_idx) else (1 if proba.shape[1] == 2 else 0)
        spam_probs = proba[:, spam_idx].tolist()

        for i, p in enumerate(spam_probs):
            probs[i] = float(p)
            preds[i] = "spam" if p >= threshold else "ham"
    else:
        # Fallback if probabilities are not available
        raw_preds = clf.predict(X)
        preds = [str(p) for p in raw_preds]

    items = [
        {"text": t, "pred": pred_label, "proba_spam": (float(pr) if pr is not None else None)}
        for t, pred_label, pr in zip(texts, preds, probs)
    ]

    return BatchPredictOut(
        size=len(items),
        items=items,
        request_id=getattr(request.state, "request_id", None),
    )


@app.post("/file-predict")
async def file_predict(
    file: UploadFile = File(...),
    profile: str | None = Query(
        default=None,
        description="Optional classification profile key (e.g. 'default', 'telco', 'bank').",
    ),
):
    """
    Receive a file (CSV / XLSX / TXT) containing messages, classify them,
    and return a new downloadable CSV file named:
        prediction_<original_name>.csv

    Conditions:
    - The file must contain a 'text' column (or in the case of .txt: each line is a message).
    - You can optionally override the spam threshold by passing ?profile=<key>.
    """

    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name")

    original_name = file.filename
    ext = Path(original_name).suffix.lower()

    # Read the file content
    content = await file.read()

    # ---------------------------------
    # 1) Read the file based on its extension
    # ---------------------------------
    try:
        if ext == ".csv":
            try:
                # Normal attempt to read a CSV file (well-structured files with columns like text, label, ...)
                df = pd.read_csv(StringIO(content.decode("utf-8")))
            except Exception:
                # If it fails (e.g., a single-column file that contains commas without quotes)
                # Treat the file as plain text: each line = one message
                lines = content.decode("utf-8").splitlines()
                lines = [ln.strip() for ln in lines if ln.strip()]
                df = pd.DataFrame({"text": lines})

        elif ext in {".xlsx", ".xls"}:
            # Requires openpyxl
            df = pd.read_excel(BytesIO(content))

        elif ext == ".txt":
            # Each line represents a message
            lines = content.decode("utf-8").splitlines()
            lines = [ln.strip() for ln in lines if ln.strip()]
            df = pd.DataFrame({"text": lines})

        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {ext}. Use .csv, .xlsx, or .txt",
            )

    except HTTPException:
        # Re-raise the errors that we originally created
        raise
    except Exception as exc:
        # Any general read error is converted to a 400 response
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    if "text" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Input must contain a 'text' column (or lines for .txt).",
        )

    # ---------------------------------
    # 2) Classification
    # ---------------------------------
    vec = MODEL_BUNDLE["vectorizer"]
    clf = MODEL_BUNDLE["model"]

    texts = df["text"].astype(str).tolist()
    X = vec.transform(texts)

    threshold = resolve_threshold(profile)

    spam_probs = None
    preds: list[str] = []

    if hasattr(clf, "predict_proba"):
        import numpy as np

        proba = clf.predict_proba(X)
        spam_idx = (clf.classes_.astype(str) == "spam").nonzero()[0]
        spam_idx = spam_idx[0] if len(spam_idx) else (1 if proba.shape[1] == 2 else 0)

        spam_probs = proba[:, spam_idx]
        for p in spam_probs:
            preds.append("spam" if p >= threshold else "ham")
    else:
        raw_preds = clf.predict(X)
        preds = [str(p) for p in raw_preds]

    df["pred"] = preds
    if spam_probs is not None:
        df["proba_spam"] = spam_probs

    # ---------------------------------
    # 3) Prepare the output file
    # Always return a CSV file named prediction_<original_stem>.csv
    # ---------------------------------
    out_name = f"prediction_{Path(original_name).stem}.csv"

    out_buf = StringIO()
    df.to_csv(out_buf, index=False)
    out_buf.seek(0)

    return StreamingResponse(
        out_buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{out_name}\"'},
    )
