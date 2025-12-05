# src/preprocess.py
from __future__ import annotations
import sys
import csv

csv.field_size_limit(min(sys.maxsize, 10_000_000))


"""
Preprocessing utilities for the spam detector project.

- clean_text: تنظيف رسالة واحدة (إزالة الروابط/HTML + توحيد الحروف).
- load_and_clean: تحميل ملف CSV وتنظيفه وإرجاع DataFrame جاهزة للتدريب.
"""

import re
from typing import Final

import pandas as pd

# Simple regex-based cleaning (no external downloads needed)
URL_RE: Final = re.compile(r"https?://\S+|www\.\S+")
HTML_RE: Final = re.compile(r"<[^>]+>")
TOK_RE: Final = re.compile(r"[A-Za-z0-9]+")  # keep letters+digits tokens only


def clean_text(s: str) -> str:
    """
    Normalize and lightly clean a single message.

    Steps:
    - Ensure input is a string.
    - Lowercase.
    - Remove URLs and HTML tags.
    - Keep only alphanumeric tokens.
    """
    if not isinstance(s, str):
        s = str(s)

    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)

    tokens = TOK_RE.findall(s)
    return " ".join(tokens)


def load_and_clean(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """
    Load a CSV file and return a cleaned DataFrame with two columns:
    [text_col, label_col].

    - Reads the CSV using Python's engine (auto-detect separator).
    - Validates required columns.
    - Normalizes label column (string + lowercase + strip).
    - Applies `clean_text` to the text column.
    - Drops empty messages and duplicates.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    text_col : str
        Name of the text column in the CSV.
    label_col : str
        Name of the label column in the CSV.
    encoding : str
        File encoding (default: latin-1 for common SMS datasets).

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns [text_col, label_col] after cleaning.
    """
    df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns `{text_col}` and `{label_col}` in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize labels and text
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[text_col] = df[text_col].astype(str).map(clean_text)

    # Drop empty messages & duplicates (helps model quality)
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col, label_col])

    return df[[text_col, label_col]].reset_index(drop=True)
