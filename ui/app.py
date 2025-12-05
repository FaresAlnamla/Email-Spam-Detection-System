import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

# ==========================
# Basic config
# ==========================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
SITE_URL = os.getenv("SITE_URL", "https://email-spam-detection.app")

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================
# Advanced Custom CSS with Animations
# ==========================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141e30);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* How to Use Card */
    .how-to-use-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12), rgba(118, 75, 162, 0.12));
        backdrop-filter: blur(25px);
        border-radius: 24px;
        border: 2px solid rgba(102, 126, 234, 0.25);
        padding: 2.5rem;
        margin-bottom: 2.5rem;
        box-shadow: 
            0 12px 40px rgba(102, 126, 234, 0.15),
            inset 0 2px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .how-to-use-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 200% 100%;
        animation: gradientSlide 3s linear infinite;
    }
    
    @keyframes gradientSlide {
        0% { background-position: 0% 0%; }
        100% { background-position: 200% 0%; }
    }
    
    .how-to-use-card h3 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .how-to-use-card .step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .how-to-step {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .how-to-step:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .how-to-step .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .how-to-step h4 {
        color: #e5e7ff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .how-to-step p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }
    
.hero-section {
    position: relative;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 1.4rem 2rem 2rem;
    border-radius: 32px;
    margin-bottom: 1.2rem;
    overflow: hidden;
    box-shadow: 
        0 20px 60px rgba(102, 126, 234, 0.4),
        0 0 80px rgba(118, 75, 162, 0.3);
}

    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.25rem;
        margin-bottom: 1.5rem;
    }
    
    .logo-shield {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.6),
            0 0 0 4px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        transform: rotate(-5deg);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .logo-shield:hover {
        transform: rotate(0deg) scale(1.1);
        box-shadow: 
            0 20px 50px rgba(102, 126, 234, 0.8),
            0 0 0 6px rgba(255, 255, 255, 0.15);
    }
    
    .logo-shield::before {
        content: '';
        position: absolute;
        inset: 4px;
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
        border-radius: 20px;
    }
    
    .logo-shield::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .logo-shield .logo-text {
        position: relative;
        z-index: 1;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e5e7ff 0%, #fff 50%, #e5e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        filter: drop-shadow(0 2px 8px rgba(255, 255, 255, 0.3));
    }
    
    .neon-stat-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        backdrop-filter: blur(10px);
        padding: 1.75rem 1.25rem;
        border-radius: 20px;
        text-align: center;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .neon-stat-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.7;
    }
    
    .neon-stat-card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 
            0 20px 50px rgba(102, 126, 234, 0.5),
            0 0 40px rgba(118, 75, 162, 0.4);
    }
    
    .neon-stat-card:hover::before {
        opacity: 1;
        animation: borderGlow 2s linear infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .stat-value {
        font-size: 2.75rem;
        font-weight: 900;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.75);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 15px rgba(102, 126, 234, 0.4),
            0 0 30px rgba(118, 75, 162, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 
            0 8px 30px rgba(102, 126, 234, 0.6),
            0 0 50px rgba(118, 75, 162, 0.4);
    }

    /* Centered pill-style radio for tab switcher */
    .mode-switcher-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }

    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        display: inline-flex;
        justify-content: center;
        align-items: center;
        gap: 0.25rem;
    }
    
    .stRadio > div > label {
        background: transparent;
        padding: 0.6rem 1.9rem;
        border-radius: 999px;
        transition: all 0.25s ease;
        font-weight: 600;
        font-size: 0.98rem;
        color: #e5e7ff;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-1px);
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        color: white;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 
            inset 0 2px 10px rgba(0, 0, 0, 0.2),
            0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 3px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 3rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        opacity: 0.3;
        animation: float 15s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) translateX(0); }
        25% { transform: translateY(-100px) translateX(50px); }
        50% { transform: translateY(-50px) translateX(-50px); }
        75% { transform: translateY(-150px) translateX(25px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse-on-load {
        animation: pulse 2s ease-in-out 3;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    .stSuccess, .stWarning, .stError {
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left-width: 4px;
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        overflow: hidden;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
    }
    
    .stApp > header {
        display: none !important;
        height: 0px !important;
    }
    
    .stApp > header {
        background-color: transparent;
        height: 0px;
    }
    
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] > div {
        display: none !important;
    }
    
    .main .block-container {
        padding-top: 0rem !important;
        margin-top: -1.5rem !important;
    }
    
    .stApp {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Floating particles
st.markdown(
    """
<div id="particles">
    <div class="particle" style="width: 50px; height: 50px; background: rgba(102, 126, 234, 0.3); top: 10%; left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="width: 30px; height: 30px; background: rgba(118, 75, 162, 0.3); top: 50%; left: 80%; animation-delay: 3s;"></div>
    <div class="particle" style="width: 40px; height: 40px; background: rgba(240, 147, 251, 0.3); top: 70%; left: 20%; animation-delay: 6s;"></div>
    <div class="particle" style="width: 25px; height: 25px; background: rgba(102, 126, 234, 0.3); top: 30%; left: 60%; animation-delay: 9s;"></div>
    <div class="particle" style="width: 35px; height: 35px; background: rgba(118, 75, 162, 0.3); top: 80%; left: 70%; animation-delay: 12s;"></div>
</div>
""",
    unsafe_allow_html=True,
)

# Additional JavaScript to remove header completely + force scroll to top on load
st.markdown(
    """
    <script>
        const removeHeader = () => {
            const header = window.parent.document
                ? window.parent.document.querySelector('header[data-testid="stHeader"]')
                : document.querySelector('header[data-testid="stHeader"]');

            if (header) {
                header.style.display = 'none';
                header.style.height = '0px';
            }

            const toolbar = window.parent.document
                ? window.parent.document.querySelector('div[data-testid="stToolbar"]')
                : document.querySelector('div[data-testid="stToolbar"]');

            if (toolbar) {
                toolbar.style.display = 'none';
            }

            const decoration = window.parent.document
                ? window.parent.document.querySelector('div[data-testid="stDecoration"]')
                : document.querySelector('div[data-testid="stDecoration"]');

            if (decoration) {
                decoration.style.display = 'none';
            }
        };

        const hardScrollTop = () => {
            try {
                if (window.location.hash) {
                    history.replaceState(
                        null,
                        '',
                        window.location.pathname + window.location.search
                    );
                }
            } catch (e) {}

            try {
                if (window.parent && window.parent.scrollTo) {
                    window.parent.scrollTo(0, 0);
                } else {
                    window.scrollTo(0, 0);
                }
            } catch (e) {
                window.scrollTo(0, 0);
            }
        };

        if ('scrollRestoration' in history) {
            history.scrollRestoration = 'manual';
        }

        window.addEventListener('load', () => {
            removeHeader();
            setTimeout(hardScrollTop, 80);
        });
    </script>
    """,
    unsafe_allow_html=True,
)

# ==========================
# API Functions
# ==========================

@st.cache_data(ttl=60)
def fetch_profiles() -> Dict[str, Any]:
    """Fetch available profiles from the FastAPI backend."""
    url = f"{API_BASE_URL}/profiles"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {
            "system_profile": "default",
            "default_threshold": 0.40,
            "profiles": [
                {
                    "key": "default",
                    "label": "Default (balanced)",
                    "threshold": 0.40,
                    "description": "General-purpose profile.",
                }
            ],
        }


def call_predict_api(text: str, profile: Optional[str]) -> Dict[str, Any]:
    """Call the /predict endpoint for a single email."""
    params: Dict[str, Any] = {}
    if profile:
        params["profile"] = profile
    resp = requests.post(
        f"{API_BASE_URL}/predict",
        json={"text": text},
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def call_file_api(uploaded_file, profile: Optional[str]) -> bytes:
    """Call the /file-predict endpoint for file-based email classification."""
    params: Dict[str, Any] = {}
    if profile:
        params["profile"] = profile

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    resp = requests.post(
        f"{API_BASE_URL}/file-predict",
        params=params,
        files=files,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


# ==========================
# UI Helper Functions
# ==========================

def normalize_label(raw_label: str) -> str:
    """Normalize model labels to UI labels."""
    v = (raw_label or "").strip().lower()
    if v == "spam":
        return "Spam"
    if v in {"ham", "real"}:
        return "Real"
    return raw_label or ""


def render_futuristic_label(label: str) -> None:
    """Render a minimal label chip."""
    ui_label = normalize_label(label)
    is_spam = ui_label.lower() == "spam"

    if is_spam:
        bg = "linear-gradient(135deg, #ff7676 0%, #e64949 100%)"
        text = "⚠️ Spam Email"
    else:
        bg = "linear-gradient(135deg, #65d177 0%, #3fbf5a 100%)"
        text = "✅ Legitimate Email"

    st.markdown(
        f"""
        <div style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 2rem;
            border-radius: 999px;
            background: {bg};
            color: #ffffff;
            font-weight: 700;
            font-size: 1.15rem;
            letter-spacing: 0.03em;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        ">
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_advanced_probability_bar(prob: Optional[float]) -> None:
    """Render probability visualization."""
    if prob is None:
        st.write("No probability available.")
        return

    pct = round(prob * 100, 1)

    if pct < 30:
        color = "#4caf50"
        label_color = "#2e7d32"
        emoji = "✅"
        status = "Low Risk"
    elif pct < 70:
        color = "#ffb300"
        label_color = "#f57c00"
        emoji = "⚡"
        status = "Medium Risk"
    else:
        color = "#e53935"
        label_color = "#b71c1c"
        emoji = "⚠️"
        status = "High Risk"

    st.markdown(
        f"""
        <div style="
            margin: 20px 0;
            padding: 20px 24px;
            background: rgba(8, 14, 40, 0.9);
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            ">
                <div>
                    <div style="
                        font-size: 0.9rem;
                        text-transform: uppercase;
                        letter-spacing: 0.15em;
                        color: rgba(255,255,255,0.7);
                        margin-bottom: 8px;
                    ">
                        Spam Probability
                    </div>
                    <span style="
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        padding: 6px 14px;
                        border-radius: 999px;
                        background: rgba(255,255,255,0.08);
                        border: 1px solid {label_color};
                        font-size: 0.85rem;
                        color: #ffffff;
                        font-weight: 600;
                    ">
                        <span>{emoji}</span>
                        <span>{status}</span>
                    </span>
                </div>
                <div style="
                    font-size: 2rem;
                    font-weight: 800;
                    color: {label_color};
                ">
                    {pct}%
                </div>
            </div>
            <div style="
                background: rgba(0,0,0,0.5);
                border-radius: 999px;
                height: 20px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct}%;
                    height: 100%;
                    border-radius: 999px;
                    background: linear-gradient(90deg, {color}, {label_color});
                    transition: width 1s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_dashboard_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level metrics for dashboards."""
    if df.empty:
        return {"total": 0, "spam": 0, "real": 0, "spam_rate": 0.0}

    labels = df["pred"].apply(normalize_label)
    total = len(labels)
    spam = int((labels == "Spam").sum())
    real = int((labels == "Real").sum())
    spam_rate = spam / total if total > 0 else 0.0
    
    return {
        "total": total,
        "spam": spam,
        "real": real,
        "spam_rate": spam_rate,
    }


def render_neon_metric_card(
    value: str,
    label: str,
    icon: str = "",
) -> None:
    st.markdown(
        f"""
        <div class='neon-stat-card pulse-on-load'>
            <div style="
                font-size: 1.4rem;
                margin-bottom: 0.5rem;
                filter: drop-shadow(0 0 6px rgba(255,255,255,0.3));
            ">{icon}</div>
            <div class='stat-value'>{value}</div>
            <div class='stat-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels and add percentage column if available."""
    if "pred" in df.columns:
        df["pred"] = df["pred"].apply(normalize_label)

    if "proba_spam" in df.columns:
        df["spam_probability_%"] = (df["proba_spam"] * 100).round(2)

    df = df.dropna(how="all")
    
    display_cols = [c for c in ["text", "pred", "spam_probability_%"] if c in df.columns]
    if display_cols:
        df = df.dropna(subset=display_cols, how="all")
    
    return df


def render_overview_and_charts(
    df: pd.DataFrame, threshold_pct: Optional[float] = None
) -> Dict[str, Any]:
    """Render overview cards and charts for a results dataframe."""
    metrics = compute_dashboard_metrics(df)
    if threshold_pct is None:
        threshold_pct = 50.0

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Overview")
        
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_neon_metric_card(str(metrics["total"]), "Total Emails", "📧")
    with m2:
        render_neon_metric_card(str(metrics["spam"]), "Spam Detected", "⚠️")
    with m3:
        render_neon_metric_card(str(metrics["real"]), "Legitimate", "✅")
    with m4:
        render_neon_metric_card(
            f"{metrics['spam_rate'] * 100:.1f}%", "Spam Rate", "📈"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    chart_cols = st.columns(2, gap="large")

    # Classification Distribution
    with chart_cols[0]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Classification Distribution")
        if "pred" in df.columns:
            labels = df["pred"].apply(normalize_label)
            counts = labels.value_counts().reindex(["Real", "Spam"], fill_value=0)
            chart_df = counts.reset_index()
            chart_df.columns = ["Label", "Count"]

            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Label:N", title="Label", axis=alt.Axis(labelColor="#ffffff", titleColor="#ffffff")),
                    y=alt.Y("Count:Q", title="Count", axis=alt.Axis(labelColor="#ffffff", titleColor="#ffffff")),
                    color=alt.Color(
                        "Label:N",
                        scale=alt.Scale(
                            domain=["Real", "Spam"],
                            range=["#2ecc71", "#e74c3c"],
                        ),
                        legend=None,
                    ),
                )
                .properties(height=400)
                .configure_view(
                    strokeWidth=0,
                    fill='#1a1f3a'
                )
                .configure_axis(
                    gridColor='rgba(255,255,255,0.1)',
                    domainColor='rgba(255,255,255,0.2)'
                )
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("ℹ️ No prediction labels available")
        st.markdown("</div>", unsafe_allow_html=True)

    # Probability Distribution
    with chart_cols[1]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if "spam_probability_%" in df.columns and not df["spam_probability_%"].isna().all():
            st.markdown("#### 📈 Probability Distribution")

            probs = df["spam_probability_%"].dropna()

            hist, bin_edges = np.histogram(probs, bins=10, range=(0, 100))
            bins_labels = []
            mids = []
            risks = []
            for i in range(len(bin_edges) - 1):
                start = int(bin_edges[i])
                end = int(bin_edges[i + 1])
                mid = (start + end) / 2
                bins_labels.append(f"{start}-{end}")
                mids.append(mid)
                risks.append("Low Risk" if mid < threshold_pct else "High Risk")

            hist_df = pd.DataFrame(
                {
                    "Range": bins_labels,
                    "Mid": mids,
                    "Count": hist,
                    "Risk": risks,
                }
            )

            color_scale = alt.Scale(
                domain=["Low Risk", "High Risk"],
                range=["#2ecc71", "#e74c3c"],
            )

            chart = (
                alt.Chart(hist_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Range:N", title="Spam Probability (%)", axis=alt.Axis(labelColor="#ffffff", titleColor="#ffffff")),
                    y=alt.Y("Count:Q", title="Number of Emails", axis=alt.Axis(labelColor="#ffffff", titleColor="#ffffff")),
                    color=alt.Color("Risk:N", scale=color_scale, legend=None),
                    tooltip=["Range", "Count", "Risk"],
                )
                .properties(height=400)
                .configure_view(
                    strokeWidth=0,
                    fill='#1a1f3a'
                )
                .configure_axis(
                    gridColor='rgba(255,255,255,0.1)',
                    domainColor='rgba(255,255,255,0.2)'
                )
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption(f"Threshold reference ≈ {threshold_pct:.0f}% spam probability")
        else:
            st.info("ℹ️ No probability data available")
        st.markdown("</div>", unsafe_allow_html=True)

    return metrics


# ==========================
# Profile helpers
# ==========================

def get_profile_style(label: str) -> Dict[str, str]:
    name = (label or "").lower()
    variant = "balanced"
    if "aggress" in name:
        variant = "aggressive"
    elif "strict" in name or "bank" in name or "financial" in name:
        variant = "strict"

    badge_bg = "rgba(15, 23, 42, 0.55)"

    if variant == "strict":
        icon_grad = "radial-gradient(circle at 30% 30%, #bfdbfe 0%, #60a5fa 40%, #2563eb 100%)"
        border = "rgba(59,130,246,0.75)"
    elif variant == "aggressive":
        icon_grad = "radial-gradient(circle at 30% 30%, #fed7aa 0%, #fb923c 40%, #f97316 100%)"
        border = "rgba(249,115,22,0.8)"
    else:
        icon_grad = "radial-gradient(circle at 30% 30%, #a5f3fc 0%, #38bdf8 40%, #0ea5e9 100%)"
        border = "rgba(56,189,248,0.8)"

    return {"badge_bg": badge_bg, "icon_grad": icon_grad, "border": border}


def render_profile_selector(
    profiles_list: List[Dict[str, Any]],
    default_profile_key: str,
    label: str,
    widget_key: str,
    show_threshold: bool = True,
) -> str:
    st.markdown(label)

    current_label = st.session_state.get("active_profile_label")
    current_index = 0
    
    if current_label:
        for i, p in enumerate(profiles_list):
            if p["label"] == current_label:
                current_index = i
                break
    else:
        for i, p in enumerate(profiles_list):
            if p["key"] == default_profile_key:
                current_index = i
                break

    selected_label = st.selectbox(
        "Select profile:",
        options=[p["label"] for p in profiles_list],
        index=current_index,
        key=widget_key,
    )

    if selected_label != st.session_state.get("active_profile_label"):
        st.session_state["active_profile_label"] = selected_label
        st.rerun()

    selected_key = next(
        (p["key"] for p in profiles_list if p["label"] == selected_label),
        default_profile_key,
    )
    desc = next(
        (p.get("description", "") for p in profiles_list if p["key"] == selected_key),
        "",
    )
    thresh = next(
        (p.get("threshold") for p in profiles_list if p["key"] == selected_key),
        None,
    )

    if desc:
        st.info(desc)
    if show_threshold and thresh is not None:
        st.metric("Threshold", f"{thresh:.2f}")

    return selected_key


def get_active_profile(
    profiles_list: List[Dict[str, Any]],
    default_key: str,
) -> Dict[str, Any]:
    label = st.session_state.get("active_profile_label")
    if label:
        for p in profiles_list:
            if p.get("label") == label:
                return p
    for p in profiles_list:
        if p.get("key") == default_key:
            return p
    return {"key": default_key, "label": "Default (balanced)", "threshold": 0.40}


def get_profile_by_key(
    profiles_list: List[Dict[str, Any]],
    key: str,
    default_key: str,
) -> Dict[str, Any]:
    for p in profiles_list:
        if p.get("key") == key:
            return p
    return get_active_profile(profiles_list, default_key)


# ==========================
# Profiles bootstrap
# ==========================

profiles_data = fetch_profiles()
profiles_list = profiles_data.get("profiles", [])
system_profile = profiles_data.get("system_profile", "default")

default_profile_key = system_profile
if not profiles_list:
    profiles_list = [
        {
            "key": "default",
            "label": "Default (balanced)",
            "threshold": 0.40,
            "description": "General-purpose profile.",
        }
    ]
    default_profile_key = "default"

if "active_profile_label" not in st.session_state:
    default_profile = get_active_profile(profiles_list, default_profile_key)
    st.session_state["active_profile_label"] = default_profile.get(
        "label", "Default (balanced)"
    )

active_profile = get_active_profile(profiles_list, default_profile_key)
active_label = active_profile.get("label", "Default (balanced)")
active_thresh = active_profile.get("threshold")
active_thresh_text = (
    f"{active_thresh:.2f}" if isinstance(active_thresh, (int, float)) else "N/A"
)
style_profile = get_profile_style(active_label)

# ==========================
# Hero Section with Unique Logo
# ==========================

current_active_profile = get_active_profile(profiles_list, default_profile_key)
current_active_label = current_active_profile.get("label", "Default (balanced)")
current_active_thresh = current_active_profile.get("threshold")
current_active_thresh_text = (
    f"{current_active_thresh:.2f}" if isinstance(current_active_thresh, (int, float)) else "N/A"
)
current_style_profile = get_profile_style(current_active_label)

hero_html = f"""
<div class="hero-section">
  <div style="position: relative; z-index: 2; text-align: center;">

    <div class="logo-container">
      <a href="{SITE_URL}" target="_blank" style="text-decoration:none; color:inherit;">
      </a>
    </div>

    <h1 style="
        font-size: 2.8rem;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 0 3px 15px rgba(0,0,0,0.4);
    ">
      Email Spam Detection
    </h1>
    <p style="
        font-size: 1.05rem;
        color: rgba(255,255,255,0.95);
        margin-top: 1.2rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    ">
      Smart, explainable spam filtering for emails & SMS with live analytics dashboard.
    </p>

    <div style="
        margin-top: 1.8rem;
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.7rem 1.5rem;
        border-radius: 999px;
        background: {current_style_profile["badge_bg"]};
        color: white;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
        border: 1px solid {current_style_profile["border"]};
    ">
      <span style="
          width: 30px;
          height: 30px;
          border-radius: 999px;
          background: {current_style_profile["icon_grad"]};
          display: inline-flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 0 0 3px rgba(15,23,42,0.7);
      ">
        <span style="font-size: 1.15rem;">🛡️</span>
      </span>
      <span style="opacity:0.95;">Active Profile:</span>
      <strong style="font-weight:700;">{current_active_label}</strong>
      <span style="opacity:0.85;">• Threshold: {current_active_thresh_text}</span>
    </div>
  </div>
</div>
"""

# ----- Hero -----
with st.container():
    components.html(hero_html, height=150, scrolling=False)

st.markdown("<br>", unsafe_allow_html=True)

# ----- How to Use -----
st.markdown(
    """
<div class="how-to-use-card">
  <h3>📘 How to Use This Tool</h3>
  <div class="step-grid">
    <div class="how-to-step">
      <div class="step-number">1</div>
      <h4>Choose Mode</h4>
      <p>Select between Single Email Prediction or file upload based on your needs.</p>
    </div>
    <div class="how-to-step">
      <div class="step-number">2</div>
      <h4>Select Profile</h4>
      <p>Pick a detection profile (balanced, strict, or aggressive) that matches your risk tolerance.</p>
    </div>
    <div class="how-to-step">
      <div class="step-number">3</div>
      <h4>Run Analysis</h4>
      <p>Click Prediction and the dashboard will automatically update with detailed metrics and visualizations.</p>
    </div>
    <div class="how-to-step">
      <div class="step-number">4</div>
      <h4>Review Results</h4>
      <p>Examine the results dashboard, export CSV files, or inspect individual suspicious emails.</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----- Tabs Section (centered pill) -----
st.markdown("<br>", unsafe_allow_html=True)

# Keep the selection in session_state
if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "🔍 Single Prediction"

# Center the radio group in the middle column
left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

with center_col:
    current_label = st.radio(
        "",
        ["🔍 Single Prediction", "📊 Multi Email Prediction"],
        index=0 if st.session_state["nav_tab"] == "🔍 Single Prediction" else 1,
        horizontal=True,
        key="nav_tab_radio",
        label_visibility="collapsed",
    )

# Update active tab
st.session_state["nav_tab"] = current_label

TAB_MAP = {
    "🔍 Single Prediction": "single",
    "📊 Multi Email Prediction": "file",
}
current_tab = TAB_MAP[current_label]


# ==========================
# Tabs content
# ==========================

if "single_email_text" not in st.session_state:
    st.session_state["single_email_text"] = ""

# ---- Single Email Tab
if current_tab == "single":
    st.markdown("## 🔍 Single Email Prediction")
    st.markdown(
        "Prediction an individual email with instant classification, probability, and a clear risk summary."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("#### 📝 Email Content")
        text_input = st.text_area(
            "Email content",
            value=st.session_state["single_email_text"],
            placeholder=(
                "Example:\nSubject: Urgent: Your Account Has Been Temporarily Restricted\n"
                "\nDear Customer\n"
                "\nWe detected unusual activity on your bank account. For your safety, access has been temporarily restricted.\n"  
                "\nPlease verify your identity immediately using the secure link below:\n"
                "\nhttps://secure-bank-verification.com/login\n"
                "\nIf you do not complete verification within 24 hours, your account may remain locked.\n"
                "\nBank Security Team\n"
            ),
            height=260,
            label_visibility="collapsed",
        )
        st.session_state["single_email_text"] = text_input

        st.markdown(
            """
            <div style="margin-top:0.5rem; font-size:0.85rem; opacity:0.8;">
                💡 <strong>Tip:</strong> Include the subject and main body of the email for best results.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        single_profile_key = render_profile_selector(
            profiles_list=profiles_list,
            default_profile_key=default_profile_key,
            label="#### ⚙️ Detection Profile",
            widget_key="single_profile_label",
            show_threshold=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn2:
        run_single = st.button("🚀 Spam Prediction", use_container_width=True)

    if run_single:
        if not text_input.strip():
            st.warning("⚠️ Please enter an email to predict.")
        else:
            with st.spinner("🔄 Predicting email..."):
                try:
                    result = call_predict_api(text_input, single_profile_key)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div id="single-result-anchor"></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Results")

                    col_a, col_b = st.columns([1, 2], gap="large")
                    with col_a:
                        render_futuristic_label(result.get("pred", ""))
                    with col_b:
                        render_advanced_probability_bar(result.get("proba_spam"))

                    proba = result.get("proba_spam")
                    if isinstance(proba, (int, float, np.floating)):
                        pct = proba * 100
                        if pct < 30:
                            st.success(
                                f"✅ This email appears **legitimate**. Estimated spam probability: **{pct:.1f}%**."
                            )
                        elif pct < 70:
                            st.warning(
                                f"⚡ This email is in a **borderline zone**. Estimated spam probability: **{pct:.1f}%**. "
                                "Review carefully before taking action."
                            )
                        else:
                            st.error(
                                f"⚠️ This email is **highly likely to be spam**. Estimated spam probability: **{pct:.1f}%**. "
                                "Avoid clicking links or sharing sensitive information."
                            )

                    st.markdown("#### 📧 Original Email")
                    st.code(result.get("text", text_input), language="")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Auto-scroll to the results section
                    components.html(
                        """
                        <script>
                        setTimeout(function() {
                            try {
                                const doc = window.parent && window.parent.document ? window.parent.document : document;
                                const anchor = doc.querySelector('#single-result-anchor');
                                if (anchor && anchor.scrollIntoView) {
                                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                } else if (window.parent && window.parent.scrollTo) {
                                    window.parent.scrollTo({ top: doc.body.scrollHeight || 0, behavior: 'smooth' });
                                } else {
                                    window.scrollTo({ top: document.body.scrollHeight || 0, behavior: 'smooth' });
                                }
                            } catch (e) {
                                window.scrollTo(0, document.body.scrollHeight || 0);
                            }
                        }, 300);
                        </script>
                        """,
                        height=0,
                    )

                except requests.HTTPError as e:
                    st.error(f"❌ API Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")

# ---- File Tab
elif current_tab == "file":
    st.markdown("## 📊 Multi Email Prediction")
    st.markdown(
        "Upload CSV / Excel for large-scale analysis. The dashboard automatically updates with comprehensive results."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("#### 📥 Upload Email File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.success(
                f"✅ File loaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)"
            )

        st.markdown(
            """
            <div style="margin-top:0.5rem; font-size:0.85rem; opacity:0.8;">
                💡 <strong>File formats:</strong><br>
                • <strong>CSV / Excel</strong>: must contain a column named <code>text</code> (one email per row).<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        file_profile_key = render_profile_selector(
            profiles_list=profiles_list,
            default_profile_key=default_profile_key,
            label="#### ⚙️ Detection Profile",
            widget_key="file_profile_label",
            show_threshold=False,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn2:
        run_file = st.button("🚀 SPAM PREDICTION ", use_container_width=True)

    if run_file:
        if not uploaded_file:
            st.warning("⚠️ Please upload a file first.")
        else:
            with st.spinner("🔄 Processing file..."):
                try:
                    content = call_file_api(uploaded_file, file_profile_key)
                    df = pd.read_csv(io.BytesIO(content))
                    df = prepare_df_for_display(df)

                    if df.empty:
                        st.info("ℹ️ No valid results to display after processing file.")
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)

                        # 🔽 Anchor for auto-scroll after analysis
                        st.markdown(
                            '<div id="file-result-anchor"></div>',
                            unsafe_allow_html=True,
                        )

                        profile_obj = get_profile_by_key(
                            profiles_list, file_profile_key, default_profile_key
                        )
                        thresh_pct = (
                            float(profile_obj.get("threshold")) * 100
                            if profile_obj.get("threshold") is not None
                            else None
                        )
                        metrics = render_overview_and_charts(
                            df, threshold_pct=thresh_pct
                        )

                        spam_rate_pct = metrics["spam_rate"] * 100
                        if spam_rate_pct < 30:
                            st.success(
                                f"✅ File looks **mostly safe**. Estimated spam rate: **{spam_rate_pct:.1f}%**."
                            )
                        elif spam_rate_pct < 70:
                            st.warning(
                                f"⚡ Mixed results in file: about **{spam_rate_pct:.1f}% spam. "
                                "Review suspicious rows or adjust your profile."
                            )
                        else:
                            st.error(
                                f"⚠️ High spam density detected in file: **{spam_rate_pct:.1f}% spam**. "
                                "Consider blocking or quarantining these messages."
                            )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            "<div class='glass-card'>", unsafe_allow_html=True
                        )
                        st.markdown("#### 📋 Sample Results (First 50 Rows)")
                        sample_cols = [
                            c
                            for c in ["text", "pred", "spam_probability_%"]
                            if c in df.columns
                        ]
                        if sample_cols:
                            display_df = df[sample_cols].head(50).dropna(how="all")
                            if not display_df.empty:
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    height=420,
                                )
                            else:
                                st.info(
                                    "ℹ️ No valid data to display in sample results."
                                )
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button(
                            label="⬇️ Download Full Results CSV",
                            data=content,
                            file_name=f"spam_analysis_{uploaded_file.name}",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        # 🔽 Auto-scroll to the results section (like Single Analysis)
                        components.html(
                            """
                            <script>
                            setTimeout(function() {
                                try {
                                    const doc = window.parent && window.parent.document ? window.parent.document : document;
                                    const anchor = doc.querySelector('#file-result-anchor');
                                    if (anchor && anchor.scrollIntoView) {
                                        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                    } else if (window.parent && window.parent.scrollTo) {
                                        window.parent.scrollTo({ top: doc.body.scrollHeight || 0, behavior: 'smooth' });
                                    } else {
                                        window.scrollTo({ top: document.body.scrollHeight || 0, behavior: 'smooth' });
                                    }
                                } catch (e) {
                                    window.scrollTo(0, document.body.scrollHeight || 0);
                                }
                            }, 300);
                            </script>
                            """,
                            height=0,
                        )

                except requests.HTTPError as e:
                    error_msg = str(e)
                    if "422" in error_msg:
                        st.error(
                            "❌ **File Format Error**: The uploaded file doesn't match the expected format. "
                            "Please ensure:\n\n"
                            "• **CSV/Excel files** contain a column named `text` with one email per row\n"
                            "• **TXT files** have one email per line\n"
                            "• The file is not empty or corrupted"
                        )
                    else:
                        st.error(f"❌ API Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {e}")
    