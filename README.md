# 📧 Email Spam Detection System

A complete machine-learning platform for detecting spam emails and SMS messages.
The system provides a **REST API** for automated classification and a **modern Streamlit UI** for interactive analysis, making it suitable for developers, analysts, and security teams.

---

## 🎯 Project Goals

* Detect spam with high accuracy using a trained ML model
* Provide flexible prediction modes: single message  and full file upload
* Offer configurable **detection profiles** (balanced, strict, aggressive…)
* Enable developers to integrate spam classification into other systems through a simple API
* Give end-users a beautiful UI with analytics, charts, and predictions

---

## 🧠 How It Works

1. **Preprocessing**
   The text is cleaned and normalized before prediction.

2. **ML Model**
   A trained TF-IDF + Linear SVM model outputs a spam probability.

3. **Threshold Profiles**
   Depending on the selected profile (bank, telco, marketing, balanced), the system decides whether the message is spam.

4. **Delivery**
   The result is returned through:

   * API (JSON)
   * Streamlit UI dashboard
   * CSV file for batch predictions

---

## 📁 Key Files & Folders

### **api/**

Contains the FastAPI backend.

* `main.py` → All API endpoints (`/predict`, `/batch`, `/file-predict`, `/profiles`, `/health`)

### **src/**

Utility scripts:

* `train_model.py` → Train the ML model
* `evaluate.py` → Evaluate model performance
* `preprocess.py` → Text cleaning functions
* `cli_predict.py` → Predict from terminal
* `batch_predict.py` → CSV batch prediction
* `verify_api.py` → Test all API endpoints

### **models/**

Stores the model files (e.g., `bundle_svm.joblib`).

### **ui/**

The Streamlit application.

* `app.py` → Complete frontend interface

### **data/raw/**

Place your dataset here (e.g., `spam.csv`).

---

## 🚀 How to Run the Backend (API)

1. Install dependencies:

   ```bash
   pip install -r ui/requirements.txt
   ```

2. Start FastAPI server:

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

3. API documentation will be available at:

   ```
   http://localhost:8000/docs
   ```

---

## 🖥 How to Run the UI (Streamlit)

Inside the project folder run:

```bash
streamlit run ui/app.py
```

The UI automatically connects to your API and allows:

* Single email prediction
* Bulk email file prediction
* Probability visualization
* Charts and analytics

---

## 🧪 Example Usage (API)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You won a free gift card!"}'
```

---

## 🧩 Training Your Own Model (Optional)

```bash
python src/train_model.py \
  --data data/raw/spam.csv \
  --out models/bundle_svm.joblib
```

---

## ✔ Summary

This project provides:

* A trained ML spam detection engine
* A full backend API
* A modern dashboard UI
* Tools for training, evaluation, and batch processing

It is built to be **easy to run**, **simple to integrate**, and **practical for real-world use**.

---
