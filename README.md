# 📧 Email Spam Detection System

A machine-learning powered platform designed to classify emails and SMS messages as **spam** or **legitimate**.
The system combines robust preprocessing, TF-IDF vectorization, and a calibrated Linear SVM model to deliver accurate, real-world spam detection suitable for security teams, developers, and data analysts.


## 🎯 Project Objectives

* Build a **reliable spam detection engine** using classical machine learning.
* Support **multiple detection profiles** (balanced, strict, aggressive…).
* Provide **consistent prediction behavior** across single and batch inputs.
* Ensure the system is easy to understand, extend, and integrate into other environments.
* Offer a clear separation between data processing, model training, and inference logic.


## 🧠 Core System Architecture

The system operates through the following stages:

### **1. Text Preprocessing**

Raw messages are normalized using:

* URL removal
* HTML stripping
* Token filtering
* Lowercasing
* Numeric/word token extraction

Handled by `src/preprocess.py`.



### **2. Feature Extraction**

Uses **TF-IDF vectorization** with:

* Word n-grams: *(1,2)*
* Sublinear TF scaling
* Minimum document frequency filtering

The result is a sparse numerical representation used for training and prediction.


### **3. Machine Learning Model**

The main classifier is a:

### **✔ Linear Support Vector Machine (Linear SVM)**

Calibrated with **CalibratedClassifierCV** to produce probability outputs.

Benefits:

* High accuracy on short text (SMS, emails)
* Fast inference time
* Well-suited for imbalanced datasets

An optional **Complement Naive Bayes** model is also available for comparison.


### **4. Decision Logic (Threshold Profiles)**

Each detection profile uses a different probability threshold:

| Profile          | Threshold | Behavior                            |
| ---------------- | --------- | ----------------------------------- |
| Balanced         | ~0.55     | General-purpose                     |
| Bank / Financial | ~0.65     | Very strict, avoids false positives |
| Marketing        | ~0.45     | Aggressive detection                |
| Telco            | ~0.55     | Optimized for OTP/alerts            |
| Conservative     | ~0.60     | Protects legitimate messages        |
| Aggressive       | ~0.45     | Captures most spam                  |

The profile determines whether the predicted probability is considered spam or legitimate.



## 📁 Project Structure (Folder Map)

```
📦 project
│
├── src/                     # ML logic & utilities
│   ├── preprocess.py        # Text normalization & cleaning
│   ├── train_model.py       # Model training pipeline
│   ├── evaluate.py          # Model evaluation
│   ├── batch_predict.py     # Batch CSV predictions (CLI)
│   └── cli_predict.py       # Command-line single prediction
│
├── ui/                      # Streamlit dashboard
│   └── app.py               # Complete interactive UI
│
├── models/                  # Saved ML models
│   └── bundle_svm.joblib    # TF-IDF + Calibrated Linear SVM
│
├── data/
│   └── raw/                 # Training datasets (e.g., spam.csv)
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE
```


## 🧩 Key Components

### **ML Pipeline**

* TF-IDF Vectorizer
* Linear SVM Classifier
* Probability calibration
* Custom thresholding per profile

### **Preprocessing Layer**

* Normalized input text
* Clean Unicode-safe pipeline
* Duplicate & empty message filtering

### **Prediction Features**

* Single message classification
* Batch prediction (CSV/Excel)
* Profile-based thresholding
* Spam probability scoring

### **UI Features (Streamlit)**

* Real-time classification
* Probability visual charts
* Multi-profile selection
* File upload for batch analysis



## ✔ Summary

This project provides a **fully structured, algorithmically solid** spam detection system using classical ML techniques.
It includes:

* A trained and packaged SVM-based classifier
* Preprocessing utilities
* A polished interactive UI
* Tools for training, evaluating, and debugging models
