# Using Data Science to Detect Email Fraud and Spam

This repository contains the full pipeline for training and evaluating a classic ML spam/fraud email detector.

## Structure
```
spam-detector/
├─ data/ (raw/, interim/, processed/)
├─ notebooks/  (01_eda.ipynb, 02_modeling.ipynb, 03_eval.ipynb)
├─ src/ (preprocess.py, features.py, train.py, evaluate.py, infer.py, utils.py)
├─ models/ (vectorizer.pkl, model.pkl)
├─ reports/ (figures/, tables/)
└─ README.md, requirements.txt
```

## Quickstart
1) Create a virtual environment and install requirements:
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

2) Place a labeled CSV in `data/raw/` with columns like:
```
label,text
spam,"WIN a FREE iPhone now! Click this link..."
ham,"Dear team, please find the attached report."
```

3) Run training (uses TF-IDF + Naive Bayes / Logistic Regression / Linear SVM baseline comparison):
```
python src/train.py --data data/raw/your_dataset.csv --target label --text text
```

4) Evaluate on held-out split and save figures:
```
python src/evaluate.py --data data/raw/your_dataset.csv --target label --text text
```

5) Inference (CLI demo):
```
python src/infer.py --text "Your account is locked. Verify here: http://bit.ly/xyz"
```

## Deliverables mapping
- 5-page report figures are saved to `reports/figures/`.
- Confusion matrices, ROC curves, and metric tables exported to `reports/tables/`.
- Persisted artifacts in `models/`.
- Notebooks mirror src scripts for transparency and iteration.
