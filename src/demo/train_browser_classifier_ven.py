"""Train a small, browser-exportable version of the Tshivenda misinformation classifier.

The real reported result (Objective 4, notes/tshivenda-classifier-proxy.md)
is AfroXLM-RoBERTa: 0.562 accuracy, 278M parameters - far too large to run in
a webpage with no backend. This trains a TF-IDF + logistic regression model
on the exact same proxy dataset and exports its weights as JSON, small enough
to embed directly in a static page and score arbitrary typed-in text
client-side, live, with no server.

This is a different, much simpler model on the same task and same data, not
a compressed copy of AfroXLM-RoBERTa - report its own accuracy honestly
alongside it, never as a replacement for the real number.

Usage:
    python src/demo/train_browser_classifier_ven.py
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "dataset" / "vukuzenzele" / "misinfo_proxy_ven.csv"
OUT_JSON = REPO_ROOT / "src" / "demo" / "browser_classifier_weights_ven.json"

# letters only (no digits/underscore), length >= 2 - matches the JS tokenizer
# used at inference time in the artifact: text.toLowerCase().match(/[\p{L}]{2,}/gu)
TOKEN_PATTERN = r"(?u)[^\W\d_]{2,}"
MAX_FEATURES = 3000


def load_rows():
    rows = []
    with open(DATA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"text": row["text"], "label": int(row["label"]),
                        "source_id": int(row["source_id"])})
    return rows


def cross_validate(rows):
    groups = [r["source_id"] for r in rows]
    labels = [r["label"] for r in rows]
    texts = [r["text"] for r in rows]

    gkf = GroupKFold(n_splits=5)
    accs, f1s = [], []
    for train_idx, eval_idx in gkf.split(texts, labels, groups):
        vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, max_features=MAX_FEATURES, lowercase=True)
        X_train = vec.fit_transform([texts[i] for i in train_idx])
        X_eval = vec.transform([texts[i] for i in eval_idx])
        y_train = [labels[i] for i in train_idx]
        y_eval = [labels[i] for i in eval_idx]

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_eval)
        accs.append(accuracy_score(y_eval, preds))
        f1s.append(f1_score(y_eval, preds, average="macro"))

    print(f"5-fold grouped CV: accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f}, "
          f"macro F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    return float(np.mean(accs)), float(np.mean(f1s))


def train_final_and_export(rows, cv_accuracy, cv_f1):
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, max_features=MAX_FEATURES, lowercase=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, labels)

    vocab = {term: int(idx) for term, idx in vec.vocabulary_.items()}
    idf = vec.idf_.tolist()
    coef = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    payload = {
        "vocab": vocab,
        "idf": idf,
        "coef": coef,
        "intercept": intercept,
        "n_train": len(rows),
        "cv_accuracy": round(cv_accuracy, 4),
        "cv_macro_f1": round(cv_f1, 4),
        "note": "label 1 = real, label 0 = fake; sigmoid(x . coef + intercept) = P(real)",
    }
    OUT_JSON.write_text(json.dumps(payload))
    print(f"exported {len(vocab)} vocab terms -> {OUT_JSON} ({OUT_JSON.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    rows = load_rows()
    cv_accuracy, cv_f1 = cross_validate(rows)
    train_final_and_export(rows, cv_accuracy, cv_f1)
