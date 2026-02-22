"""
Final model comparison for keystroke dynamics authentication.
Reports:
  1. Baseline: single split (train s0+s1, test s2) on baseline condition.
  2. Session rotation: 3-fold leave-one-session-out (approx. mean).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from data_utils import (
    load_keystroke_data,
    get_single_split,
    iter_session_rotation_splits,
)

# -------------------------
# Load data
# -------------------------
df_baseline = load_keystroke_data(condition="baseline")

results_baseline = []
results_rotation = []


def _eval_one(X_train, X_test, y_train, y_test, scaled, fit_predict_fn):
    """Run one model on one split; return (accuracy, macro_f1)."""
    if scaled:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    acc, f1 = fit_predict_fn(X_train, X_test, y_train, y_test)
    return acc, f1


def _run_session_rotation(scaled, fit_predict_fn):
    """Run model on all 3 session-rotation folds; return (mean_acc, std_acc, mean_f1, std_f1)."""
    accs, f1s = [], []
    for X_train, X_test, y_train, y_test, _ in iter_session_rotation_splits(df_baseline):
        if scaled:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test
        acc, f1 = fit_predict_fn(X_tr, X_te, y_train, y_test)
        accs.append(acc)
        f1s.append(f1)
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


# -------------------------
# Model runners (fit and return accuracy, macro F1)
# -------------------------
def _logistic(X_train, X_test, y_train, y_test):
    m = LogisticRegression(max_iter=2000)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def _knn(X_train, X_test, y_train, y_test):
    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def _dt(X_train, X_test, y_train, y_test):
    m = DecisionTreeClassifier(random_state=42)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def _rf(X_train, X_test, y_train, y_test):
    m = RandomForestClassifier(
        n_estimators=700,
        max_depth=25,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def _svm(X_train, X_test, y_train, y_test):
    m = SVC(kernel="rbf", C=10, gamma="scale")
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


MODELS = [
    ("Logistic Regression", True, _logistic),
    ("kNN", True, _knn),
    ("Decision Tree", False, _dt),
    ("Tuned Random Forest (Final)", False, _rf),
    ("SVM (RBF)", True, _svm),
]

# -------------------------
# 1. Baseline: single split (s0+s1 train, s2 test)
# -------------------------
X_train, X_test, y_train, y_test, _ = get_single_split(df_baseline)
for name, scaled, fn in MODELS:
    acc, f1 = _eval_one(X_train, X_test, y_train, y_test, scaled, fn)
    results_baseline.append({"Model": name, "Accuracy": acc, "Macro F1": f1})

# -------------------------
# 2. Session rotation: 3-fold (each session as test once)
# -------------------------
for name, scaled, fn in MODELS:
    mean_acc, _, mean_f1, _ = _run_session_rotation(scaled, fn)
    results_rotation.append({
        "Model": name,
        "Accuracy": round(mean_acc, 4),
        "Macro F1": round(mean_f1, 4),
    })

# -------------------------
# Print tables
# -------------------------
baseline_df = pd.DataFrame(results_baseline)
print("\n" + "=" * 60)
print("1. BASELINE (train: s0+s1, test: s2, condition: baseline)")
print("=" * 60)
print(baseline_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))

rotation_df = pd.DataFrame(results_rotation)
print("\n" + "=" * 60)
print("2. SESSION ROTATION (3-fold leave-one-session-out)")
print("=" * 60)
print(rotation_df.to_string(index=False))

print()
