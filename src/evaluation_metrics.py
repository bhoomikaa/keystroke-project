import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import label_binarize

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df.fillna(0)
df = df[df["condition"] == "baseline"]

# -------------------------
# Session-based split
# -------------------------
train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

# -------------------------
# Train final tuned Random Forest
# -------------------------
model = RandomForestClassifier(
    n_estimators=700,
    max_depth=25,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# -------------------------
# Basic metrics
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))

# -------------------------
# ROC-AUC (One-vs-Rest)
# -------------------------
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")

print("ROC-AUC (OvR):", round(roc_auc, 4))

# -------------------------
# Calibration (Brier Score)
# -------------------------
# For multiclass, average Brier across classes
brier_scores = []

for i in range(y_prob.shape[1]):
    brier = brier_score_loss(
        (y_test == i).astype(int),
        y_prob[:, i]
    )
    brier_scores.append(brier)

mean_brier = np.mean(brier_scores)

print("Mean Brier Score:", round(mean_brier, 4))
