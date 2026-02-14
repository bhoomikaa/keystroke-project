import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df.fillna(0)
df = df[df["condition"] == "baseline"]

train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

results = []

# -------------------------
# Logistic Regression
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=2000, multi_class="multinomial")
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, log_pred),
    "Macro F1": f1_score(y_test, log_pred, average="macro")
})

# -------------------------
# kNN
# -------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

results.append({
    "Model": "kNN",
    "Accuracy": accuracy_score(y_test, knn_pred),
    "Macro F1": f1_score(y_test, knn_pred, average="macro")
})

# -------------------------
# Decision Tree
# -------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

results.append({
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test, dt_pred),
    "Macro F1": f1_score(y_test, dt_pred, average="macro")
})

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=700,
    max_depth=25,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

results.append({
    "Model": "Tuned Random Forest (Final)",
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Macro F1": f1_score(y_test, rf_pred, average="macro")
})

# -------------------------
# SVM
# -------------------------
svm_model = SVC(kernel="rbf", C=10, gamma="scale")
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

results.append({
    "Model": "SVM (RBF)",
    "Accuracy": accuracy_score(y_test, svm_pred),
    "Macro F1": f1_score(y_test, svm_pred, average="macro")
})

# -------------------------
# Final Table
# -------------------------
results_df = pd.DataFrame(results)

print("\nFinal Model Comparison Table")
print("-----------------------------------")
print(results_df.sort_values(by="Accuracy", ascending=False))
