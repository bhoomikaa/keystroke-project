import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")

# Explicit missing data handling
df = df.fillna(0)

# Baseline-only subset
df = df[df["condition"] == "baseline"]

# -------------------------
# Train/Test split (session-based)
# -------------------------
train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

# Use ALL behavioral features
feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

# -------------------------
# Feature scaling (required for Logistic Regression)
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Train Logistic Regression
# -------------------------
model = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs"
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Logistic Regression (Full Feature Set)")
print("----------------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))