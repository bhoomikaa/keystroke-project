import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
# Train Decision Tree
# -------------------------
model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
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

print("Decision Tree (Full Feature Set)")
print("----------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))
