import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df[df["condition"] == "baseline"]

# Explicit missing data handling
df = df.fillna(0)

# Session-based split
train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

# Train final tuned Random Forest
model = RandomForestClassifier(
    n_estimators=700,
    max_depth=25,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# Error Analysis
# -------------------------

results_df = test_df.copy()
results_df["true_user"] = y_test
results_df["predicted_user"] = y_pred
results_df["correct"] = results_df["true_user"] == results_df["predicted_user"]

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Test Accuracy:", round(accuracy, 4))

# Count errors per user
error_counts = results_df[~results_df["correct"]]["true_user"].value_counts()

print("\nUsers with most misclassifications:")
print(error_counts.head(10))

# Overall error rate
print("\nTotal misclassifications:", len(error_counts))
