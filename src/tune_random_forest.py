import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df[df["condition"] == "baseline"]

# -------------------------
# Session-based split
# -------------------------
train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

# Use all behavioral features automatically
feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

# -------------------------
# Define parameter grid
# -------------------------
param_grid = {
    "n_estimators": [400, 700, 1000],
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,            # Cross-validation only on training sessions
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Starting Grid Search...")
grid.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid.best_params_)

# -------------------------
# Train best model
# -------------------------
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\nFinal Tuned Random Forest Performance")
print("---------------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))
