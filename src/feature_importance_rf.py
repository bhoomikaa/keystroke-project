import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df[df["condition"] == "baseline"]

# -------------------------
# Session-based split
# -------------------------
train_df = df[df["session"].isin(["s0", "s1"])]

# Select all behavioral features
feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]

le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])

# -------------------------
# Train tuned Random Forest again
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
# Extract feature importance
# -------------------------
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)

print("Top 15 Most Important Features:\n")
print(feature_importance_df.head(15))

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10,6))
plt.barh(
    feature_importance_df["feature"].head(15)[::-1],
    feature_importance_df["importance"].head(15)[::-1]
)
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
