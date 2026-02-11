import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Clean visual style
# -------------------------
sns.set_style("white")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

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

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])

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
# SHAP Explanation
# -------------------------
print("Computing SHAP values...")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Handle multiclass safely
if len(shap_values.values.shape) == 3:
    shap_values_single = shap_values[:, :, 0]
else:
    shap_values_single = shap_values

print("Generating SHAP summary plot...")

plt.figure(figsize=(9, 6))

shap.summary_plot(
    shap_values_single,
    X_test,
    plot_type="dot",
    max_display=10,
    color_bar=True,
    show=False
)

plt.title("SHAP Feature Importance (Random Forest)", pad=15)
plt.xlabel("SHAP Value (Impact on Model Output)")
plt.tight_layout()

# Save high-resolution version for report
plt.savefig("shap_summary_rf_clean.png", dpi=300, bbox_inches="tight")

plt.show()
