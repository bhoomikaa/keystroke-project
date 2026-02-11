import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load feature dataset
df = pd.read_csv("../outputs/keystroke_features.csv")

print("Original dataset shape:", df.shape)

# ---------------------------------
# Step 1: Baseline-only subset (cleanest start)
# ---------------------------------

df = df[df["condition"] == "baseline"]

print("Baseline subset shape:", df.shape)

# ---------------------------------
# Step 2: Leakage-safe split by session
# ---------------------------------

train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ---------------------------------
# Step 3: Prepare features and labels
# ---------------------------------

feature_cols = [
    "hold_mean", "hold_std", "hold_median",
    "inter_mean", "inter_std", "inter_median",
    "typing_rate", "backspace_count"
]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Encode user labels
le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

print("\nNumber of unique users:", len(le.classes_))
print("Feature matrix shape (train):", X_train.shape)
print("Feature matrix shape (test):", X_test.shape)
