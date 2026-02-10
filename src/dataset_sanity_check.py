import pandas as pd

# Load the feature dataset
df = pd.read_csv("../outputs/keystroke_features.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# -------------------------
# Basic sanity checks
# -------------------------

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of unique users:")
print(df["user"].nunique())

print("\nRows per session:")
print(df["session"].value_counts())

print("\nRows per condition (baseline vs rotation):")
print(df["condition"].value_counts())

# -------------------------
# Check samples per user
# -------------------------

print("\nSamples per user (first 10 users):")
print(df["user"].value_counts().head(10))

# -------------------------
# Optional: baseline-only subset
# -------------------------

baseline_df = df[df["condition"] == "baseline"]
print("\nBaseline-only subset shape:", baseline_df.shape)
