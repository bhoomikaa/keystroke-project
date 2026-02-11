import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("../outputs/keystroke_features.csv")
df = df[df["condition"] == "baseline"]

# -------------------------
# Explicit Missing Data Handling
# -------------------------
print("Checking for missing values...")
print(df.isnull().sum())

# Fill any missing values (defensive programming)
df = df.fillna(0)

# Session-based split
train_df = df[df["session"].isin(["s0", "s1"])]
test_df = df[df["session"] == "s2"]

# Select all behavioral features automatically
feature_cols = [col for col in df.columns if col not in ["user", "session", "condition", "file"]]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df["user"])
y_test = le.transform(test_df["user"])

# Tuned Random Forest
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

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Random Forest with Full Feature Set")
print("-------------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(f1, 4))
