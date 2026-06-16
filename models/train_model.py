import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data/Student Mental health.csv")

# -------------------------
# Data Cleaning
# -------------------------

df["Your current year of Study"] = (
    df["Your current year of Study"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["What is your CGPA?"] = (
    df["What is your CGPA?"]
    .astype(str)
    .str.strip()
)

df["Choose your gender"] = (
    df["Choose your gender"]
    .astype(str)
    .str.strip()
)

df["Marital status"] = (
    df["Marital status"]
    .astype(str)
    .str.strip()
)

# Remove rows with missing Age
df = df.dropna(subset=["Age"])

# -------------------------
# Features and Target
# -------------------------

X = df[
    [
        "Choose your gender",
        "Age",
        "Your current year of Study",
        "What is your CGPA?",
        "Marital status"
    ]
].copy()

y = df["Do you have Depression?"]

# Encode categorical columns
encoders = {}

for col in [
    "Choose your gender",
    "Your current year of Study",
    "What is your CGPA?",
    "Marital status"
]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# -------------------------
# Train Model
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "models/depression_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("Model saved successfully!")