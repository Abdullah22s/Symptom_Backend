# ================================
# STEP 1: Import libraries
# ================================
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ================================
# STEP 2: Load datasets
# ================================
df1 = pd.read_csv("doctor_specialty_bilingual_dataset.csv")


# Combine datasets
df = pd.concat([df1], ignore_index=True)
print("✅ Combined dataset loaded. Total rows:", len(df))

# Drop rows with missing values in 'text' or 'specialty'
df = df.dropna(subset=["text", "specialty"])
print("✅ Dataset after dropping missing values. Total rows:", len(df))


# ================================
# STEP 3: Clean text (BILINGUAL SAFE)
# ================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep English letters & numbers
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)


# ================================
# STEP 4: Split input & output
# ================================
X = df["text"]
y = df["specialty"]


# ================================
# STEP 5: Train-test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # stratified sampling by specialty
)


# ================================
# STEP 6: ML Pipeline (FIXED)
# ================================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ))
])


# ================================
# STEP 7: Train model
# ================================
print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Training completed.")


# ================================
# STEP 8: Evaluate model
# ================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))


# ================================
# STEP 9: Save trained model
# ================================
joblib.dump(model, "doctor_specialty_model.pkl")
print("\n💾 Model saved as doctor_specialty_model.pkl")
