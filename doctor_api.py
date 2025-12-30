# ================================
# doctor_api.py
# ================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# ================================
# Helper function to clean text
# ================================
def clean_text(text: str) -> str:
    """
    Lowercase, remove special characters, extra spaces
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================================
# Load trained model
# ================================
model = joblib.load("doctor_specialty_model.pkl")
print("✅ Model loaded successfully.")

# ================================
# Initialize FastAPI app
# ================================
app = FastAPI(
    title="Doctor Specialty Prediction API",
    description="Send patient symptoms and get the predicted doctor specialty",
    version="1.0"
)

# ================================
# Define request body
# ================================
class PatientText(BaseModel):
    text: str

# ================================
# Prediction endpoint
# ================================
@app.post("/predict")
def predict_specialty(data: PatientText):
    """
    Receives patient text and predicts doctor specialty
    """
    # Step 1: Clean text
    cleaned_text = clean_text(data.text)
    
    # Step 2: Predict using trained model
    prediction = model.predict([cleaned_text])[0]
    
    # Step 3: Return as JSON
    return {"specialty": prediction}
