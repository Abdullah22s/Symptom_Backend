import joblib
import re

# STEP 1: Load the trained model
model = joblib.load("doctor_specialty_model.pkl")
print("✅ Model loaded successfully")

# STEP 2: Clean text (MUST MATCH TRAINING)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# STEP 3: Test sentences (English + Roman Urdu)
test_sentences = [
    "I have runny nose",
    "i have chest pain and breathing problem",
    "skin par khujli aur daane ho gaye hain",
    "seene mein dard aur dil tez dhadak raha hai",
    "tanngo mein dard hai",
    "unable to sleep and mental tension"
]

# STEP 4: Predict
for sentence in test_sentences:
    cleaned = clean_text(sentence)
    prediction = model.predict([cleaned])[0]
    print(f"Input: {sentence}")
    print(f"Predicted Specialty: {prediction}\n")
