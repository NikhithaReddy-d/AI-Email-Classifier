import joblib
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "email_model.pkl")
model, vectorizer = joblib.load(MODEL_PATH)

# Pre-defined reply templates
TEMPLATES = {
    "Urgent Client": [
        "Thanks for the update — we're investigating this now.",
        "Apologies for the inconvenience. Our team will respond shortly."
    ],
    "Meeting": [
        "Noted. I will attend the meeting.",
        "Confirmed. Please share the agenda if any."
    ],
    "HR": [
        "Received. I will follow the instructions.",
        "Noted. Thank you for the update."
    ],
    "Task": [
        "Acknowledged. I will complete this and update you.",
        "Got it. I will start immediately."
    ],
    "Spam": [
        "This appears to be spam. No action needed.",
        "Marked as spam automatically."
    ]
}

def classify_email(text):
    X_test = vectorizer.transform([text])
    label = model.predict(X_test)[0]
    replies = TEMPLATES.get(label, ["No template available"])
    return {"category": label, "suggested_replies": replies}
