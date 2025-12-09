import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Sample training data
emails = [
    "payment failure urgent fix required",
    "please attend the meeting tomorrow",
    "congrats! you won a bonus gift card",
    "received the HR instructions for leave",
    "complete this task by today",
    "spam email, click here now"
]
labels = [
    "Urgent Client",
    "Meeting",
    "Spam",
    "HR",
    "Task",
    "Spam"
]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Save both model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "email_model.pkl")
joblib.dump((model, vectorizer), MODEL_PATH)

print(f"Model saved at {MODEL_PATH}")
