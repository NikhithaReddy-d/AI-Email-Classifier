# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Path to your CSV dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'emails.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'email_model.pkl')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and labels
X = df['text']
y = df['label']

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline: TF-IDF vectorizer + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Test model
preds = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds))

# Save the trained model
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")