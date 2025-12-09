import streamlit as st
from src.email_classifier import classify_email

st.title("📧 AI Email Classifier & Auto-Reply System")
st.write("Paste an email and get its category with suggested reply.")

email_text = st.text_area("Enter Email Content:")

if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter email text first.")
    else:
        result = classify_email(email_text)
        st.success(f"Category: {result['category']}")
        st.write("Suggested Replies:")
        for reply in result['suggested_replies']:
            st.code(reply)
