
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ==============================
# 1. Load Data and Train Model
# ==============================

@st.cache_resource  # Train only once
def train_model():
    # Load data
    df = pd.read_csv("full_14_specialties_100_each.csv")

    X = df["symptoms"]
    y = df["specialty"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # Initial evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    # Save model and vectorizer
    joblib.dump(model, "symptom_to_specialty_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer, acc

# Train or load model
if os.path.exists("symptom_to_specialty_model.pkl") and os.path.exists("vectorizer.pkl"):
    model = joblib.load("symptom_to_specialty_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    acc = None
else:
    model, vectorizer, acc = train_model()

# ==============================
# 2. Streamlit UI
# ==============================

st.set_page_config(page_title="Medical Symptom Checker", page_icon="🩺")

st.title("🩺 AI Symptom to Specialty Chatbot")
st.write("Enter your symptoms, and the AI will direct you to the appropriate medical specialty out of 14 options.")

if acc:
    st.info(f"📊 Model Accuracy: {acc:.2f}")

# Symptom input
user_input = st.text_area("✍️ Enter your symptoms here:")

if st.button("🔍 Find Specialty"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        st.success(f"🏥 Recommended Specialty: **{prediction}**")
    else:
        st.warning("Please enter at least one symptom.")

st.markdown("---")
st.markdown("⚠️ **Note**: This is an experimental system for initial guidance only and does not replace professional medical advice.")
