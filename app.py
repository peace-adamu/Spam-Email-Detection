import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Load model and vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("📧 Spam Detector")
user_input = st.text_area("Enter your email message:")

if st.button("Predict"):
    processed = preprocess_text(user_input)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    label = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    st.success(f"Prediction: {label}")
