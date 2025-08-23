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
st.sidebar.title("About")
st.sidebar.info("Built by Peace using Streamlit, NLTK, and XGBoost.")
st.markdown("This app predicts whether an email message is Spam or Not Spam." \
" Enter your email message below and click 'Predict'." \
" The model is trained on the SMS Spam Collection dataset." \
" It uses XGBoost for classification and NLTK for text preprocessing." \
" Let's check if your message is spam!")

user_input = st.text_area("Enter your email message:")

if st.button("Predict"):
    processed = preprocess_text(user_input)
    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    label = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    st.success(f"Prediction: {label}")

