import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer

# Load pre-trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectors.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Sidebar Theme Selection
theme = st.sidebar.radio("🎨 Choose Theme:", ["🌞 Light Mode", "🌙 Dark Mode"])

# Apply theme styling
if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
            body {
                background-color: #121212;
                color: white;
            }
            .stApp {
                background-color: #1e1e1e;
                color: white;
            }
            .title {
                color: #00c8ff;
                text-align: center;
                font-size: 36px;
                font-weight: bold;
            }
            .stTextArea textarea {
                background-color: #222;
                color: white;
                border: 1px solid #00c8ff;
                padding: 10px;
                border-radius: 10px;
                font-size: 16px;
            }
            .stButton>button {
                background-color: #00c8ff;
                color: black;
                border-radius: 8px;
                padding: 10px;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #009ec3;
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            body {
                background-color: #f5f5f5;
                color: black;
            }
            .title {
                color: #007bff;
                text-align: center;
                font-size: 36px;
                font-weight: bold;
            }
            .stTextArea textarea {
                background-color: white;
                color: black;
                border: 1px solid #007bff;
                padding: 10px;
                border-radius: 10px;
                font-size: 16px;
            }
            .stButton>button {
                background-color: #007bff;
                color: white;
                border-radius: 8px;
                padding: 10px;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #0056b3;
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# App Title
st.markdown('<h1 class="title">📩 Spam Detection Web App</h1>', unsafe_allow_html=True)
st.write("Enter a message below to check if it's spam or not.")

# Sidebar for additional settings
st.sidebar.header("⚙️ Settings")
st.sidebar.info("This app uses a Machine Learning model trained on spam messages.")

# Text input
user_input = st.text_area("Enter your message:", placeholder="Type your message here...")

# File Upload Option
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

if uploaded_file:
    user_input = uploaded_file.read().decode("utf-8")
    st.text_area("📂 File Content:", user_input, disabled=True)

# Spam detection logic
if st.button("🚀 Check Spam"):
    if user_input.strip():
        # Preprocess text
        processed_text = user_input.lower().translate(str.maketrans("", "", string.punctuation))
        transformed_text = vectorizer.transform([processed_text])

        # Get prediction and probability (if model supports it)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(transformed_text)[0][1]  # Probability of spam
            if proba > 0.5:
                st.error(f"🚨 This message is spam! (Confidence: {proba:.2%})")
            else:
                st.success(f"✅ This message is not spam. (Confidence: {100 - (proba * 100):.2f}%)")
        else:
            prediction = model.predict(transformed_text)[0]
            if prediction == 1:
                st.error("🚨 This message is spam!")
            else:
                st.success("✅ This message is not spam.")
    else:
        st.warning("⚠️ Please enter a valid message or upload a file.")
