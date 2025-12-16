import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Page Config (Dark Mode)
# ------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# Custom Dark Theme CSS
# ------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextArea textarea {
        background-color: #161B22;
        color: #FAFAFA;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        height: 45px;
        font-size: 16px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #2EA043;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .real {
        background-color: #0f5132;
        color: #d1e7dd;
    }
    .fake {
        background-color: #842029;
        color: #f8d7da;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Load Model & Vectorizer
# ------------------------------
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ------------------------------
# UI
# ------------------------------
st.markdown(
    "<h1 style='text-align:center;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#9CA3AF;'>"
    "Logistic Regression + TF-IDF based News Classification"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

news_text = st.text_area(
    "Enter News Article",
    height=220,
    placeholder="Paste news content here..."
)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        news_vec = tfidf.transform([news_text])
        prediction = model.predict(news_vec)[0]
        probability = model.predict_proba(news_vec)[0]

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box real">
                ✅ REAL NEWS <br>
                Confidence: {probability[1]*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box fake">
                ❌ FAKE NEWS <br>
                Confidence: {probability[0]*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

# ------------------------------
# Footer
# ------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#6B7280;'>"
    "Developed by <b>Naveen Kumar</b> | ML Project"
    "</p>",
    unsafe_allow_html=True
)
