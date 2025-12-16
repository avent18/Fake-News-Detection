📰 Fake News Detection System

A machine learning–based web application that classifies news articles as Fake or Real using Logistic Regression and TF-IDF vectorization, deployed with a modern dark-mode Streamlit UI.


🚀 Features

🔍 Real-time news classification

🧠 Logistic Regression model for binary classification

📊 TF-IDF Vectorizer for text feature extraction

🌙 Dark mode UI (GitHub-style)

📈 Confidence score for each prediction

⚡ Fast & lightweight inference

🎯 Clean, interview-ready architecture



🧠 Machine Learning Approach

Text Vectorization: TF-IDF (Term Frequency–Inverse Document Frequency)

Model: Logistic Regression

Problem Type: Binary Classification (Fake / Real)

Input: Raw news article text

Output: News authenticity + confidence score



| Layer         | Technology           |
| ------------- | -------------------- |
| Frontend      | Streamlit (Dark UI)  |
| ML Model      | Logistic Regression  |
| NLP           | TF-IDF Vectorization |
| Language      | Python               |
| Model Storage | Pickle               |



Fake-News-Detection/
│
├── app.py                    # Streamlit application
├── logistic_model.pkl        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation



⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py
