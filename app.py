import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# ✅ This MUST be the very first Streamlit command
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Download NLTK data (move this after set_page_config)
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Call the download function
download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    return model, tfidf

model, tfidf = load_models()

# Preprocessing setup
@st.cache_resource
def setup_preprocessing():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

lemmatizer, stop_words = setup_preprocessing()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("🕵️ Fake News Detector")
st.write("""
This app predicts whether a news article is **real** or **fake** using Natural Language Processing.
The model was trained on a dataset of ~40,000 news articles.
""")

# Input
user_input = st.text_area("Paste the news article content here:", height=200)

if st.button("Analyze"):
    if user_input:
        # Preprocess and predict
        cleaned_text = preprocess_text(user_input)
        vectorized = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        
        # Display results
        st.subheader("Results")
        if prediction == 1:
            st.success("✅ This article appears to be REAL")
        else:
            st.error("❌ This article appears to be FAKE")
        
        st.write(f"Confidence: {max(proba)*100:.2f}%")
        
        # Explanation
        st.subheader("Analysis")
        st.write("""
        The model analyzed the text for:
        - Sensationalist language
        - Unusual word patterns
        - Credibility indicators
        - Writing style markers
        """)
        
        # Show top predictive words
        feature_names = tfidf.get_feature_names_out()
        coefs = model.coef_[0]
        top_fake = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:5]
        top_real = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:5]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("🚩 Words suggesting FAKE news:")
            for word, score in top_fake:
                st.write(f"- {word} (score: {score:.2f})")
        with col2:
            st.write("✅ Words suggesting REAL news:")
            for word, score in top_real:
                st.write(f"- {word} (score: {score:.2f})")
    else:
        st.warning("Please enter some text to analyze")

# Footer
st.markdown("---")
st.write("""
**Note**: This is a machine learning model and may not be 100% accurate. 
Always verify information from multiple reliable sources.
""")