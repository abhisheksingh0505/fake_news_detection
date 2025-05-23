import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

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
    try:
        # Try loading from models folder first
        model_path = os.path.join('models', 'model.pkl')
        tfidf_path = os.path.join('models', 'tfidf_vectorizer.pkl')  # Updated filename
        
        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(tfidf_path, 'rb') as f:
                tfidf = pickle.load(f)
        else:
            # Fallback to current directory - try both possible filenames
            model_files = ['model.pkl']
            tfidf_files = ['tfidf_vectorizer.pkl', 'tfidf.pkl']
            
            # Load model
            model = None
            for model_file in model_files:
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    break
            
            # Load TF-IDF vectorizer
            tfidf = None
            for tfidf_file in tfidf_files:
                if os.path.exists(tfidf_file):
                    with open(tfidf_file, 'rb') as f:
                        tfidf = pickle.load(f)
                    break
            
            if model is None:
                raise FileNotFoundError("Model file not found")
            if tfidf is None:
                raise FileNotFoundError("TF-IDF vectorizer file not found")
        
        return model, tfidf
    
    except FileNotFoundError as e:
        st.error("⚠️ Model files not found!")
        st.error("Please ensure the following files are available:")
        st.error("- `model.pkl` (your trained model)")
        st.error("- `tfidf_vectorizer.pkl` or `tfidf.pkl` (your TF-IDF vectorizer)")
        st.error("Place them in a 'models' folder or in the same directory as app.py")
        st.error(f"Error details: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, tfidf = load_models()

# Preprocessing setup
@st.cache_resource
def setup_preprocessing():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

lemmatizer, stop_words = setup_preprocessing()

def preprocess_text(text):
    """
    Preprocess text exactly as done during training:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenize by splitting on whitespace
    4. Remove stopwords and lemmatize
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Streamlit UI
st.title("🕵️ Fake News Detector")
st.markdown("""
### Detect whether a news article is **Real** or **Fake** using Machine Learning

This app uses a **Logistic Regression** model trained on approximately **40,000** news articles 
to classify news content. The model analyzes text patterns, language style, and other linguistic 
features to make predictions.

**How it works:**
- Text preprocessing (cleaning, removing stopwords, lemmatization)
- TF-IDF vectorization (converting text to numerical features)
- Machine learning classification using Logistic Regression
""")

st.markdown("---")

# Input section
st.subheader("📰 Enter News Article")
user_input = st.text_area(
    "Paste the news article content here:", 
    height=200,
    placeholder="Enter the full text of the news article you want to analyze..."
)

# Add example button
if st.button("📝 Load Example Article"):
    example_text = """
    Scientists at the University of California have announced a groundbreaking discovery in renewable energy technology. 
    The new solar panel design can convert sunlight to electricity with 40% efficiency, nearly double that of current 
    commercial panels. The research team, led by Dr. Sarah Johnson, published their findings in the journal Nature Energy. 
    The technology uses a novel combination of perovskite and silicon materials to capture a broader spectrum of light. 
    Industry experts believe this advancement could significantly reduce the cost of solar power and accelerate the 
    transition to clean energy. The university has already filed patents for the technology and is seeking partnerships 
    with major solar manufacturers for commercial development.
    """
    st.text_area("Example loaded:", value=example_text, height=200, key="example")

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze Article", type="primary")
with col2:
    if st.button("🗑️ Clear"):
        st.rerun()

if analyze_button:
    if user_input.strip():
        try:
            with st.spinner("Analyzing article..."):
                # Preprocess and predict
                cleaned_text = preprocess_text(user_input)
                vectorized = tfidf.transform([cleaned_text])
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                
                # Calculate confidence
                confidence = max(proba) * 100
                real_prob = proba[1] * 100  # Probability of being real
                fake_prob = proba[0] * 100  # Probability of being fake
            
            # Display results with better styling
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Main prediction display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 1:
                    st.success("### ✅ REAL NEWS")
                    st.success(f"**Confidence: {confidence:.1f}%**")
                else:
                    st.error("### ❌ FAKE NEWS")
                    st.error(f"**Confidence: {confidence:.1f}%**")
            
            with col2:
                st.subheader("Probability Breakdown:")
                st.write(f"🟢 **Real News:** {real_prob:.1f}%")
                st.write(f"🔴 **Fake News:** {fake_prob:.1f}%")
                
                # Progress bars for visual representation
                st.progress(real_prob / 100, text=f"Real: {real_prob:.1f}%")
                st.progress(fake_prob / 100, text=f"Fake: {fake_prob:.1f}%")
            
            # Text analysis insights
            st.markdown("---")
            st.subheader("🔬 Text Analysis Insights")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Article Statistics:**")
                word_count = len(user_input.split())
                cleaned_word_count = len(cleaned_text.split())
                st.write(f"- Original word count: {word_count}")
                st.write(f"- Processed word count: {cleaned_word_count}")
                st.write(f"- Text reduction: {((word_count - cleaned_word_count) / word_count * 100):.1f}%")
            
            with col2:
                st.write("**Model Features:**")
                st.write("- Preprocessing: Stopword removal, lemmatization")
                st.write("- Vectorization: TF-IDF with 5000 features")
                st.write("- Algorithm: Logistic Regression")
                st.write(f"- Training data: ~40,000 articles")
            
            # Show top predictive words if model has coefficients
            if hasattr(model, 'coef_') and hasattr(tfidf, 'get_feature_names_out'):
                st.markdown("---")
                st.subheader("🎯 Most Influential Words")
                
                try:
                    feature_names = tfidf.get_feature_names_out()
                    coefs = model.coef_[0]
                    
                    # Get top words that suggest fake news (most negative coefficients)
                    top_fake_indices = coefs.argsort()[:10]
                    top_fake_words = [(feature_names[i], coefs[i]) for i in top_fake_indices]
                    
                    # Get top words that suggest real news (most positive coefficients)
                    top_real_indices = coefs.argsort()[-10:][::-1]
                    top_real_words = [(feature_names[i], coefs[i]) for i in top_real_indices]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**🚩 Words suggesting FAKE news:**")
                        for word, score in top_fake_words:
                            st.write(f"- {word} (weight: {score:.3f})")
                    with col2:
                        st.write("**✅ Words suggesting REAL news:**")
                        for word, score in top_real_words:
                            st.write(f"- {word} (weight: {score:.3f})")
                except Exception as e:
                    st.write("Could not display feature importance analysis.")
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check that your model and vectorizer files are compatible.")
    else:
        st.warning("⚠️ Please enter some text to analyze")

# Sidebar with additional information
st.sidebar.header("ℹ️ About This Model")
st.sidebar.markdown("""
**Dataset:** Fake and Real News Dataset

**Training Details:**
- Fake news articles: ~21,000
- Real news articles: ~21,000
- Total training samples: ~42,000

**Preprocessing Steps:**
1. Text cleaning (remove special characters)
2. Lowercase conversion
3. Stopword removal
4. Lemmatization
5. TF-IDF vectorization (5000 features)

**Model Performance:**
- Algorithm: Logistic Regression
- Evaluation: Train/Test split (80/20)
- Metrics: Accuracy, F1-score, Precision, Recall
""")

st.sidebar.markdown("---")
st.sidebar.header("📚 How to Use")
st.sidebar.markdown("""
1. **Paste Article:** Copy and paste the full text of a news article
2. **Click Analyze:** Press the analyze button to get predictions
3. **Review Results:** Check the prediction and confidence score
4. **Interpret:** Higher confidence = more certain prediction
""")

st.sidebar.markdown("---")
st.sidebar.header("⚠️ Important Disclaimer")
st.sidebar.markdown("""
This tool is for educational purposes only. 

**Always:**
- Verify information from multiple sources
- Check the original source credibility
- Consider the publication date
- Use critical thinking

**Remember:** No AI model is 100% accurate!
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Fake News Detector</strong> | Built with Streamlit & Scikit-learn</p>
    <p>⚠️ This is a machine learning model and may not be 100% accurate. Always verify information from multiple reliable sources.</p>
</div>
""", unsafe_allow_html=True)

# Debug info (optional - you can remove this in production)
with st.expander("Debug Info"):
    st.write("Current working directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir('.'))
    if os.path.exists('models'):
        st.write("Files in models folder:", os.listdir('models'))
    else:
        st.write("Models folder does not exist")