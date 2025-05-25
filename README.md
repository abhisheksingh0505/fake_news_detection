# 🕵️ Fake News Detection

[![GitHub Stars](https://img.shields.io/github/stars/abhisheksingh0505/fake_news_detection?style=social)](https://github.com/abhisheksingh0505/fake_news_detection/stargazers)
[![Forks](https://img.shields.io/github/forks/abhisheksingh0505/fake_news_detection?style=social)](https://github.com/abhisheksingh0505/fake_news_detection/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/abhisheksingh0505/fake_news_detection)](https://github.com/abhisheksingh0505/fake_news_detection/commits/main)

A machine learning-powered web application that classifies news articles as **Real** or **Fake** using Natural Language Processing techniques. Built with Streamlit and scikit-learn.



## 🌟 Features

- **Real-time Classification**: Instant analysis of news article authenticity
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Confidence Scoring**: Probability breakdown for predictions
- **Text Analysis**: Detailed preprocessing and feature importance insights
- **Example Articles**: Pre-loaded samples for testing
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Live Demo

[((http://192.168.43.232:8501))]

## 📊 Model Performance

- **Algorithm**: Logistic Regression
- **Training Data**: ~42,000 news articles (21k fake + 21k real)
- **Features**: TF-IDF vectorization with 5,000 features
- **Preprocessing**: Text cleaning, stopword removal, lemmatization
- **Accuracy**: [Accuracy: 0.9879 and F1 Score: 0.9875]

## 🏗️ Project Structure

```
streamlit-fake-news/

├── datasets/                        # Training datasets
│   ├── Fake.csv                     # Fake news dataset
│   └── True.csv                     # Real news dataset
├── models/                          # Saved model artifacts
│   ├── model.pkl                    # Trained classification model
│   └── tfidf.pkl                    # TF-IDF vectorizer
├── venv/                            # Virtual environment (ignored)
├── .git/                            # Git repository
├── .gitignore                       # Git ignore file
├── app.py                           # Main Streamlit application
├── fake_news_classifier.ipynb       # Jupyter notebook (model training)
├──Fake_news_detection_project_report # Project Report    
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies


```



## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/abhisheksingh0505/fake_news_detection.git
cd fake-news-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```


## 📦 Dependencies

```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
nltk>=3.8.0
pickle-mixin>=1.0.2
```

## 🔬 How It Works

### 1. **Data Collection**
- **Fake News Dataset**: ~21,000 articles from unreliable sources
- **Real News Dataset**: ~21,000 articles from credible news sources
- **Source**: Kaggle Fake and Real News Dataset

### 2. **Text Preprocessing**
```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in text.split() 
             if word not in stop_words]
    
    return ' '.join(words)
```

### 3. **Feature Extraction**
- **TF-IDF Vectorization**: Convert text to numerical features
- **Max Features**: 5,000 most important words
- **Dimensionality**: High-dimensional sparse matrix

### 4. **Model Training**
- **Algorithm**: Logistic Regression with L2 regularization
- **Train/Test Split**: 80/20 ratio
- **Evaluation Metrics**: Accuracy, F1-score, Precision, Recall

### 5. **Prediction Pipeline**
1. User inputs news article text
2. Text preprocessing (cleaning, lemmatization)
3. TF-IDF vectorization
4. Model prediction with confidence scores
5. Results display with feature importance

## 🎯 Usage Examples

### Basic Usage

1. **Launch the App**: Run `streamlit run app.py`
2. **Input Text**: Paste a news article in the text area
3. **Analyze**: Click the "🔍 Analyze Article" button
4. **Review Results**: Check the prediction and confidence score

### Example Article (Fake News)

```
BREAKING: Scientists Discover That Drinking Coffee Backwards 
Can Reverse Aging Process! Local man looks 20 years younger 
after following this one weird trick that doctors hate!
```

### Example Article (Real News)

```
Researchers at Stanford University published a study in Nature 
showing that regular exercise can improve cognitive function 
in older adults. The six-month clinical trial involved 200 
participants aged 65-80 and measured improvements in memory 
and attention span.
```

## 📈 Model Evaluation

### Training Results
- **Training Accuracy**: [Insert accuracy]%
- **Test Accuracy**: [Insert accuracy]%
- **F1-Score**: [Insert F1-score]
- **Precision**: [Insert precision]%
- **Recall**: [Insert recall]%

### Confusion Matrix
                Predicted
              Fake    Real
Actual Fake   4,187    53
       Real      49  4,191
```

## 🔍 Feature Importance

The model identifies key linguistic patterns that distinguish fake from real news:

**Words Suggesting Fake News:**
- Sensationalist language
- Emotional triggers
- Clickbait phrases
- Unverified claims

**Words Suggesting Real News:**
- Factual reporting terms
- Attribution phrases
- Professional journalism language
- Source citations

## 🚨 Limitations & Disclaimers

⚠️ **Important Notes:**

- **Not 100% Accurate**: This is a machine learning model with inherent limitations
- **Training Bias**: Performance depends on training data quality and representation
- **Context Matters**: Cannot verify factual accuracy, only predicts based on writing patterns
- **Always Verify**: Cross-check information with multiple reliable sources
- **Educational Purpose**: Designed for learning and demonstration, not professional fact-checking

## 🛡️ Ethical Considerations

- **Responsible Use**: This tool should supplement, not replace, critical thinking
- **Source Verification**: Always check the credibility of news sources
- **Media Literacy**: Encourage users to develop critical evaluation skills
- **Transparency**: Model limitations and biases are clearly communicated

## 🔄 Future Improvements

- [ ] **Deep Learning Models**: Experiment with BERT, RoBERTa, or other transformers
- [ ] **Real-time Data**: Integration with live news feeds
- [ ] **Multi-language Support**: Expand beyond English articles
- [ ] **Source Analysis**: Incorporate domain reputation scoring
- [ ] **Explainable AI**: Better interpretation of model decisions
- [ ] **Batch Processing**: Upload and analyze multiple articles
- [ ] **API Integration**: RESTful API for external applications

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/new-feature`
3. **Make Changes**: Implement your improvements
4. **Add Tests**: Ensure functionality works correctly
5. **Commit Changes**: `git commit -m "Add new feature"`
6. **Push to Branch**: `git push origin feature/new-feature`
7. **Submit Pull Request**: Describe your changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/abhisheksingh0505/fake_news_detection.git
cd fake-news-detector

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install jupyter notebook  # for model development

# Run tests (if available)
python -m pytest tests/
```



## 👥 Authors

- **Abhishek Singh** - *Initial work* - [YourGitHub](https://github.com/abhisheksingh0505/fake_news_detection)

## 🙏 Acknowledgments

- **Dataset**: Kaggle Fake and Real News Dataset contributors
- **Libraries**: scikit-learn, NLTK, Streamlit communities
- **Inspiration**: Growing need for media literacy tools
- **Research**: Academic papers on fake news detection

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/abhisheksingh0505/fake_news_detection)
- **Email**: sing050530@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/abhishek-singh-139181279)

## 📚 References

1. "The spread of true and false news online" - Science, 2018
2. "Natural Language Processing with Python" - NLTK Documentation
3. "scikit-learn: Machine Learning in Python" - Journal of Machine Learning Research


⭐ **If you find this project helpful, please [give it a star](https://github.com/abhisheksingh0505/fake_news_detection/stargazers)!** ⭐

[![Fake News Detection](https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Fake+News+Detector+🕵️)](https://github.com/abhisheksingh0505/fake_news_detection/stargazers)

---

*Built with ❤️ for media literacy and education*
