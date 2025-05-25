# 🤖 IMDB Sentiment Analysis with BERT

A powerful Natural Language Processing (NLP) pipeline that classifies IMDB movie reviews as **positive** or **negative** using **BERT (Bidirectional Encoder Representations from Transformers)**. This project demonstrates an advanced, real-world application of transformer-based deep learning to understand human sentiment with near-human accuracy.

---

## 🎯 Project Summary

This project focuses on building an intelligent sentiment classification system that processes thousands of IMDB movie reviews and determines whether they express positive or negative sentiment. It highlights the contrast between traditional machine learning models and modern deep learning architectures like **BERT**, showing a significant performance leap.

---

## 🧠 Key Highlights

- 🧹 Advanced text preprocessing using custom NLP pipelines
- 🛠️ Comparative modeling: **Multinomial Naive Bayes (86.77%)** vs. **BERT (98.77%)**
- 🧾 Leveraged Hugging Face Transformers for fine-tuning `bert-base-uncased`
- 📊 Achieved **98.77% accuracy** on the IMDB dataset
- 📦 End-to-end implementation including training, evaluation, and error analysis
- 📉 Used confusion matrix and classification report for model validation

---

## 🚀 Technologies & Tools

| Category               | Tools / Libraries                          |
|------------------------|--------------------------------------------|
| Language               | Python                                     |
| Traditional ML         | scikit-learn, CountVectorizer              |
| Deep Learning / NLP    | Hugging Face Transformers, PyTorch         |
| Preprocessing          | NLTK, TextBlob, Regex, HTML stripping      |
| Model Evaluation       | Accuracy, Confusion Matrix, Classification Report |
| Datasets               | IMDB Movie Reviews (50,000+ samples)       |
| Visualization          | Matplotlib, Seaborn (optional)             |

---

## 📌 Project Workflow

### 1. Data Collection
- Loaded over 50,000 movie reviews from IMDB dataset.
- Ensured class balance (positive/negative).

### 2. Preprocessing Pipeline
- Lowercased reviews
- Removed HTML tags, URLs, and punctuation
- Tokenized text using BERT tokenizer (for deep learning path)
- Optional: Text correction, stemming, and lemmatization (for baseline model)

### 3. Feature Engineering
- **Traditional Path**: Used `CountVectorizer` with stopword removal
- **BERT Path**: Tokenized inputs with `bert-base-uncased`, converted into attention masks and input IDs

### 4. Modeling
- **Baseline**: Trained `MultinomialNB` for initial benchmarking
- **Advanced**: Fine-tuned a pre-trained BERT model using Hugging Face and PyTorch

### 5. Evaluation
- Accuracy Comparison:
  - ✅ Multinomial Naive Bayes: **86.77%**
  - ✅ BERT: **98.77%**
- Evaluation metrics included accuracy, precision, recall, F1-score, and confusion matrix

---

## 📈 Performance Comparison

| Model                   | Accuracy   |
|-------------------------|------------|
| Multinomial Naive Bayes | 86.77%     |
| BERT (Fine-tuned)       | **98.77%** |

---

## 🔍 Real-World Applications

- 🎥 **Entertainment Analytics**: Quickly gauge public opinion on film releases
- 🛒 **E-commerce**: Analyze customer reviews for product feedback
- 📱 **Social Listening**: Monitor brand sentiment across platforms
- ✉️ **Email Filtering**: Adapt pipeline for spam/ham classification
- 🧠 **Customer Support AI**: Detect sentiment in user queries or feedback

---

## 🔧 Future Enhancements

- 🧪 Hyperparameter tuning and model ensembling
- 🕵️‍♂️ Explainable AI: Use SHAP or LIME for model interpretability
- 📊 Deploy as an API using FastAPI or Flask
- 📱 Build a frontend using React for real-time review classification
- 💬 Expand to multi-class sentiment (positive/neutral/negative)

---


