# Toxic Comment Classification with Gradio 
🧠💬
This project leverages Natural Language Processing (NLP) and Deep Learning (BiLSTM) to detect toxic comments in online platforms. It uses a multi-label classification approach to identify six categories of toxicity and deploys an interactive user interface using Gradio.
## Data Link Details: https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge?select=train.csv
# 🔍 Features
Detects toxicity across 6 categories:

Toxic

Severe Toxic

Obscene

Threat

Insult

Identity Hate

Preprocessing with NLTK: Stopword removal, lemmatization, and punctuation cleaning.

Real-time user interface using Gradio.

Trained on Jigsaw Toxic Comment Classification Challenge Dataset.

# 📦 Requirements
## Hardware
Processor: Intel i5/i7 or equivalent

RAM: Minimum 8 GB (Recommended 16 GB)

Storage: 10 GB free space

GPU: (Optional) NVIDIA with CUDA support

## Software
Python 3.8+

Libraries:

TensorFlow

Keras

NLTK

Gradio

NumPy

Pandas

Scikit-learn

Matplotlib

# 🚀 Getting Started
## 1. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
OR manually install:

bash
Copy
Edit
pip install tensorflow keras nltk gradio pandas numpy scikit-learn matplotlib
## 2. Download NLTK Resources
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
## 3. Run the App
bash
Copy
Edit
python toxic_comment_classifier.py
Gradio will start a local server or a public link for live testing.

# 🧠 Model
The model is a Bidirectional LSTM designed for multi-label classification with a sigmoid output for each class.

python
Copy
Edit
model = Sequential([
    Embedding(input_dim=20000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='sigmoid')
])
# 📊 Results
High accuracy on common classes like toxic and obscene.

Challenges with imbalanced classes like threat.

Real-time prediction interface using Gradio.

# 🛠 Future Enhancements
Upgrade to transformer models (e.g., BERT).

Multilingual support.

Improve model interpretability using SHAP or LIME.

Host backend using Flask/FastAPI.

Apply class balancing techniques.
