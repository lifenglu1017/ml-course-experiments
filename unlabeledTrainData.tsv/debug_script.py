import sys

def log(message):
    print(message)
    sys.stdout.flush()

log("Step 1: Importing libraries...")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

log("Step 2: Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    log("NLTK data downloaded successfully")
except Exception as e:
    log(f"Error downloading NLTK data: {e}")
    sys.exit(1)

log("Step 3: Loading stopwords...")
try:
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'nor', 'none', 'nothing', 'nowhere', 'neither', 'nobody'}
    stop_words = stop_words - negation_words
    log(f"Loaded {len(stop_words)} stopwords")
except Exception as e:
    log(f"Error loading stopwords: {e}")
    sys.exit(1)

log("Step 4: Defining preprocessing function...")
def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 1 or token in negation_words]
    return tokens

log("Step 5: Loading data...")
try:
    labeled_path = r'../labeledTrainData.tsv/labeledTrainData.tsv'
    unlabeled_path = r'unlabeledTrainData.tsv'
    test_path = r'../testData.tsv/testData.tsv'

    labeled_df = pd.read_csv(labeled_path, sep='\t')
    log(f"Labeled data shape: {labeled_df.shape}")

    unlabeled_df = pd.read_csv(unlabeled_path, sep='\t', on_bad_lines='skip')
    log(f"Unlabeled data shape: {unlabeled_df.shape}")

    test_df = pd.read_csv(test_path, sep='\t')
    log(f"Test data shape: {test_df.shape}")
except Exception as e:
    log(f"Error loading data: {e}")
    sys.exit(1)

log("Step 6: Preprocessing text...")
try:
    labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
    log("Labeled data preprocessed")
    
    unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
    log("Unlabeled data preprocessed")
    
    test_df['tokens'] = test_df['review'].apply(preprocess_text)
    log("Test data preprocessed")
except Exception as e:
    log(f"Error preprocessing text: {e}")
    sys.exit(1)

log("Step 7: Training Word2Vec...")
try:
    all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
    log(f"Total sentences for Word2Vec: {len(all_sentences)}")
    
    model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=10, workers=4, epochs=10)
    log(f"Word2Vec vocabulary size: {len(model.wv)}")
except Exception as e:
    log(f"Error training Word2Vec: {e}")
    sys.exit(1)

log("Step 8: Converting to vectors...")
try:
    vector_size = 100
    def sentence_to_vector(tokens):
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
        if len(vectors) == 0:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    X = np.array([sentence_to_vector(tokens) for tokens in labeled_df['tokens']])
    y = labeled_df['sentiment'].values
    log(f"Training features shape: {X.shape}")
    
    X_test = np.array([sentence_to_vector(tokens) for tokens in test_df['tokens']])
    log(f"Test features shape: {X_test.shape}")
except Exception as e:
    log(f"Error converting to vectors: {e}")
    sys.exit(1)

log("Step 9: Splitting data...")
try:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    log(f"Train: {X_train.shape}, Val: {X_val.shape}")
except Exception as e:
    log(f"Error splitting data: {e}")
    sys.exit(1)

log("Step 10: Training Logistic Regression...")
try:
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    log("Logistic Regression trained successfully")
except Exception as e:
    log(f"Error training Logistic Regression: {e}")
    sys.exit(1)

log("Step 11: Evaluating...")
try:
    y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    log(f"Validation AUC: {auc_score:.4f}")
except Exception as e:
    log(f"Error evaluating: {e}")
    sys.exit(1)

log("Step 12: Predicting on test...")
try:
    test_df['sentiment'] = clf.predict_proba(X_test)[:, 1]
    log("Test predictions completed")
except Exception as e:
    log(f"Error predicting on test: {e}")
    sys.exit(1)

log("Step 13: Saving submission...")
try:
    submission = test_df[['id', 'sentiment']]
    submission.to_csv('submission.csv', index=False)
    log("Submission file saved successfully!")
    log(f"Submission shape: {submission.shape}")
    log(f"First few rows:\n{submission.head().to_string()}")
except Exception as e:
    log(f"Error saving submission: {e}")
    sys.exit(1)

log("All steps completed successfully!")
