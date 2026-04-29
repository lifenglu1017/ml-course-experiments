print("Step 1: Importing libraries...")
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

print("Step 2: Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')

print("Step 3: Loading stopwords...")
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'nor', 'none', 'nothing', 'nowhere', 'neither', 'nobody'}
stop_words = stop_words - negation_words

print("Step 4: Defining preprocessing function...")
def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 1 or token in negation_words]
    return tokens

print("Step 5: Loading data...")
labeled_path = r'../labeledTrainData.tsv/labeledTrainData.tsv'
unlabeled_path = r'unlabeledTrainData.tsv'
test_path = r'../testData.tsv/testData.tsv'

labeled_df = pd.read_csv(labeled_path, sep='\t')
print(f"Labeled data shape: {labeled_df.shape}")

unlabeled_df = pd.read_csv(unlabeled_path, sep='\t', on_bad_lines='skip')
print(f"Unlabeled data shape: {unlabeled_df.shape}")

test_df = pd.read_csv(test_path, sep='\t')
print(f"Test data shape: {test_df.shape}")

print("Step 6: Preprocessing text...")
labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
test_df['tokens'] = test_df['review'].apply(preprocess_text)

print("Step 7: Training Word2Vec...")
all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=10, workers=4, epochs=10)
print(f"Word2Vec vocabulary size: {len(model.wv)}")

print("Step 8: Converting to vectors...")
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
X_test = np.array([sentence_to_vector(tokens) for tokens in test_df['tokens']])

print("Step 9: Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

print("Step 10: Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_train, y_train)

print("Step 11: Evaluating...")
y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {auc_score:.4f}")

print("Step 12: Predicting on test...")
test_df['sentiment'] = clf.predict_proba(X_test)[:, 1]

print("Step 13: Saving submission...")
submission = test_df[['id', 'sentiment']]
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")
print(submission.head())
