import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 内置停用词列表
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
}

negation_words = {'not', 'no', 'never', 'nor', 'none', 'nothing', 'nowhere', 'neither', 'nobody'}
stop_words = STOPWORDS - negation_words

def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return tokens

print("=== Step 1: Loading data ===")
labeled_df = pd.read_csv('../labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
unlabeled_df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
test_df = pd.read_csv('../testData.tsv/testData.tsv', sep='\t')

print(f"Labeled data: {labeled_df.shape}")
print(f"Unlabeled data: {unlabeled_df.shape}")
print(f"Test data: {test_df.shape}")

print("\n=== Step 2: Preprocessing text ===")
labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
test_df['tokens'] = test_df['review'].apply(preprocess_text)

print("\n=== Step 3: Training Word2Vec ===")
all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=5)
print(f"Word2Vec vocabulary size: {len(model.wv)}")

print("\n=== Step 4: Converting to vectors ===")
def sentence_to_vector(tokens):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

X = np.array([sentence_to_vector(tokens) for tokens in labeled_df['tokens']])
y = labeled_df['sentiment'].values
X_test = np.array([sentence_to_vector(tokens) for tokens in test_df['tokens']])

print(f"Training features: {X.shape}")
print(f"Test features: {X_test.shape}")

print("\n=== Step 5: Training Logistic Regression ===")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_train, y_train)

print("\n=== Step 6: Evaluating ===")
y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {auc_score:.4f}")

print("\n=== Step 7: Predicting on test ===")
test_df['sentiment'] = clf.predict_proba(X_test)[:, 1]

print("\n=== Step 8: Saving submission ===")
submission = test_df[['id', 'sentiment']]
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")
print("\nSample output:")
print(submission.head())
