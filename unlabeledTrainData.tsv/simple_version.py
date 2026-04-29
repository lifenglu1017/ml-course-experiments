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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = re.findall(r'[a-zA-Z]+', text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 1 or token in negation_words]
    return tokens

# 加载数据
print("Loading data...")
labeled_df = pd.read_csv('../labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
unlabeled_df = pd.read_csv('unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
test_df = pd.read_csv('../testData.tsv/testData.tsv', sep='\t')

# 预处理
print("Preprocessing...")
labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
test_df['tokens'] = test_df['review'].apply(preprocess_text)

# 训练Word2Vec
print("Training Word2Vec...")
all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=10, workers=4, epochs=10)

# 转换为向量
print("Converting to vectors...")
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

# 训练逻辑回归
print("Training Logistic Regression...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_train, y_train)

# 评估
y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation AUC: {auc_score:.4f}")

# 预测测试集
print("Predicting on test...")
test_df['sentiment'] = clf.predict_proba(X_test)[:, 1]

# 保存结果
submission = test_df[['id', 'sentiment']]
submission.to_csv('submission.csv', index=False)
print("Submission saved!")
print(submission.head())
