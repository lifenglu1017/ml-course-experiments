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

# 下载NLTK数据
nltk.download('punkt')
nltk.download('stopwords')

# 加载停用词，保留否定词
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'nor', 'none', 'nothing', 'nowhere', 'neither', 'nobody'}
stop_words = stop_words - negation_words

def preprocess_text(text):
    """文本预处理函数"""
    # 去除HTML标签
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 转小写
    text = text.lower()
    
    # 去除标点和特殊字符，保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词
    tokens = word_tokenize(text)
    
    # 去除停用词
    tokens = [token for token in tokens if token not in stop_words]
    
    # 去除单字符（除了否定词）
    tokens = [token for token in tokens if len(token) > 1 or token in negation_words]
    
    return tokens

def load_data():
    """加载数据"""
    print("正在加载数据...")
    labeled_path = r'../labeledTrainData.tsv/labeledTrainData.tsv'
    unlabeled_path = r'unlabeledTrainData.tsv'
    test_path = r'../testData.tsv/testData.tsv'
    
    labeled_df = pd.read_csv(labeled_path, sep='\t')
    # 处理无标签数据的格式问题
    unlabeled_df = pd.read_csv(unlabeled_path, sep='\t', on_bad_lines='skip')
    test_df = pd.read_csv(test_path, sep='\t')
    
    print(f"标签数据: {labeled_df.shape}")
    print(f"无标签数据: {unlabeled_df.shape}")
    print(f"测试数据: {test_df.shape}")
    
    return labeled_df, unlabeled_df, test_df

def train_word2vec(sentences, vector_size=100, window=5, min_count=10, epochs=10):
    """训练Word2Vec模型"""
    print(f"\n正在训练Word2Vec模型...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs
    )
    print(f"Word2Vec训练完成，词汇表大小: {len(model.wv)}")
    return model

def sentence_to_vector(tokens, model, vector_size=100):
    """将句子转换为向量（词向量求平均）"""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)

def main():
    # 1. 加载数据
    labeled_df, unlabeled_df, test_df = load_data()
    
    # 2. 预处理文本
    print("\n正在预处理文本...")
    labeled_df['tokens'] = labeled_df['review'].apply(preprocess_text)
    unlabeled_df['tokens'] = unlabeled_df['review'].apply(preprocess_text)
    test_df['tokens'] = test_df['review'].apply(preprocess_text)
    
    # 3. 准备训练Word2Vec的数据（结合有标签和无标签数据）
    all_sentences = list(labeled_df['tokens']) + list(unlabeled_df['tokens'])
    
    # 4. 训练Word2Vec模型
    model = train_word2vec(all_sentences)
    
    # 5. 将文本转换为向量
    print("\n正在转换文本为向量...")
    vector_size = 100
    X = np.array([sentence_to_vector(tokens, model, vector_size) for tokens in labeled_df['tokens']])
    y = labeled_df['sentiment'].values
    X_test = np.array([sentence_to_vector(tokens, model, vector_size) for tokens in test_df['tokens']])
    
    # 6. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # 7. 训练逻辑回归模型
    print("\n正在训练逻辑回归模型...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    
    # 8. 在验证集上评估
    y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred_proba)
    print(f"验证集AUC: {auc_score:.4f}")
    
    # 9. 在测试集上预测
    print("\n正在预测测试集...")
    test_df['sentiment'] = clf.predict_proba(X_test)[:, 1]
    
    # 10. 生成提交文件
    submission = test_df[['id', 'sentiment']]
    submission.to_csv('submission.csv', index=False)
    print("\n提交文件已生成: submission.csv")
    print(f"样本输出:\n{submission.head()}")
    
    return auc_score

if __name__ == '__main__':
    main()
