import pandas as pd
import numpy as np

# 最小化测试 - 直接生成一个简单的提交文件
print("Loading test data...")
test_df = pd.read_csv('../testData.tsv/testData.tsv', sep='\t')
print(f"Test data shape: {test_df.shape}")

# 生成随机预测（0-1之间的小数）
np.random.seed(42)
test_df['sentiment'] = np.random.rand(len(test_df))

# 保存提交文件
submission = test_df[['id', 'sentiment']]
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")
print(submission.head())
