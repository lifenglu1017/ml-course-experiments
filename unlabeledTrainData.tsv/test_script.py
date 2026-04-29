import pandas as pd

# 测试数据加载
try:
    labeled_path = r'../labeledTrainData.tsv/labeledTrainData.tsv'
    labeled_df = pd.read_csv(labeled_path, sep='\t')
    print(f"标签数据加载成功: {labeled_df.shape}")
    print(labeled_df.head())
except Exception as e:
    print(f"加载标签数据失败: {e}")

try:
    unlabeled_path = r'unlabeledTrainData.tsv'
    unlabeled_df = pd.read_csv(unlabeled_path, sep='\t')
    print(f"\n无标签数据加载成功: {unlabeled_df.shape}")
    print(unlabeled_df.head())
except Exception as e:
    print(f"加载无标签数据失败: {e}")

try:
    test_path = r'../testData.tsv/testData.tsv'
    test_df = pd.read_csv(test_path, sep='\t')
    print(f"\n测试数据加载成功: {test_df.shape}")
    print(test_df.head())
except Exception as e:
    print(f"加载测试数据失败: {e}")
