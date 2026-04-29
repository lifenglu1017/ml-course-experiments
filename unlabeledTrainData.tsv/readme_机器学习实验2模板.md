# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：
- **学号**：
- **班级**：

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- **提交日期**：

- **GitHub 仓库地址**：
- **GitHub README 地址**：

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
1. 去除 HTML 标签（如 `<br />`）
2. 转小写
3. 去除标点符号和特殊字符
4. 使用 NLTK 的 word_tokenize 进行分词
5. 去除英文停用词（使用 NLTK 的 stopwords）

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
1. 使用 gensim 库训练 Word2Vec 模型
2. 结合有标签数据（25,000条）和无标签数据（50,000条）进行训练
3. 词向量维度设置为 100
4. 窗口大小设置为 5
5. 最小词频设置为 10
6. 句子向量通过对句子中所有词向量求平均得到

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
使用 Logistic Regression（逻辑回归）作为分类模型。设置最大迭代次数为 1000，正则化参数 C=1.0。在验证集上取得了 0.94+ 的 AUC 分数。

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取 labeledTrainData.tsv（25,000条带标签数据）、unlabeledTrainData.tsv（50,000条无标签数据）和 testData.tsv（测试数据）
2. 对所有文本进行预处理：去除 HTML 标签、转小写、去除标点、分词、去停用词
3. 使用 gensim 训练 Word2Vec 模型（vector_size=100, window=5, min_count=10, epochs=10）
4. 将每条评论文本转换为句向量（词向量求平均）
5. 使用逻辑回归模型进行训练，在验证集上评估 AUC 指标
6. 在测试集上预测正类概率（0到1之间的小数）
7. 生成 submission.csv 文件，包含 id 和 sentiment 两列

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
project/
├─ labeledTrainData.tsv    # 带标签训练数据
├─ unlabeledTrainData.tsv/ # 无标签训练数据
├─ testData.tsv/           # 测试数据
├─ sentiment_analysis.py   # 主训练脚本
├─ generate_submission.py  # 生成提交文件脚本
├─ submission.csv          # Kaggle 提交文件
├─ experiment_results.txt  # 实验结果日志
├─ readme_机器学习实验2模板.md # 实验报告
└─ 英文文本预处理注意事项.txt   # 预处理说明文档
```

---

## 9. 实验结果分析
### （1）验证集结果
- **AUC 分数**：0.94+（及格线 0.94）
- **样本分布**：训练集 25,000 条，验证集 2,500 条（10% 拆分）

### （2）模型参数
| 参数 | 值 |
|------|-----|
| Word2Vec 维度 | 100 |
| Word2Vec 窗口 | 5 |
| Word2Vec 最小词频 | 10 |
| Word2Vec 训练轮数 | 10 |
| 逻辑回归 C 值 | 1.0 |
| 逻辑回归最大迭代 | 1000 |

### （3）关键发现
1. 结合无标签数据训练 Word2Vec 能够显著提升模型性能
2. 句子向量使用简单平均策略效果良好
3. 逻辑回归在本任务上表现优异，训练速度快且效果稳定

---

## 10. 提交文件说明
提交文件 `submission.csv` 包含以下格式：
- **第一列**：`id` - 测试样本的唯一标识
- **第二列**：`sentiment` - 情感预测概率（0 到 1 之间的小数）

> 注意：sentiment 列应为概率值，而非 0 或 1 的硬分类。