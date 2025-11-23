import pandas as pd
import re
import numpy as np
import os
import csv
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC  # 必须用LinearSVC，否则还是慢
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec

# ----------------------
# 1. 文本预处理（不变）
# ----------------------
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now'
}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return ' '.join(words)

# ----------------------
# 2. 数据路径配置（不变）
# ----------------------
FILE_PATHS = {
    "positive_train": "positive_titles.csv",
    "negative_train": "negative_titles.csv",
    "test_set": "testSet-1000.xlsx"
}

# 检查文件存在性
for file_type, file_path in FILE_PATHS.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到{file_type}文件：{file_path}")

# ----------------------
# 3. 读取训练集（核心：只取10%样本）
# ----------------------
def load_train_dataset(file_path, encoding, label):
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        header=None,
        names=['title'],
        engine='python',
        on_bad_lines='skip',
        quoting=csv.QUOTE_NONE
    )
    df['label'] = label
    df['cleaned_title'] = df['title'].apply(preprocess_text)
    # 只取10%的样本（减少计算量）
    return df.sample(frac=0.1, random_state=42)  # 随机抽样10%

positive_train = load_train_dataset(FILE_PATHS["positive_train"], "latin-1", 1)
negative_train = load_train_dataset(FILE_PATHS["negative_train"], "latin-1", 0)
train_data = pd.concat([positive_train, negative_train], ignore_index=True)
print(f"⚠️ 已使用10%训练样本：共{len(train_data)}条（正例{len(positive_train)}，负例{len(negative_train)}）")

# ----------------------
# 4. 读取测试集（不变）
# ----------------------
test_excel = pd.read_excel(FILE_PATHS["test_set"])
required_columns = ['title given by manchine', 'Y/N', 'original title']
if not all(col in test_excel.columns for col in required_columns):
    raise ValueError(f"❌ 测试集缺少列：{required_columns}")

test_excel['label'] = test_excel['Y/N'].str.strip().str.upper().map({'Y': 1, 'N': 0})
test_excel['cleaned_title'] = test_excel['title given by manchine'].apply(preprocess_text)

# ----------------------
# 5. 朴素贝叶斯（不变）
# ----------------------
print("\n【模型1：朴素贝叶斯分类器】")
vectorizer = CountVectorizer()
X_train_nb = vectorizer.fit_transform(train_data['cleaned_title'])
X_test_nb = vectorizer.transform(test_excel['cleaned_title'])
y_train_nb, y_test_nb = train_data['label'], test_excel['label']

nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train_nb)
y_pred_nb = nb_model.predict(X_test_nb)

print("📊 混淆矩阵：")
print(confusion_matrix(y_test_nb, y_pred_nb))
print("\n📋 分类报告：")
print(classification_report(y_test_nb, y_pred_nb))

# ----------------------
# 6. Word2Vec + LinearSVC（快速版）
# ----------------------
print("\n【模型2：Word2Vec + LinearSVC分类器】")
start_time = time.time()

# 训练Word2Vec（基于10%样本）
print("  - 训练Word2Vec...")
train_corpus = [text.split() for text in train_data['cleaned_title'] if text.strip()]
w2v_model = Word2Vec(
    sentences=train_corpus,
    vector_size=30,
    window=3,
    min_count=1,  # 允许低频词（样本少了）
    workers=4,
    epochs=1
)

# 文本转向量
print("  - 转换文本向量...")
def get_text_vector(text):
    words = text.split()
    valid_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(30)

X_train_svm = np.array([get_text_vector(t) for t in train_data['cleaned_title']])
X_test_svm = np.array([get_text_vector(t) for t in test_excel['cleaned_title']])
y_train_svm, y_test_svm = train_data['label'], test_excel['label']

# 训练LinearSVC（必须用这个，否则还是慢）
print("  - 训练LinearSVC...")
svm_model = LinearSVC(max_iter=500)
svm_model.fit(X_train_svm, y_train_svm)

# 预测结果
y_pred_svm = svm_model.predict(X_test_svm)
print("\n📊 混淆矩阵：")
print(confusion_matrix(y_test_svm, y_pred_svm))
print("\n📋 分类报告：")
print(classification_report(y_test_svm, y_pred_svm))

print(f"\n总耗时：{time.time() - start_time:.2f}秒")