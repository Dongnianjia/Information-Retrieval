import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ===================== #
# Step 1. 数据加载（保留原始文本）
# ===================== #
def read_all_lines(path):
    """逐行读取文本文件，保留所有符号"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"title": lines})

pos_path = r"D:\homework\大三上\信息检索\作业2\positive_trainingSet.txt"
neg_path = r"D:\homework\大三上\信息检索\作业2\negative_trainingSet.txt"
test_path = r"D:\homework\大三上\信息检索\作业2\testSet-1000.xlsx"

# 分别读取正负样本
pos = read_all_lines(pos_path)
neg = read_all_lines(neg_path)

# 打标签
pos["label"] = 1
neg["label"] = 0

# 合并训练集
train_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)
print(f"[INFO] 成功加载训练集：正样本 {len(pos)} 条，负样本 {len(neg)} 条，总计 {len(train_df)} 条")

# ===================== #
# Step 1.5 测试集加载与逻辑修正
# ===================== #
test_df = pd.read_excel(test_path)

# 重命名列
test_df.rename(columns={"title given by manchine": "title"}, inplace=True)
test_df["label"] = test_df["Y/N"].map({"Y": 1, "N": 0})

# 若机器标题为空且 Y（正确），则用 original title 替代
missing_title_mask = test_df["title"].isna() & (test_df["label"] == 1)
if missing_title_mask.sum() > 0:
    print(f"[INFO] 发现 {missing_title_mask.sum()} 条 Y 样本缺失机器标题，用原始标题替换。")
    test_df.loc[missing_title_mask, "title"] = test_df.loc[missing_title_mask, "original title"]

# 仍有空标题（一般为错误样本），填充为空字符串防止崩溃
empty_count = test_df["title"].isna().sum()
if empty_count > 0:
    print(f"[WARN] 仍有 {empty_count} 条空标题（多为提取失败样本），已填充为空字符串。")
    test_df["title"] = test_df["title"].fillna("")

# 转为字符串类型
test_df["title"] = test_df["title"].astype(str)

# 打印前 5 行确认
print("\n[DEBUG] 测试集前 5 行样本：")
print(test_df.head()[["title", "original title", "Y/N", "label"]])

print(f"\n[INFO] 成功加载测试集，共 {test_df.shape[0]} 条样本")

# ===================== #
# Step 2. 向量化文本
# ===================== #
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, lowercase=True)
X_train = tfidf.fit_transform(train_df["title"])
y_train = train_df["label"]
X_test = tfidf.transform(test_df["title"])
y_test = test_df["label"]

# ===================== #
# Step 3. 训练 Naive Bayes 模型
# ===================== #
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# ===================== #
# Step 4. 结果评估
# ===================== #
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, digits=4))
print("=== 混淆矩阵 ===")
print(confusion_matrix(y_test, y_pred))

# ===================== #
# Step 5. t-SNE 可视化
# ===================== #
print("\n[INFO] 正在执行 t-SNE 降维可视化（可能较慢）...")
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_test.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='coolwarm', alpha=0.6)
plt.title("t-SNE Visualization of Test Titles (Red=Wrong, Blue=Correct)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

print("\n[INFO] 实验完成 ✅")
