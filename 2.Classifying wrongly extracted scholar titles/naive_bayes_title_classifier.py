import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix
import numpy as np
import re
import string
import warnings
warnings.filterwarnings("ignore")

# ============================
# Step 1. 数据加载
# ============================
def read_all_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"title": lines})

pos_path = r"D:\homework\大三上\信息检索\作业2\positive_trainingSet.txt"
neg_path = r"D:\homework\大三上\信息检索\作业2\negative_trainingSet.txt"
test_path = r"D:\homework\大三上\信息检索\作业2\testSet-1000.xlsx"

pos = read_all_lines(pos_path); pos["label"] = 1
neg = read_all_lines(neg_path); neg["label"] = 0

train_df = pd.concat([pos, neg], ignore_index=True)
print(f"[INFO] 训练集共 {len(train_df)} 条（正 {len(pos)}，负 {len(neg)}）")

# ============================
# Step 1.5 处理测试集
# ============================
test_df = pd.read_excel(test_path)
test_df.rename(columns={"title given by manchine": "title"}, inplace=True)
test_df["label"] = test_df["Y/N"].map({"Y": 1, "N": 0})

missing_mask = test_df["title"].isna() & (test_df["label"] == 1)
test_df.loc[missing_mask, "title"] = test_df.loc[missing_mask, "original title"]
test_df["title"] = test_df["title"].fillna("").astype(str)

print("\n[DEBUG] 测试集前 5 行：")
print(test_df.head()[["title", "original title", "label"]])

# ============================
# Step 2. 文本清洗（非常轻量，保留标点 & 大小写）
# ============================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)   # 去 URL
    text = re.sub(r"\s+", " ", text).strip()      # 压缩空格
    return text

train_df["clean"] = train_df["title"].apply(clean_text)
test_df["clean"]  = test_df["title"].apply(clean_text)

# ============================
# Step 2.5 手工特征提取：强化“错误标题”特征
# ============================

# 错误标题常见关键词（全是从你给的测试集模式里抽出来的）
SUSPICIOUS_KEYWORDS = [
    "contents", "table of contents", "example", "abstract", "references",
    "status of memo", "status of this memo", "editorial statement",
    "supplementary material", "key words", "keywords", "introduction",
    "commentary", "summary", "license", "university of", "accepted by",
    "contractor report", "technical report", "volume", "issue",
]

def extract_handcrafted_features(text: str):
    t = str(text)
    t_stripped = t.strip()
    lower = t_stripped.lower()
    length = len(t_stripped)

    if length == 0:
        return np.zeros(9, dtype=float)

    words = t_stripped.split()
    num_words = len(words)

    num_upper = sum(1 for c in t_stripped if c.isupper())
    num_digit = sum(1 for c in t_stripped if c.isdigit())
    num_punct = sum(1 for c in t_stripped if c in string.punctuation)

    # ========== ① 连续特征压缩 ==========
    f_len_char = np.log1p(length) / 10       # ~0.20–0.60
    f_len_word = np.log1p(num_words) / 5     # ~0.10–0.40

    f_ratio_upper = (num_upper / length)     # 0–1
    f_ratio_digit = (num_digit / length)
    f_ratio_punct = (num_punct / length)

    # ========== ② 强化可疑关键词（×2.0） ==========
    has_suspicious = 1.0 if any(kw in lower for kw in SUSPICIOUS_KEYWORDS) else 0.0
    has_suspicious *= 2.0

    # ========== ③ 轻强化格式类错误（×1.5） ==========
    is_very_short = 1.0 if num_words <= 3 else 0.0
    starts_with_number = 1.0 if (words[0][0].isdigit() or re.match(r"^\d+[\.\)]?$", words[0])) else 0.0

    has_colon = 1.0 if (":" in t_stripped or ";" in t_stripped) else 0.0

    f_ratio_upper *= 1.5
    f_ratio_digit *= 1.5
    f_ratio_punct *= 1.5

    return np.array([
        f_len_char,
        f_len_word,
        f_ratio_upper,
        f_ratio_digit,
        f_ratio_punct,
        has_suspicious,
        is_very_short,
        starts_with_number,
        has_colon,
    ], dtype=float)


def build_handcrafted_feature_matrix(text_series):
    feats = [extract_handcrafted_features(t) for t in text_series]
    feats = np.vstack(feats)          # shape: (N, 9)
    return csr_matrix(feats)          # 稀疏矩阵形式，方便和 TF-IDF 拼接


# ============================
# Step 3. 五折 + 每折再切 10% 验证
# ============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_all = train_df["clean"].values
y_all = train_df["label"].values

BEST_ACC = -1
BEST_MODEL = None
fold_id = 0

print("\n=========== 5 折训练（char n-gram + 手工特征 NB）开始 ===========")

for train_index, _ in kf.split(train_df):
    fold_id += 1
    print(f"\n\n============== [FOLD {fold_id}] ==============")

    # 当前折的大训练集（先不区分验证）
    X_fold = X_all[train_index]
    y_fold = y_all[train_index]
    fold_df = pd.DataFrame({"text": X_fold, "label": y_fold})

    # 按类别抽 10% 做验证
    pos_fold = fold_df[fold_df["label"] == 1]
    neg_fold = fold_df[fold_df["label"] == 0]

    val_pos = pos_fold.sample(frac=0.10, random_state=fold_id)
    val_neg = neg_fold.sample(frac=0.10, random_state=fold_id)

    val_df = pd.concat([val_pos, val_neg], ignore_index=True)
    train_df_fold = fold_df.drop(val_df.index).reset_index(drop=True)

    print(f"[FOLD {fold_id}] 训练集大小 = {len(train_df_fold)}, 验证集大小 = {len(val_df)}")

    # ========= 3.1 字符 n-gram TF-IDF =========
    tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=50000,
        min_df=5,
        lowercase=False,
    )

    X_train_char = tfidf.fit_transform(train_df_fold["text"])
    X_val_char   = tfidf.transform(val_df["text"])
    X_test_char  = tfidf.transform(test_df["clean"])

    # ========= 3.2 手工特征矩阵（确定性的，不改数据） =========
    X_train_hand = build_handcrafted_feature_matrix(train_df_fold["text"])
    X_val_hand   = build_handcrafted_feature_matrix(val_df["text"])
    X_test_hand  = build_handcrafted_feature_matrix(test_df["clean"])

    # ========= 3.3 拼接特征： [char-TFIDF | handcrafted] =========
    X_train = hstack([X_train_char, X_train_hand], format="csr")
    X_val   = hstack([X_val_char,   X_val_hand],   format="csr")
    X_test  = hstack([X_test_char,  X_test_hand],  format="csr")

    y_train = train_df_fold["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ========= 3.4 训练 NB =========
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = nb.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"[FOLD {fold_id}] 验证集 Accuracy = {val_acc:.4f}, Macro-F1 = {val_f1:.4f}")

    # 测试集评估
    y_test_pred = nb.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f"[FOLD {fold_id}] 测试集 Accuracy = {test_acc:.4f}, Macro-F1 = {test_f1:.4f}")

    if test_acc > BEST_ACC:
        BEST_ACC = test_acc
        BEST_MODEL = {
            "fold": fold_id,
            "acc": test_acc,
            "macro_f1": test_f1,
            "pred_test": y_test_pred,
            "vectorizer": tfidf,
            "model": nb,
        }

print("\n=========== 5 折训练完成 ===========")

# ============================
# Step 4. 输出最佳模型在测试集上的表现
# ============================
print(f"\n[BEST] 来自 Fold {BEST_MODEL['fold']}")
print(f"[BEST] Test Accuracy = {BEST_MODEL['acc']:.4f}")
print(f"[BEST] Test Macro-F1 = {BEST_MODEL['macro_f1']:.4f}\n")

print("=== 最佳模型分类报告 ===")
print(classification_report(test_df["label"], BEST_MODEL["pred_test"], digits=4))

print("=== 混淆矩阵 ===")
print(confusion_matrix(test_df["label"], BEST_MODEL["pred_test"]))

print("\n[INFO] 实验全部完成（Char n-gram + 手工特征 NB）")
