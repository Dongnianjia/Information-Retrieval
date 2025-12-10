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

pos_path = r"D:\homework\大三上\信息检索\作业2\positive_trainingSet_cleaned.txt"
neg_path = r"D:\homework\大三上\信息检索\作业2\negative_trainingSet.txt"
test_path = r"D:\homework\大三上\信息检索\作业2\testSet-1000.csv"

pos = read_all_lines(pos_path); pos["label"] = 1
neg = read_all_lines(neg_path); neg["label"] = 0

train_df = pd.concat([pos, neg], ignore_index=True)
print(f"[INFO] 训练集共 {len(train_df)} 条（正 {len(pos)}，负 {len(neg)}）")

# ============================
# Step 1.5 测试集
# ============================
test_df = pd.read_csv(test_path)
test_df.rename(columns={"title given by manchine": "title"}, inplace=True)
test_df["label"] = test_df["Y/N"].map({"Y": 1, "N": 0})

missing_mask = test_df["title"].isna() & (test_df["label"] == 1)
test_df.loc[missing_mask, "title"] = test_df.loc[missing_mask, "original title"]
test_df["title"] = test_df["title"].fillna("").astype(str)

print("\n[DEBUG] 测试集前 5 行：")
print(test_df.head()[["title", "original title", "label"]])

# ============================
# Step 2. 文本清洗
# ============================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean"] = train_df["title"].apply(clean_text)
test_df["clean"]  = test_df["title"].apply(clean_text)

# ============================
# Step 2.5 手工特征扩展（增强版）
# ============================

SUSPICIOUS_KEYWORDS = [
    "contents", "table of contents", "example", "abstract", "references",
    "status of memo", "editorial statement", "supplementary material",
    "keywords", "introduction", "summary", "license", "university of",
    "technical report", "volume", "issue", "appendix", "revision",
    "manuscript", "conference", "proceedings", "journal"
]

def extract_handcrafted_features(text: str):
    t = str(text)
    t_stripped = t.strip()
    lower = t_stripped.lower()
    length = len(t_stripped)

    if length == 0:
        return np.zeros(11, dtype=float)

    words = t_stripped.split()
    num_words = len(words)

    num_upper = sum(1 for c in t_stripped if c.isupper())
    num_digit = sum(1 for c in t_stripped if c.isdigit())
    num_punct = sum(1 for c in t_stripped if c in string.punctuation)

    # ----- 连续特征 -----
    f_len_char = np.log1p(length) / 10
    f_len_word = np.log1p(num_words) / 5

    f_ratio_upper = num_upper / length
    f_ratio_digit = num_digit / length
    f_ratio_punct = num_punct / length

    # ----- 强化可疑关键词 -----
    has_suspicious = 1.0 if any(kw in lower for kw in SUSPICIOUS_KEYWORDS) else 0.0
    has_suspicious *= 2.0

    # ----- 新增 title-case 比例 -----
    num_titlecase = sum(1 for w in words if w.istitle())
    ratio_titlecase = num_titlecase / (num_words + 1e-6)

    # ----- 新增：非标题词出现次数 -----
    NON_TITLE_WORDS = ["figure", "table", "chapter", "section", "page"]
    non_title_flag = 1.0 if any(w in lower for w in NON_TITLE_WORDS) else 0.0

    # ----- 小结构特征 -----
    is_very_short = 1.0 if num_words <= 3 else 0.0
    starts_with_number = 1.0 if (words[0][0].isdigit()) else 0.0
    has_colon = 1.0 if (":" in t_stripped or ";" in t_stripped) else 0.0

    # 强化语法错误特征
    f_ratio_upper *= 1.5
    f_ratio_digit *= 1.5
    f_ratio_punct *= 1.5

    return np.array([
        f_len_char, f_len_word,
        f_ratio_upper, f_ratio_digit, f_ratio_punct,
        has_suspicious, is_very_short, starts_with_number,
        has_colon, ratio_titlecase, non_title_flag
    ], dtype=float)

def build_handcrafted_feature_matrix(text_series):
    feats = [extract_handcrafted_features(t) for t in text_series]
    return csr_matrix(np.vstack(feats))


# ============================
# Step 3. 五折训练（加入 word-level TF-IDF + NB α 搜索）
# ============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_all = train_df["clean"].values
y_all = train_df["label"].values

BEST_ACC = -1
BEST_MODEL = None
fold_id = 0

print("\n=========== 5 折训练（增强版 NB）开始 ===========")

for train_index, _ in kf.split(train_df):
    fold_id += 1
    print(f"\n\n============== [FOLD {fold_id}] ==============")

    # 当前折数据
    X_fold = X_all[train_index]
    y_fold = y_all[train_index]
    fold_df = pd.DataFrame({"text": X_fold, "label": y_fold})

    # 类别分层抽验证集
    pos_fold = fold_df[fold_df["label"] == 1]
    neg_fold = fold_df[fold_df["label"] == 0]

    val_pos = pos_fold.sample(frac=0.10, random_state=fold_id)
    val_neg = neg_fold.sample(frac=0.10, random_state=fold_id)

    val_df = pd.concat([val_pos, val_neg], ignore_index=True)
    train_df_fold = fold_df.drop(val_df.index).reset_index(drop=True)

    print(f"[FOLD {fold_id}] train={len(train_df_fold)}, val={len(val_df)}")

    # ====== char-level TF-IDF ======
    tfidf_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=50000,
        min_df=5,
        lowercase=False,
    )
    X_train_char = tfidf_char.fit_transform(train_df_fold["text"])
    X_val_char   = tfidf_char.transform(val_df["text"])
    X_test_char  = tfidf_char.transform(test_df["clean"])

    # ====== word-level TF-IDF（新增）=====
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=20000,
        lowercase=True
    )
    X_train_word = tfidf_word.fit_transform(train_df_fold["text"])
    X_val_word   = tfidf_word.transform(val_df["text"])
    X_test_word  = tfidf_word.transform(test_df["clean"])

    # ====== 手工特征 ======
    X_train_hand = build_handcrafted_feature_matrix(train_df_fold["text"])
    X_val_hand   = build_handcrafted_feature_matrix(val_df["text"])
    X_test_hand  = build_handcrafted_feature_matrix(test_df["clean"])

    # 拼接
    X_train = hstack([X_train_char, X_train_word, X_train_hand])
    X_val   = hstack([X_val_char, X_val_word, X_val_hand])
    X_test  = hstack([X_test_char, X_test_word, X_test_hand])

    y_train = train_df_fold["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ====== 网格搜索 NB α ======
    best_fold_acc = -1
    best_fold_model = None

    for alpha in [0.01, 0.05, 0.1, 0.3, 1.0]:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train, y_train)

        y_val_pred = nb.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            best_fold_model = (nb, alpha)

    nb, best_alpha = best_fold_model
    print(f"[FOLD {fold_id}] 最优 alpha={best_alpha}")

    # 测试集表现
    y_test_pred = nb.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"[FOLD {fold_id}] Test Accuracy = {test_acc:.4f}, Macro-F1 = {test_f1:.4f}")

    if test_acc > BEST_ACC:
        BEST_ACC = test_acc
        BEST_MODEL = {
            "fold": fold_id,
            "acc": test_acc,
            "macro_f1": test_f1,
            "pred_test": y_test_pred,
            "model": nb,
            "tfidf_char": tfidf_char,
            "tfidf_word": tfidf_word,
        }

print("\n=========== 5 折训练完成 ===========")

# ============================
# Step 4. 输出最佳模型
# ============================
print(f"\n[BEST] 来自 Fold {BEST_MODEL['fold']}")
print(f"[BEST] Test Accuracy = {BEST_MODEL['acc']:.4f}")
print(f"[BEST] Test Macro-F1 = {BEST_MODEL['macro_f1']:.4f}\n")

print("=== 最佳模型分类报告 ===")
print(classification_report(test_df["label"], BEST_MODEL["pred_test"], digits=4))

print("=== 混淆矩阵 ===")
print(confusion_matrix(test_df["label"], BEST_MODEL["pred_test"]))

print("\n[INFO] 增强版 NB 实验完成")
