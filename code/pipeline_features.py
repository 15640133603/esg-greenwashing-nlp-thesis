# -*- coding: utf-8 -*-
"""
pipeline_features.py —— 第二步+第三步：文本预处理与特征提取

实现课件"五步法"中的：
  ② 文本预处理：清洗 → jieba 精确分词 → 三维度停用词过滤 → 归一化
  ③ 特征提取：  词典法 / TF-IDF / Word2Vec 词嵌入 / LDA 主题模型

输出（data/processed/）：
  sentences_tokenized.csv   分词结果（含词数，供篇幅归一）
  doc_level_features.csv    文档级特征表（企业-年度 × 各特征）
  w2v_model/                Word2Vec 模型
  lda_model/                LDA 模型与主题词表
"""

import os
import sys
from collections import Counter

import gensim
import jieba
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from gensim.models.coherencemodel import CoherenceModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATH_PROCESSED, PATH_RESULTS, PATH_RAW, RANDOM_SEED, set_seed
from esg_lexicon import (HEDGE, LEXICON_META, NEGATION, PROMOTIONAL, RISK_PHRASE,
                         STOPWORDS_FIN, SUBSTANTIVE, TIME_LONG, TIME_SHORT)

set_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# ②-1 文本清洗（对应课件"去页眉页脚、去乱码与重复空行"）
# ---------------------------------------------------------------------------
import re

NOISE_PATTERNS = [
    re.compile(r"\(第\s*\d+\s*页\)"), re.compile(r"第\s*\d+\s*页"),
    re.compile(r"证券代码[:：]\s*\d{6}"), re.compile(r"证券简称[:：]\s*[\u4e00-\u9fa5A-Za-z]+"),
    re.compile(r"\d{4}-\d{2}-\d{2}"), re.compile(r"[\u3000\t\r]"),
    re.compile(r"-même|\ufffd"),          # 乱码字符
    re.compile(r"\s{2,}"),
]


def clean_text(s: str) -> str:
    for p in NOISE_PATTERNS:
        s = p.sub(" ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# ②-2 分词与停用词过滤
# ---------------------------------------------------------------------------
# 向 jieba 词典中添加领域专有词，避免切碎（如"碳排放强度"）
DOMAIN_TERMS = (SUBSTANTIVE + PROMOTIONAL + RISK_PHRASE + TIME_SHORT + TIME_LONG
                + ["双碳", "碳达峰", "碳中和", "ESG", "漂绿", "绿色治理",
                   "环境信息披露", "ESG报告", "社会责任报告"])
for w in DOMAIN_TERMS:
    jieba.add_word(w, freq=100000)

GENERAL_STOPWORDS = set([
    "，", "。", "、", "；", "：", "（", "）", "(", ")", "%", ".", "的", "了",
    "和", "是", "在", "与", "及", "对", "为", "以", "并", "有", "个", "等",
    "中", "上", "下", "其", "或", "从", "由", "将", "已", "均", "都", "不",
    "要", "会", "可", "能", "地", "得", "着", "过", "也", "又", "很", "较",
    "万", "亿", "约", "含", "至", "到", "该", "此", "这些", "那些", "我们",
])
STOPWORDS = GENERAL_STOPWORDS | set(STOPWORDS_FIN)


def tokenize(text: str) -> list:
    """jieba 精确模式分词 + 停用词过滤 + 单字符过滤。"""
    words = jieba.lcut(clean_text(text))  # 默认即精确模式
    return [w for w in words
            if w not in STOPWORDS and len(w) > 1 and not w.isdigit()]


# ---------------------------------------------------------------------------
# ③-1 词典法特征（核心路线，对应课件 3.3 路线一）
# ---------------------------------------------------------------------------
def dict_hits(tokens: list, lexicon: list) -> int:
    """按词表精确匹配计数（整词命中）。"""
    tset = Counter(tokens)
    return sum(tset.get(w, 0) for w in lexicon)


def build_doc_features(txt: pd.DataFrame) -> pd.DataFrame:
    """句子级语料 → 企业-年度文档特征。"""
    grp = txt.groupby(["stkcd", "year"], sort=True)

    rows = []
    for (code, year), g in grp:
        full_text = "。".join(g["text"].tolist())
        all_tokens = []
        sec_tokens = {"ESG报告": [], "年报MD&A": []}
        for _, r in g.iterrows():
            toks = tokenize(r["text"])
            all_tokens.extend(toks)
            sec_tokens[r["source"]].extend(toks)

        n_total = len(all_tokens)
        if n_total < 10:
            continue

        # --- 各词典命中数（分子） ---
        hits_sub = dict_hits(all_tokens, SUBSTANTIVE)
        hits_pro = dict_hits(all_tokens, PROMOTIONAL)
        hits_risk = dict_hits(all_tokens, RISK_PHRASE)
        hits_hedge = dict_hits(all_tokens, HEDGE)
        hits_short = dict_hits(all_tokens, TIME_SHORT)
        hits_long = dict_hits(all_tokens, TIME_LONG)

        # ESG 报告部分词数（分母口径一：仅 ESG 报告）
        n_esg = len(sec_tokens["ESG报告"])

        # 否定修正：否定词出现时，邻近正向词的权重打折（简化处理：按比例）
        n_neg = dict_hits(all_tokens, NEGATION)
        neg_ratio = min(n_neg / max(n_total, 1) * 10, 0.5)   # 上限 0.5

        rows.append({
            "stkcd": code, "year": int(year),
            "n_sents": len(g),
            "n_words": n_total,
            "n_words_esg": n_esg,
            "neg_ratio": neg_ratio,
            # L1 原始计数
            "cnt_substantive": hits_sub,
            "cnt_promotional": hits_pro,
            "cnt_risk": hits_risk,
            "cnt_hedge": hits_hedge,
            "cnt_time_short": hits_short,
            "cnt_time_long": hits_long,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ③-2 TF-IDF 特征（区分度加权，为聚合效度提供"绿色语义得分"）
# ---------------------------------------------------------------------------
def tfidf_green_score(txt: pd.DataFrame) -> pd.DataFrame:
    """以实质性绿色词与宣传性词在 TF-IDF 空间中的投影作为对照测度。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    doc = (txt.groupby(["stkcd", "year"])["text"]
           .apply(lambda s: "。".join(s.tolist()))
           .reset_index())
    doc["text_clean"] = doc["text"].apply(clean_text)

    vocab = list(set(SUBSTANTIVE + PROMOTIONAL))
    # 直接以词典词为词表（分词后精确匹配），min_df 过滤噪声
    vec = TfidfVectorizer(vocabulary=vocab, tokenizer=tokenize,
                          lowercase=False, token_pattern=None)
    X = vec.fit_transform(doc["text_clean"].tolist())
    names = vec.get_feature_names_out()

    sub_idx = [i for i, w in enumerate(names) if w in SUBSTANTIVE]
    pro_idx = [i for i, w in enumerate(names) if w in PROMOTIONAL]

    out = pd.DataFrame({
        "stkcd": doc["stkcd"], "year": doc["year"],
        "tfidf_substantive": X[:, sub_idx].sum(axis=1).A1,
        "tfidf_promotional": X[:, pro_idx].sum(axis=1).A1,
    })
    return out


# ---------------------------------------------------------------------------
# ③-3 Word2Vec 词嵌入（对应课件 3.3 路线二）
# ---------------------------------------------------------------------------
def word2vec_features(txt: pd.DataFrame) -> pd.DataFrame:
    set_seed(RANDOM_SEED)
    sents_tok = [tokenize(t) for t in txt["text"].tolist()]
    sents_tok = [s for s in sents_tok if len(s) >= 3]

    w2v = Word2Vec(sents_tok, vector_size=64, window=5, min_count=3,
                   workers=4, sg=1, seed=RANDOM_SEED, epochs=8)

    model_dir = os.path.join(PATH_PROCESSED, "w2v_model")
    os.makedirs(model_dir, exist_ok=True)
    w2v.save(os.path.join(model_dir, "word2vec.model"))

    # 文档向量 = 词向量均值（课件建议做法）
    def doc_vec(tokens):
        vs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        return np.mean(vs, axis=0) if vs else np.zeros(w2v.vector_size)

    # 语义锚点：实质行动语义簇 vs 宣话语义簇 的质心
    anchor_sub = np.mean([w2v.wv[w] for w in SUBSTANTIVE if w in w2v.wv], axis=0)
    anchor_pro = np.mean([w2v.wv[w] for w in PROMOTIONAL if w in w2v.wv], axis=0)

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    doc = (txt.groupby(["stkcd", "year"])["text"]
           .apply(lambda s: [t for sent in s for t in tokenize(sent)])
           .reset_index())
    res = []
    for _, r in doc.iterrows():
        v = doc_vec(r["text"])
        res.append({"stkcd": r["stkcd"], "year": r["year"],
                    "w2v_cos_sub": cos(v, anchor_sub),
                    "w2v_cos_pro": cos(v, anchor_pro)})
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# ③-4 LDA 主题模型（对应课件 3.3 路线三，路径 B：主题定词）
# ---------------------------------------------------------------------------
def lda_features(txt: pd.DataFrame, k_list=(4, 6, 8, 10)) -> pd.DataFrame:
    set_seed(RANDOM_SEED)
    doc = (txt.groupby(["stkcd", "year"])["text"]
           .apply(lambda s: [t for sent in s for t in tokenize(sent)])
           .reset_index())
    docs = [d for d in doc["text"].tolist() if len(d) >= 10]

    dict_corpus = corpora.Dictionary(docs)
    dict_corpus.filter_extremes(no_below=15, no_above=0.5)
    bow = [dict_corpus.doc2bow(d) for d in docs]

    # --- 主题数 K 选择：连贯性 C_v 曲线（课件要求必须报告） ---
    coherence_scores = {}
    best = None
    for k in k_list:
        lda = LdaModel(bow, num_topics=k, id2word=dict_corpus,
                       passes=8, iterations=200, random_state=RANDOM_SEED,
                       alpha="auto", eta="auto")
        cm = CoherenceModel(model=lda, texts=docs, dictionary=dict_corpus,
                            coherence="c_v", processes=2)
        c = float(cm.get_coherence())
        coherence_scores[k] = c
        if best is None or c > best[1]:
            best = (k, c, lda)
        print(f"  LDA K={k:>2}  连贯性 C_v = {c:.4f}")

    k_best, c_best, lda_best = best
    print(f"  → 选定 K = {k_best}（连贯性最高 {c_best:.4f}）")

    # 保存主题词表
    topic_words = []
    for tid in range(k_best):
        words = [w for w, _ in lda_best.show_topic(tid, topn=15)]
        topic_words.append({"topic_id": tid, "top_words": " ".join(words)})
    pd.DataFrame(topic_words).to_csv(
        os.path.join(PATH_PROCESSED, "lda_topics.csv"),
        index=False, encoding="utf-8-sig")

    model_dir = os.path.join(PATH_PROCESSED, "lda_model")
    os.makedirs(model_dir, exist_ok=True)
    lda_best.save(os.path.join(model_dir, "lda.model"))
    dict_corpus.save(os.path.join(model_dir, "dictionary.dict"))

    pd.DataFrame({"k": list(coherence_scores.keys()),
                  "coherence_c_v": list(coherence_scores.values())}).to_csv(
        os.path.join(PATH_PROCESSED, "lda_coherence.csv"),
        index=False, encoding="utf-8-sig")

    # --- 主题词频求和（路径 B：用主题词做词典计数） ---
    # 识别"绿色行动主题"：主题词与实质词表重合度最高者
    sub_set = set(SUBSTANTIVE)
    overlap = []
    for tid in range(k_best):
        words = [w for w, _ in lda_best.show_topic(tid, topn=30)]
        overlap.append(sum(1 for w in words if w in sub_set) / len(words))
    green_topic = int(np.argmax(overlap))
    green_words = [w for w, _ in lda_best.show_topic(green_topic, topn=25)
                   if len(w) > 1]
    print(f"  绿色行动主题 = Topic {green_topic}（与实质词表重合 {max(overlap):.1%}）")
    print(f"  主题词: {' '.join(green_words[:12])}")
    pd.DataFrame({"green_words": green_words}).to_csv(
        os.path.join(PATH_PROCESSED, "lda_green_words.csv"),
        index=False, encoding="utf-8-sig")

    # 文档归属
    doc_topics = []
    idx_map = {i: (r["stkcd"], r["year"]) for i, r in doc.iterrows()
               if len(r["text"]) >= 10}
    for i, d in enumerate(bow):
        tp = lda_best.get_document_topics(d, minimum_probability=0)
        code, yr = idx_map[i]
        row = {"stkcd": code, "year": yr}
        for tid, p in tp[:k_best]:
            row[f"lda_topic{tid}"] = p
        doc_topics.append(row)
    res = pd.DataFrame(doc_topics)
    res["lda_green_topic_share"] = res.get(f"lda_topic{green_topic}", 0.0)
    return res[["stkcd", "year", "lda_green_topic_share"]]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("步骤② 文本预处理 + 步骤③ 特征提取")
    print("=" * 62)
    txt = pd.read_csv(os.path.join(PATH_RAW, "esg_reports.csv"))
    print(f"读入句子级语料 {len(txt):,} 条")

    # ② 预处理 + 文档级词频特征
    print("\n[2/4] 分词与词典法特征...")
    feats = build_doc_features(txt)
    print(f"  文档数 {len(feats)}，平均词数 {feats['n_words'].mean():.0f}")

    # ③-2 TF-IDF
    print("[3/4] TF-IDF 特征...")
    tfidf = tfidf_green_score(txt)

    # ③-3 Word2Vec
    print("[3.5/4] Word2Vec 词嵌入...")
    w2v = word2vec_features(txt)

    # ③-4 LDA
    print("[3.8/4] LDA 主题模型（K 选择中）...")
    lda = lda_features(txt)

    # 合并输出
    doc = feats.merge(tfidf, on=["stkcd", "year"], how="left")
    doc = doc.merge(w2v, on=["stkcd", "year"], how="left")
    doc = doc.merge(lda, on=["stkcd", "year"], how="left")
    doc.to_csv(os.path.join(PATH_PROCESSED, "doc_level_features.csv"),
               index=False, encoding="utf-8-sig")

    # 分词示例（论文附录用）
    sample = txt.sample(4, random_state=RANDOM_SEED).copy()
    sample["tokens"] = sample["text"].apply(lambda s: " / ".join(tokenize(s)))
    sample[["text", "tokens"]].to_csv(
        os.path.join(PATH_PROCESSED, "tokenize_demo.csv"),
        index=False, encoding="utf-8-sig")

    print(f"\n输出: {PATH_PROCESSED}")
    for fn in sorted(os.listdir(PATH_PROCESSED)):
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
