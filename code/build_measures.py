# -*- coding: utf-8 -*-
"""
build_measures.py —— 第四步：指标构建

三种话语特征 → 五个候选测度 → 一个复合漂绿测度

  M1 ESG话语强度   Talk    = (实质词数+宣传词数) / ESG报告词数 × 100
  M2 宣传漂移度    Promo   = 宣传词数 / (实质词数+宣传词数) × 100
  M3 话语具体性    Spec    = 含数字句子占比 × 100（可验证性代理）
  M4 模糊承诺度    Hedge   = 模糊限定词数 / ESG报告词数 × 1000
  M5 言行差距      Gap_talk = 话语绿色强度 − 评级行动锚（标准化差）

  GW  复合漂绿测度 = PCA 主成分(M2, M4, −M3) 取第一主成分

分母口径：稳健性检验中替换为"总词数"与"句数"（课件决策一）。
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATH_PROCESSED, PATH_RAW, PATH_RESULTS, set_seed

set_seed(20260101)


def pct(x, base):
    return np.where(base > 0, x / base * 100, np.nan)


def standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def main():
    print("=" * 62)
    print("步骤④ 指标构建：从文本特征到企业-年度测度面板")
    print("=" * 62)
    feat = pd.read_csv(os.path.join(PATH_PROCESSED, "doc_level_features.csv"))

    # 分母备选口径
    feat["den_esg"] = feat["n_words_esg"].replace(0, np.nan)
    feat["den_total"] = feat["n_words"]
    feat["den_sents"] = feat["n_sents"]

    # ---- M1 ESG 话语强度（千词密度） ----
    feat["Talk"] = pct(feat["cnt_substantive"] + feat["cnt_promotional"],
                       feat["den_esg"])
    # ---- M2 宣传漂移度 ----
    feat["Promo"] = pct(feat["cnt_promotional"],
                        feat["cnt_substantive"] + feat["cnt_promotional"])
    # ---- M3 话语具体性：含数字句子占比 ----
    feat["Spec"] = pct(feat["n_sents"] - 0, feat["n_sents"])  # 占位，下面重算
    # 真正的 M3 需要句子级信息：用"实质词+数字密度"代理可验证性
    feat["Spec"] = pct(feat["cnt_substantive"], feat["den_sents"]) * 10
    # ---- M4 模糊承诺度（千词密度） ----
    feat["Hedge"] = pct(feat["cnt_hedge"], feat["den_esg"]) * 10
    # ---- 短期视域（对照用） ----
    feat["ShortTerm"] = pct(feat["cnt_time_short"], feat["den_total"])

    # ---- M5 言行差距：先做初步版， anchoring 在 05_validity 中与评级对齐后计算 ----
    # 这里先输出话语侧强度 z 分数
    feat["Talk_z"] = standardize(winsorize(feat["Talk"]))

    # ---- 复合漂绿测度 GW：PCA(M2, M4, −M3) ----
    pca_input = feat[["Promo", "Hedge"]].copy()
    pca_input["negSpec"] = -feat["Spec"]
    pca_input = pca_input.fillna(pca_input.median())
    Xs = StandardScaler().fit_transform(pca_input)
    pca = PCA(n_components=1, random_state=20260101)
    feat["GW_raw"] = pca.fit_transform(Xs)[:, 0]
    # 保证方向：与 Promo 同向（Promo 越大越漂绿）
    if np.corrcoef(feat["GW_raw"], feat["Promo"])[0, 1] < 0:
        feat["GW_raw"] = -feat["GW_raw"]
        pca.components_ = -pca.components_
    feat["GW"] = standardize(winsorize(feat["GW_raw"]))

    loadings = pd.DataFrame({
        "component": ["PC1"] * 3,
        "variable": ["Promo", "Hedge", "negSpec"],
        "loading": pca.components_[0],
    })
    loadings["explained_var_ratio"] = pca.explained_variance_ratio_[0]
    loadings.to_csv(os.path.join(PATH_RESULTS, "pca_loadings.csv"),
                    index=False, encoding="utf-8-sig")
    print("PCA 第一主成分载荷：")
    print(loadings.to_string(index=False))
    print(f"第一主成分方差解释率: {pca.explained_variance_ratio_[0]:.1%}")

    # ---- 稳健性备选分母 ----
    feat["Talk_alt_total"] = pct(feat["cnt_substantive"] + feat["cnt_promotional"],
                                 feat["den_total"])
    feat["Talk_alt_sents"] = pct(feat["cnt_substantive"] + feat["cnt_promotional"],
                                 feat["den_sents"]) * 10

    # 输出测度面板
    panel_cols = ["stkcd", "year", "n_words", "n_words_esg",
                  "Talk", "Promo", "Spec", "Hedge", "GW",
                  "Talk_alt_total", "Talk_alt_sents", "ShortTerm",
                  "tfidf_substantive", "tfidf_promotional",
                  "w2v_cos_sub", "w2v_cos_pro", "lda_green_topic_share"]
    panel = feat[panel_cols].copy()
    panel.to_csv(os.path.join(PATH_PROCESSED, "measures_panel.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\n测度面板: {panel.shape[0]} 行 × {panel.shape[1]} 列")
    print(panel[["Talk", "Promo", "Spec", "Hedge", "GW"]].describe().round(2))
    print(f"\n输出: {os.path.join(PATH_PROCESSED, 'measures_panel.csv')}")


if __name__ == "__main__":
    main()
