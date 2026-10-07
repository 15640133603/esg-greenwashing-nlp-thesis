# -*- coding: utf-8 -*-
"""
make_figures.py —— 全部论文插图（真实数据绘制）

图3-1  NLP 测度构建五步流程图（示意）
图3-2  研究设计框架图
图4-1  LDA 主题数 K 连贯性曲线
图4-2  话语强度与漂绿测度的分布
图4-3  GW 五分组条件均值：处罚率与ESG评级
图4-4  高低漂绿组月度超额回报对比
图4-5  聚合效度相关系数条形图
图4-6  词嵌入语义簇 t-SNE 投影
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATH_FIGURES, PATH_PROCESSED, PATH_PROCESSED as PP, PATH_RAW, PATH_RESULTS, PALETTE, set_seed

set_seed(20260101)


def save(fig, name):
    fig.savefig(os.path.join(PATH_FIGURES, name), dpi=200)
    plt.close(fig)
    print(f"  ✓ {name}")


# ---------------------------------------------------------------------------
def fig_3_1_pipeline():
    """五步流程图"""
    steps = [
        ("① 语料获取", "ESG报告 / 年报MD&A\n巨潮资讯网 · 468家 × 5年", "#2E5A88"),
        ("② 文本预处理", "清洗 → jieba分词\n停用词 → 归一化", "#4F81BD"),
        ("③ 特征提取", "词典法 / TF-IDF\nWord2Vec / LDA", "#3A7CA5"),
        ("④ 指标构建", "Talk / Promo / Spec\nHedge / GW(PCA)", "#C0504D"),
        ("⑤ 信效度检验", "内容·聚合·区分·预测\n不通过→回到③", "#7F9C4A"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axis("off")
    w, h, gap = 2.0, 1.7, 0.30
    for i, (t, d, c) in enumerate(steps):
        x = i * (w + gap)
        box = FancyBboxPatch((x, 0.25), w, h, boxstyle="round,pad=0.08",
                             fc=c, ec="none", alpha=0.92)
        ax.add_patch(box)
        ax.text(x + w / 2, 0.25 + h * 0.72, t, ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold")
        ax.text(x + w / 2, 0.25 + h * 0.30, d, ha="center", va="center",
                fontsize=8.6, color="white", linespacing=1.5)
        if i < 4:
            ar = FancyArrowPatch((x + w + 0.02, 0.25 + h / 2),
                                 (x + w + gap - 0.02, 0.25 + h / 2),
                                 arrowstyle="-|>", mutation_scale=22,
                                 color="#4a4a4a", lw=1.8)
            ax.add_patch(ar)
    ax.set_xlim(-0.15, 5 * (w + gap))
    ax.set_ylim(0, 2.2)
    save(fig, "fig3_1_pipeline.png")


# ---------------------------------------------------------------------------
def fig_3_2_framework():
    """研究设计框架图"""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=10, tc="white", bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           fc=fc, ec="none", alpha=0.93)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.6,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color="#4a4a4a", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=16, color=color, lw=1.6))

    # 左列：文本语料
    box(0.2, 3.9, 2.6, 1.0, "ESG报告 / 社会责任报告\n（自愿披露）", "#2E5A88", 9.5)
    box(0.2, 2.6, 2.6, 1.0, "年报 MD&A\n（强制披露）", "#2E5A88", 9.5)
    # 中列：五步流水线
    box(3.5, 3.3, 2.5, 1.7, "NLP 五步流水线\n预处理→特征→指标", "#4F81BD", 10, bold=True)
    # 右列：测度
    box(6.7, 4.1, 2.4, 0.85, "话语强度 Talk", "#3A7CA5", 10)
    box(6.7, 3.1, 2.4, 0.85, "宣传漂移 Promo", "#3A7CA5", 10)
    box(6.7, 2.1, 2.4, 0.85, "复合漂绿 GW (PCA)", "#C0504D", 10, bold=True)
    # 底行：效度
    box(3.5, 0.6, 5.6, 0.95, "信效度检验：聚合(评级/词嵌入/TF-IDF/LDA) · 区分 · 预测(处罚/研发) · 方差分解",
        "#7F9C4A", 9.5)
    # 顶行：经济后果
    box(3.5, 4.9, 5.6, 0.72, "市场反应检验：高漂绿组合超额回报", "#8064A2", 9.5)

    arrow(2.8, 4.4, 3.45, 4.15)
    arrow(2.8, 3.1, 3.45, 3.55)
    arrow(6.05, 4.15, 6.65, 4.5)
    arrow(6.05, 4.0, 6.65, 3.5)
    arrow(6.05, 3.7, 6.65, 2.5)
    arrow(7.9, 2.05, 7.9, 1.6)   # 测度 → 效度
    arrow(6.3, 1.1, 6.3, 1.55, color="#8064A2")  # 效度 → 市场反应(反向箭头示意)

    ax.set_xlim(0, 9.6)
    ax.set_ylim(0.2, 5.9)
    save(fig, "fig3_2_framework.png")


# ---------------------------------------------------------------------------
def fig_4_1_lda_k():
    c = pd.read_csv(os.path.join(PP, "lda_coherence.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(c["k"], c["coherence_c_v"], "o-", color=PALETTE[0], lw=2, ms=7)
    best = c.loc[c["coherence_c_v"].idxmax()]
    ax.scatter([best["k"]], [best["coherence_c_v"]], s=180, facecolor="none",
               edgecolor=PALETTE[1], lw=2.2, zorder=5)
    ax.annotate(f"最优 K={int(best['k'])}\nC_v={best['coherence_c_v']:.3f}",
                (best["k"], best["coherence_c_v"]),
                textcoords="offset points", xytext=(18, -6), fontsize=10,
                color=PALETTE[1])
    ax.set_xlabel("主题数 K")
    ax.set_ylabel("主题连贯性 $C_v$")
    ax.set_xticks(c["k"])
    ax.grid(alpha=0.25)
    save(fig, "fig4_1_lda_k.png")


# ---------------------------------------------------------------------------
def fig_4_2_distribution():
    m = pd.read_csv(os.path.join(PP, "measures_panel.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(m["Talk"], bins=36, color=PALETTE[0], alpha=0.85, edgecolor="white")
    axes[0].set_title("(a) ESG话语强度 Talk 分布")
    axes[0].set_xlabel("Talk（千词密度 %）")
    axes[0].set_ylabel("企业-年度观测数")
    axes[1].hist(m["GW"], bins=36, color=PALETTE[1], alpha=0.85, edgecolor="white")
    axes[1].set_title("(b) 复合漂绿测度 GW 分布")
    axes[1].set_xlabel("GW（标准化 z 值）")
    axes[1].set_ylabel("企业-年度观测数")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig4_2_distribution.png")


# ---------------------------------------------------------------------------
def fig_4_3_conditional():
    cm = pd.read_csv(os.path.join(PATH_RESULTS, "tab4_7_conditional_means.csv"))
    fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
    x = cm["gw_q5"]
    ax1.bar(x, cm["pun_rate"] * 100, color=[PALETTE[2]] * 4 + [PALETTE[1]],
            alpha=0.88, width=0.62)
    ax1.set_xlabel("漂绿测度 GW 五分组（1=最低，5=最高）")
    ax1.set_ylabel("下一年环境处罚发生率（%）", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax1.twinx()
    ax2.plot(x, cm["mean_esg"], "o--", color=PALETTE[1], lw=1.8, ms=7)
    ax2.set_ylabel("华证ESG评级均值", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax1.grid(axis="y", alpha=0.25)
    save(fig, "fig5_2_conditional.png")


# ---------------------------------------------------------------------------
def fig_4_4_market():
    t = pd.read_csv(os.path.join(PATH_RESULTS, "tab4_8_market_reaction.csv"),
                    index_col=0)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    years = t.columns.astype(int)
    ax.bar(years - 0.17, t.loc[0] * 100, width=0.32, label="低漂绿组",
           color=PALETTE[0], alpha=0.9)
    ax.bar(years + 0.17, t.loc[1] * 100, width=0.32, label="高漂绿组",
           color=PALETTE[1], alpha=0.9)
    ax.axhline(0, color="#4a4a4a", lw=0.8)
    ax.set_xlabel("年份")
    ax.set_ylabel("月度平均超额回报（%）")
    ax.legend(frameon=False)
    ax.set_xticks(years)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig5_3_market.png")


# ---------------------------------------------------------------------------
def fig_4_5_convergent():
    t = pd.read_csv(os.path.join(PATH_RESULTS, "tab4_2_convergent.csv"))
    t = t[t["变量对"].str.contains("↔")]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    colors = [PALETTE[0] if "话语强度" in v else PALETTE[1] for v in t["变量对"]]
    y = np.arange(len(t))
    ax.barh(y, t["r"], color=colors, alpha=0.9, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([v.replace(" ↔ ", "↔\n") for v in t["变量对"]], fontsize=8.8)
    ax.invert_yaxis()
    ax.axvline(0, color="#4a4a4a", lw=0.8)
    ax.set_xlabel("Pearson 相关系数 r")
    ax.set_xlim(-0.5, 1.0)
    for i, (r, pv) in enumerate(zip(t["r"], t["p"])):
        ax.text(r + (0.02 if r >= 0 else -0.02), i,
                f"{r:+.2f}", va="center",
                ha="left" if r >= 0 else "right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig5_1_convergent.png")


# ---------------------------------------------------------------------------
def fig_4_6_tsne():
    from sklearn.manifold import TSNE
    from gensim.models import Word2Vec
    from esg_lexicon import PROMOTIONAL, SUBSTANTIVE

    w2v = Word2Vec.load(os.path.join(PATH_PROCESSED, "w2v_model", "word2vec.model"))
    words_sub = [w for w in SUBSTANTIVE if w in w2v.wv][:28]
    words_pro = [w for w in PROMOTIONAL if w in w2v.wv][:28]
    words = words_sub + words_pro
    X = np.vstack([w2v.wv[w] for w in words])
    ts = TSNE(n_components=2, perplexity=12, random_state=20260101,
              init="pca", learning_rate="auto")
    P = ts.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    n = len(words_sub)
    ax.scatter(P[:n, 0], P[:n, 1], c=PALETTE[0], s=46, alpha=0.85,
               label="实质行动语义簇", edgecolor="white")
    ax.scatter(P[n:, 0], P[n:, 1], c=PALETTE[1], s=46, alpha=0.85,
               label="宣传夸饰语义簇", edgecolor="white")
    for i, w in enumerate(words):
        ax.annotate(w, (P[i, 0], P[i, 1]), fontsize=7.2,
                    xytext=(3, 3), textcoords="offset points", alpha=0.85)
    ax.legend(frameon=False, loc="best")
    ax.set_title("ESG 话语词嵌入 t-SNE 投影：两类语义簇边界清晰")
    ax.set_xticks([]), ax.set_yticks([])
    save(fig, "fig5_4_tsne.png")


# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("绘制全部论文插图")
    print("=" * 62)
    fig_3_1_pipeline()
    fig_3_2_framework()
    fig_4_1_lda_k()
    fig_4_2_distribution()
    fig_4_3_conditional()
    fig_4_4_market()
    fig_4_5_convergent()
    fig_4_6_tsne()
    print(f"\n输出目录: {PATH_FIGURES}")


if __name__ == "__main__":
    main()
