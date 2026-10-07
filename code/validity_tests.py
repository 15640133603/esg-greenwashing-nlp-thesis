# -*- coding: utf-8 -*-
"""
validity_tests.py —— 第五步：信效度检验 + 描述性统计 + 回归分析

对应课件 3.5"四类必做检验"：
  ① 内容效度   —— 词表来源与理论依据（论文表格呈现，代码输出来源表）
  ② 聚合效度   —— 与华证 ESG 评级的相关；与 Word2Vec/TF-IDF 对照测度相关
  ③ 区分效度   —— 与公司规模、杠杆、行业虚拟变量低相关
  ④ 预测效度   —— GW/Talk 对 下一年环境处罚 (logit)、研发投入 (OLS) 的预测

附加（教材实战示范）：
  - 条件均值检验：按 GW 分组的处罚率单调性
  - 方差分解：测度方差中企业间占比（须显著大于组内噪声）
  - 市场反应：高漂绿组合处罚公告月的超额回报

输出：results/ 下全部表格（CSV + 可直接粘贴论文的三线表文本）
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATH_PROCESSED, PATH_RAW, PATH_RESULTS
from esg_lexicon import LEXICON, LEXICON_META


def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def load_panel() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(PATH_PROCESSED, "measures_panel.csv"))
    esg = pd.read_csv(os.path.join(PATH_RAW, "esg_ratings.csv"))
    fin = pd.read_csv(os.path.join(PATH_RAW, "financials.csv"))
    firms = pd.read_csv(os.path.join(PATH_RAW, "firms.csv"))
    pun = pd.read_csv(os.path.join(PATH_RAW, "punishments.csv"))
    ret = pd.read_csv(os.path.join(PATH_RAW, "stock_returns.csv"))

    df = (m.merge(esg, on=["stkcd", "year"])
           .merge(fin, on=["stkcd", "year"])
           .merge(firms[["stkcd", "industry", "heavy_pollution", "soe"]],
                  on="stkcd"))

    # 下一年处罚（t+1 期处罚事件在 punishments 中记为 year+1）
    pun_next = (pun.groupby(["stkcd", "year"])["fine_wan"]
                .agg(["count", "sum"]).reset_index()
                .rename(columns={"count": "n_pun", "sum": "fine_sum"}))
    pun_next["year"] = pun_next["year"] - 1        # 对齐到 t 期（用 t 期测度预测 t+1 处罚）
    df = df.merge(pun_next, on=["stkcd", "year"], how="left")
    df[["n_pun", "fine_sum"]] = df[["n_pun", "fine_sum"]].fillna(0)
    df["PunNext"] = (df["n_pun"] > 0).astype(int)
    df["ln_fine_next"] = np.log1p(df["fine_sum"])

    # ESG 评级 z 分数（行动锚）
    df["ESG_z"] = (df["esg_score"] - df["esg_score"].mean()) / df["esg_score"].std()
    df["RD"] = df["rd_intensity"] * 100            # 研发强度 %
    df["Size"] = df["asset"]                       # ln 总资产
    return df


# ---------------------------------------------------------------------------
def content_validity(df_panel: pd.DataFrame):
    """① 内容效度：词表来源表（论文表 4-1）"""
    rows = []
    for dim, meta in LEXICON_META.items():
        rows.append({"维度": dim, "词数": len(LEXICON[dim]),
                     "理论依据": meta["理论依据"], "构建方式": meta["构建方式"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PATH_RESULTS, "tab4_1_lexicon_content.csv"),
               index=False, encoding="utf-8-sig")
    print("[① 内容效度] 词典来源表已输出（表4-1）")


def convergent_validity(df: pd.DataFrame):
    """② 聚合效度：话语测度 vs 评级 / 词嵌入 / TF-IDF / LDA"""
    rows = []
    pairs = [
        ("Talk", "esg_score", "话语强度 ↔ 华证ESG评级"),
        ("GW", "esg_score", "漂绿测度 ↔ 华证ESG评级(应低/负)"),
        ("Promo", "esg_score", "宣传漂移 ↔ 华证ESG评级"),
        ("Talk", "w2v_cos_sub", "话语强度 ↔ 词嵌入实质簇相似度"),
        ("Promo", "w2v_cos_pro", "宣传漂移 ↔ 词嵌入宣传簇相似度"),
        ("Talk", "tfidf_substantive", "话语强度 ↔ TF-IDF实质词得分"),
        ("Promo", "tfidf_promotional", "宣传漂移 ↔ TF-IDF宣传词得分"),
        ("Talk", "lda_green_topic_share", "话语强度 ↔ LDA绿色主题份额"),
    ]
    print("\n[② 聚合效度] Pearson 相关（表4-2）")
    for a, b, desc in pairs:
        sub = df[[a, b]].dropna()
        r, p = stats.pearsonr(sub[a], sub[b])
        rows.append({"变量对": desc, "r": round(r, 3),
                     "p": f"{p:.2g}", "n": len(sub)})
        print(f"  {desc:<38} r = {r:+.3f} (p = {p:.2g})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PATH_RESULTS, "tab4_2_convergent.csv"),
               index=False, encoding="utf-8-sig")


def discriminant_validity(df: pd.DataFrame):
    """③ 区分效度：与'不该相关'的变量相关性应低"""
    rows = []
    print("\n[③ 区分效度] （表4-3）")
    for v in ["Talk", "GW"]:
        for w in ["Size", "lev", "indep_ratio", "board_size"]:
            sub = df[[v, w]].dropna()
            r, p = stats.pearsonr(sub[v], sub[w])
            rows.append({"测度": v, "对照变量": w, "r": round(r, 3),
                         "p": f"{p:.2g}",
                         "判定": "低相关 ✓" if abs(r) < 0.3 else "需讨论"})
            print(f"  {v:<5} ↔ {w:<12} r = {r:+.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PATH_RESULTS, "tab4_3_discriminant.csv"),
               index=False, encoding="utf-8-sig")


def variance_decomposition(df: pd.DataFrame):
    """方差分解：企业间（组间）方差占比"""
    rows = []
    print("\n[方差分解] 组间方差占比（表4-4）")
    for v in ["Talk", "GW", "Promo", "Hedge"]:
        sub = df[["stkcd", v]].dropna()
        grand = sub[v].mean()
        gm = sub.groupby("stkcd")[v].mean()
        n_i = sub.groupby("stkcd")[v].size()
        ss_between = float((n_i * (gm - grand) ** 2).sum())
        ss_total = float(((sub[v] - grand) ** 2).sum())
        share = ss_between / ss_total
        rows.append({"测度": v, "企业间方差占比": round(share, 3),
                     "判定": "信号充足 ✓" if share > 0.4 else "噪声偏大"})
        print(f"  {v:<6} between-share = {share:.1%}")
    pd.DataFrame(rows).to_csv(
        os.path.join(PATH_RESULTS, "tab4_4_var_decomp.csv"),
        index=False, encoding="utf-8-sig")


def predictive_validity(df: pd.DataFrame):
    """④ 预测效度：GW → 下一年环境处罚（Logit）；Talk → 研发强度（OLS）"""
    print("\n[④ 预测效度] 回归检验（表4-5 / 表4-6）")

    d = df.copy()
    d["GW_w"] = winsorize(d["GW"])
    d["Talk_w"] = winsorize(d["Talk"])

    # 表4-5: Logit PunNext ~ GW + controls
    X_cols = ["GW_w", "Talk_w", "Size", "lev", "roa", "soe", "heavy_pollution"]
    dd = d[X_cols + ["PunNext", "stkcd", "year"]].dropna()
    logit = smf.logit("PunNext ~ GW_w + Talk_w + Size + lev + roa + soe "
                      "+ heavy_pollution + C(year)", data=dd).fit(disp=0)
    rows = []
    for name in logit.params.index:
        if name == "Intercept":
            continue
        rows.append({"变量": name, "系数": round(logit.params[name], 3),
                     "OR": round(np.exp(logit.params[name]), 3),
                     "z": round(logit.tvalues[name], 2),
                     "p": round(logit.pvalues[name], 4),
                     "显著性": star(logit.pvalues[name])})
    logit_df = pd.DataFrame(rows)
    logit_df.to_csv(os.path.join(PATH_RESULTS, "tab4_5_logit_punish.csv"),
                    index=False, encoding="utf-8-sig")
    print(logit_df.to_string(index=False))

    # 表4-6: OLS RD ~ Talk + controls（话语强度与创新投入）
    ols = smf.ols("RD ~ Talk_w + GW_w + Size + lev + roa + soe + C(year)",
                  data=d).fit(cov_type="cluster", cov_kwds={"groups": d["stkcd"]})
    rows = []
    for name in ols.params.index:
        if name == "Intercept":
            continue
        rows.append({"变量": name, "系数": round(ols.params[name], 3),
                     "t": round(ols.tvalues[name], 2),
                     "p": round(ols.pvalues[name], 4),
                     "显著性": star(ols.pvalues[name])})
    ols_df = pd.DataFrame(rows)
    ols_df.to_csv(os.path.join(PATH_RESULTS, "tab4_6_ols_rd.csv"),
                  index=False, encoding="utf-8-sig")
    print("\n" + ols_df.to_string(index=False))
    return logit, ols


def conditional_means(df: pd.DataFrame):
    """条件均值检验：按 GW 五分组的处罚率单调性"""
    d = df.copy()
    d["gw_q5"] = pd.qcut(d["GW"], 5, labels=False, duplicates="drop") + 1
    cm = d.groupby("gw_q5").agg(
        pun_rate=("PunNext", "mean"),
        mean_GW=("GW", "mean"),
        mean_esg=("esg_score", "mean")).reset_index()
    cm.to_csv(os.path.join(PATH_RESULTS, "tab4_7_conditional_means.csv"),
              index=False, encoding="utf-8-sig")
    print("\n[条件均值检验] 按 GW 五分组（表4-7）")
    print(cm.round(3).to_string(index=False))
    spearman = stats.spearmanr(cm["gw_q5"], cm["pun_rate"])
    print(f"  单调性 Spearman ρ = {spearman.statistic:+.2f} (p = {spearman.pvalue:.3f})")
    with open(os.path.join(PATH_RESULTS, "tab4_7_monotonicity.txt"), "w") as f:
        f.write(f"Spearman rho = {spearman.statistic:+.3f}, p = {spearman.pvalue:.4f}\n")
    return cm


def market_reaction(df: pd.DataFrame):
    """市场反应：高漂绿组 vs 低漂绿组的月度超额回报差"""
    ret = pd.read_csv(os.path.join(PATH_RAW, "stock_returns.csv"))
    ret = ret.merge(df[["stkcd", "year", "GW"]], on=["stkcd", "year"], how="inner")
    ret["abn"] = ret["ret"] - ret["mkt_ret"]
    ret["gw_half"] = (ret["GW"] > ret["GW"].median()).astype(int)
    g = ret.groupby(["gw_half", "year"])["abn"].mean().unstack()
    g.to_csv(os.path.join(PATH_RESULTS, "tab4_8_market_reaction.csv"),
             encoding="utf-8-sig")
    print("\n[市场反应] 年度平均超额回报（高漂绿=1 vs 低漂绿=0，表4-8）")
    print(g.round(4).to_string())
    for yr in g.columns:
        hi, lo = g[yr][1], g[yr][0]
        t, p = stats.ttest_ind(
            ret.loc[(ret["year"] == yr) & (ret["gw_half"] == 1), "abn"],
            ret.loc[(ret["year"] == yr) & (ret["gw_half"] == 0), "abn"])
        print(f"  {yr}: 高漂绿 {hi:+.4f} vs 低漂绿 {lo:+.4f}  "
              f"差 {hi-lo:+.4f} (t = {t:+.2f}, p = {p:.3f})")


def descriptive(df: pd.DataFrame):
    """描述性统计（表3-2）"""
    cols = ["Talk", "Promo", "Spec", "Hedge", "GW", "esg_score",
            "RD", "Size", "lev", "roa"]
    desc = df[cols].describe().T
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.columns = ["样本数", "均值", "标准差", "最小值", "P25", "中位数", "P75", "最大值"]
    desc = desc.round(3)
    desc.to_csv(os.path.join(PATH_RESULTS, "tab3_2_descriptive.csv"),
                encoding="utf-8-sig")
    print("\n[描述性统计]（表3-2）")
    print(desc.to_string())


def main():
    print("=" * 62)
    print("步骤⑤ 信效度检验与实证分析")
    print("=" * 62)
    df = load_panel()
    df.to_csv(os.path.join(PATH_PROCESSED, "analysis_panel.csv"),
              index=False, encoding="utf-8-sig")

    descriptive(df)
    content_validity(df)
    convergent_validity(df)
    discriminant_validity(df)
    variance_decomposition(df)
    conditional_means(df)
    predictive_validity(df)
    market_reaction(df)

    print(f"\n全部表格已输出至: {PATH_RESULTS}")


if __name__ == "__main__":
    main()
