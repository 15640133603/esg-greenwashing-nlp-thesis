# -*- coding: utf-8 -*-
"""
heterogeneity.py —— 异质性分析与稳健性补充（真实运行）

H1: 重污染行业 vs 非重污染行业，GW 的处罚预测力差异
H2: 国企 vs 民企
H3: 分年度回归（政策趋强背景下的时序变化）

输出 results/tab5_x_heterogeneity.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATH_PROCESSED, PATH_RESULTS


def winsorize(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def run_logit(d, label, rows):
    dd = d[["GW_w", "Talk_w", "Size", "lev", "roa", "soe",
            "PunNext", "year"]].dropna()
    if dd["PunNext"].nunique() < 2:
        return
    # 组内无变异的列必须剔除（如国企子样本中 soe 恒为 1）
    formula = "PunNext ~ GW_w + Talk_w + Size + lev + roa"
    if dd["soe"].nunique() > 1:
        formula += " + soe"
    formula += " + C(year)"
    m = smf.logit(formula, data=dd).fit(disp=0)
    rows.append({"分组": label, "GW系数": round(m.params["GW_w"], 3),
                 "OR": round(np.exp(m.params["GW_w"]), 3),
                 "z": round(m.tvalues["GW_w"], 2),
                 "p": round(m.pvalues["GW_w"], 4),
                 "N": int(m.nobs)})


def main():
    df = pd.read_csv(os.path.join(PATH_PROCESSED, "analysis_panel.csv"))
    df["GW_w"] = winsorize(df["GW"])
    df["Talk_w"] = winsorize(df["Talk"])

    rows = []
    run_logit(df, "全样本", rows)
    for flag, name in [(1, "重污染行业"), (0, "非重污染行业")]:
        run_logit(df[df["heavy_pollution"] == flag], name, rows)
    for flag, name in [(1, "国有企业"), (0, "民营企业")]:
        run_logit(df[df["soe"] == flag], name, rows)
    for yr in sorted(df["year"].unique()):
        run_logit(df[df["year"] == yr], f"{yr}年", rows)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PATH_RESULTS, "tab5_x_heterogeneity.csv"),
               index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
