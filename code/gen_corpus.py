# -*- coding: utf-8 -*-
"""
gen_corpus.py —— 第一步：语料获取（模拟）

按照 NLP 测度构建五步法的"语料获取"要求，构造一套结构逼真的
企业-年度面板语料。真实研究中的对应来源为：

  1. ESG 报告 / 社会责任报告文本   —— 巨潮资讯网、企业官网（自愿披露）
  2. 年报 MD&A 文本              —— 巨潮资讯网（强制披露）
  3. 华证 ESG 评级得分            —— 华证指数（季度更新，本文取年末值）
  4. 财务与治理数据              —— CSMAR 国泰安数据库
  5. 股票月度回报                —— CSMAR / Wind

本脚本依据既定的经济逻辑（潜变量结构）生成模拟数据：
  - 每家企业有一个"真实绿色行为水平" latent_green（不直接可观测）
  - 文本话语强度 = f(latent_green) + 话语夸大噪声（漂绿成分）
  - ESG 评级 = g(latent_green) + 评级噪声
  - 环境处罚 = h(latent_green)（负向）
由此保证：文本测度与评级正相关（聚合效度）、与处罚负相关（预测效度）、
"言行差距"变量与后续处罚正相关（漂绿的经济含义）。

输出文件（data/raw/）：
  firms.csv              企业静态信息
  esg_reports.csv        ESG 报告文本语料（句子级，含来源标注）
  mda_texts.csv          年报 MD&A 语料（句子级）
  esg_ratings.csv        华证式 ESG 评级得分
  financials.csv         财务与治理变量
  punishments.csv        环境行政处罚记录
  stock_returns.csv      股票月度回报（用于市场反应检验）
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BASE_DIR, HEAVY_POLLUTION, INDUSTRIES, INDUSTRY_WEIGHTS,
                    N_FIRMS, PATH_RAW, PATH_PROCESSED, RANDOM_SEED, YEARS,
                    set_seed)

set_seed(RANDOM_SEED)

RNG = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 词库：用于生成"像真的"的中文 ESG 话语句子
# ---------------------------------------------------------------------------
SUBSTANTIVE_GREEN = [
    # 实质性行动话语（有具体数字、项目、投入）
    "报告期内公司投入{}亿元用于{}废水处理系统升级改造",
    "公司建成{}个国家级绿色工厂，累计完成{}项清洁生产工艺改造",
    "本年度新增分布式光伏装机容量{}兆瓦，年减排二氧化碳约{}万吨",
    "公司上线能源管理平台，单位产值综合能耗同比下降{}%",
    "报告期内完成{}条生产线余热回收改造，节约标准煤{}吨",
    "公司设立环境管理委员会，专职环保人员{}人，环保培训覆盖{}人次",
    "公司投资{}万元建设中水回用系统，水资源重复利用率提升至{}%",
    "报告期内公司绿色产品收入占总营收比重达{}%，较上年提升{}个百分点",
    "公司通过ISO14001环境管理体系认证，覆盖{}家子公司",
    "本年度淘汰落后产能{}万吨，关停高耗能生产线{}条",
    "公司开展供应商ESG审核，覆盖一级供应商{}家，占比{}%",
    "报告期内环保设施运行费用{}万元，占营业收入比重{}%",
    "公司建成覆盖全集团的能耗在线监测系统，接入重点用能单位{}家",
    "报告期内公司完成{}个产品的全生命周期碳足迹核算",
    "公司参与{}项国家绿色制造标准制定，获绿色专利授权{}项",
]

PROMOTIONAL_GREEN = [
    # 宣传性话语（空泛、形容词多、无具体数字）——漂绿信号
    "公司始终坚持绿色发展理念，积极践行社会责任，致力于成为行业标杆",
    "公司高度重视环境保护，多措并举，成效显著，得到社会各界广泛赞誉",
    "公司以双碳目标为引领，全面推进绿色低碳转型，迈上新台阶",
    "公司深化绿色布局，厚植高质量发展底色，生态底色更加鲜明",
    "公司持续提升ESG治理水平，打造可持续发展标杆企业",
    "报告期内，公司绿色发展战略深入推进，各项环保工作取得积极成效",
    "公司秉承人与自然和谐共生理念，擦亮绿色发展名片",
    "公司积极拥抱双碳机遇，绿色转型之路行稳致远",
    "公司不断完善环境管理体系，环保绩效稳居行业先进水平",
    "公司绿色文化深入人心，全员环保意识显著增强",
    "公司加快培育绿色新动能，塑造可持续发展新优势",
    "公司坚持生态优先，绿色发展迈出坚实步伐",
    "公司聚力打造绿色供应链，生态圈建设再发力",
    "公司以高度的责任感和使命感，推动ESG事业蓬勃发展",
]

RISK_SENTENCES = [
    "报告期内公司{}面临环保监管趋严风险，污染物排放标准可能进一步提高",
    "若{}碳配额收紧，公司能源成本存在上行压力",
    "公司部分{}生产基地位于环境敏感区域，存在突发环境事件风险",
    "极端天气可能对{}公司供应链稳定性造成不利影响",
    "环保税与排污权交易成本上升，或对{}公司盈利水平形成一定压力",
]

GENERIC_MDA = [
    "报告期内，公司围绕年度经营计划，扎实推进各项重点工作",
    "公司实现营业收入{}亿元，同比增长{}%；归母净利润{}亿元",
    "面对复杂多变的外部环境，公司坚持稳中求进，经营业绩保持稳健",
    "公司持续优化产品结构，毛利率较上年提升{}个百分点",
    "报告期末公司资产总额{}亿元，资产负债率{}%",
]

POLLUTION_OBJECTS = ["化工", "印染", "电镀", "焦化", "水泥", "造纸",
                     "钢铁", "发酵", "喷涂", "采矿"]


def _sent_substantive(rng):
    t = rng.choice(SUBSTANTIVE_GREEN)
    nums = sorted(rng.uniform(0.3, 9.9, size=4), key=lambda x: -x)
    try:
        return t.format(*[f"{n:.1f}" for n in nums[:t.count("{}")]])
    except IndexError:
        return t.format(*[f"{n:.1f}" for n in nums], 12)


def _sent_promo(rng):
    return rng.choice(PROMOTIONAL_GREEN)


def _sent_risk(rng):
    t = rng.choice(RISK_SENTENCES)
    return t.format(rng.choice(POLLUTION_OBJECTS))


def _sent_mda(rng):
    t = rng.choice(GENERIC_MDA)
    n = rng.uniform(3, 300, size=5)
    try:
        return t.format(f"{n[0]:.1f}", f"{n[1]*100:.1f}", f"{n[2]:.1f}",
                        f"{min(n[3]*100, 40):.1f}", f"{n[4]:.0f}")
    except Exception:
        return t


# ---------------------------------------------------------------------------
# 1. 企业静态信息
# ---------------------------------------------------------------------------
def gen_firms() -> pd.DataFrame:
    codes = []
    for i in range(N_FIRMS):
        codes.append(f"{RNG.integers(1, 9999):06d}")
    codes = sorted(set(codes))
    while len(codes) < N_FIRMS:
        c = f"{RNG.integers(1, 9999):06d}"
        if c not in codes:
            codes.append(c)

    ind = RNG.choice(list(INDUSTRIES.keys()), size=N_FIRMS, p=list(INDUSTRY_WEIGHTS.values()))
    firms = pd.DataFrame({
        "stkcd": codes[:N_FIRMS],
        "industry": ind,
        "list_year": RNG.integers(2001, 2018, size=N_FIRMS),
        "soe": RNG.binomial(1, 0.38, size=N_FIRMS),   # 国企占比约 38%
    })
    firms["heavy_pollution"] = firms["industry"].isin(HEAVY_POLLUTION).astype(int)
    firms["size_base"] = RNG.lognormal(mean=22.3, sigma=1.15, size=N_FIRMS)  # 总资产(元)
    firms["leverage_base"] = np.clip(RNG.normal(0.42, 0.16, size=N_FIRMS), 0.05, 0.92)
    firms["green_capability"] = RNG.normal(0, 1, size=N_FIRMS)   # 真实绿色行为潜质
    firms["talkativeness"] = RNG.normal(0, 0.8, size=N_FIRMS)    # 话语夸大倾向（漂绿人格）
    return firms


# ---------------------------------------------------------------------------
# 2. 企业-年度核心潜变量与观测数据
# ---------------------------------------------------------------------------
def gen_panel(firms: pd.DataFrame) -> dict:
    rows_fin, rows_esg, rows_txt, rows_pun, rows_ret = [], [], [], [], []

    for _, f in firms.iterrows():
        firm_mu = f["green_capability"]
        talk = f["talkativeness"]
        # 绿色行为随时间缓慢改善（双碳政策 2020 年后边际趋强）
        for t, year in enumerate(YEARS):
            policy = 0.10 * max(0, year - 2019)          # 政策时间趋势
            shock = RNG.normal(0, 0.25)                  # 年度随机冲击
            latent_green = firm_mu + policy + shock      # 真实绿色行为水平

            # --- 话语生成参数 ---
            # 话语强度 = 真实行为 + 夸大倾向 + 噪声 → 天然产生"言行差距"
            exaggeration = 0.55 * talk + RNG.normal(0, 0.30)
            talk_green = 0.62 * latent_green + exaggeration  # 话语潜变量

            # 句子数量与文本篇幅
            n_sub = max(2, int(np.exp(1.2 + 0.55 * max(talk_green, -1)) + RNG.normal(0, 1.2)))
            n_pro = max(1, int(np.exp(1.0 + 0.75 * max(exaggeration, -0.5)) + RNG.normal(0, 1.0)))
            n_risk = max(0, int(3 - 1.1 * latent_green + RNG.normal(0, 1.0)))
            n_mda = RNG.integers(14, 30)

            sentences = []
            for _ in range(n_sub):
                sentences.append(("ESG报告-环境章节", _sent_substantive(RNG), "substantive"))
            for _ in range(n_pro):
                sentences.append(("ESG报告-董事长致辞", _sent_promo(RNG), "promotional"))
            for _ in range(n_risk):
                sentences.append(("ESG报告-风险因素", _sent_risk(RNG), "risk"))
            for _ in range(n_mda):
                sentences.append(("年报MD&A", _sent_mda(RNG), "mda"))
            RNG.shuffle(sentences)
            for sec, s, kind in sentences:
                rows_txt.append({
                    "stkcd": f["stkcd"], "year": year,
                    "source": sec.split("-")[0], "section": sec,
                    "sentence_type": kind, "text": s,
                    "doc_chars": len(s),
                })

            # --- ESG 评级（华证式 0~100，含评级噪声与行业基准） ---
            rating = np.clip(
                58 + 9.5 * latent_green - 1.8 * f["heavy_pollution"]
                + 2.2 * f["soe"] + RNG.normal(0, 4.2), 15, 95)
            rows_esg.append({"stkcd": f["stkcd"], "year": year,
                             "esg_score": round(rating, 1)})

            # --- 财务数据 ---
            size = f["size_base"] + 0.04 * t + RNG.normal(0, 0.05)
            lev = np.clip(f["leverage_base"] + 0.01 * t + RNG.normal(0, 0.035), 0.03, 0.95)
            roa = np.clip(0.045 + 0.012 * latent_green - 0.02 * lev + RNG.normal(0, 0.03), -0.15, 0.25)
            rd = np.clip(0.021 + 0.011 * latent_green + RNG.normal(0, 0.015), 0.0002, 0.2)
            rows_fin.append({
                "stkcd": f["stkcd"], "year": year,
                "asset": round(size, 2),            # ln 总资产
                "lev": round(lev, 3), "roa": round(roa, 4),
                "rd_intensity": round(rd, 4),       # 研发投入强度
                "board_size": int(RNG.integers(5, 15)),
                "indep_ratio": round(np.clip(RNG.normal(0.375, 0.05), 0.3, 0.6), 3),
                "dual": int(RNG.binomial(1, 0.3)),
            })

            # --- 环境处罚（泊松，受真实绿色行为负向影响） ---
            lam = np.exp(-1.05 - 0.55 * latent_green + 0.35 * f["heavy_pollution"])
            n_pun = RNG.poisson(max(lam, 0.02))
            for k in range(n_pun):
                rows_pun.append({
                    "stkcd": f["stkcd"], "year": year + 1,  # 处罚滞后一年出现
                    "fine_wan": round(float(RNG.gamma(2.0, 18.0)), 1),
                    "type": RNG.choice(["超标排放", "未批先建", "固废违规", "噪声扰民", "废水直排"]),
                })

            # --- 月度股票回报（用于事件研究/市场反应） ---
            for m in range(1, 13):
                mkt = RNG.normal(0.004, 0.055)
                # 漂绿暴露的公司，处罚月前后回报更差（弱式有效市场对信息定价）
                greenwash_premium = -0.006 * max(talk, 0)
                ret = mkt + 0.012 * latent_green + greenwash_premium + RNG.normal(0, 0.075)
                rows_ret.append({"stkcd": f["stkcd"], "year": year, "month": m,
                                 "ret": round(ret, 4), "mkt_ret": round(mkt, 4)})

    txt = pd.DataFrame(rows_txt)
    esg = pd.DataFrame(rows_esg)
    fin = pd.DataFrame(rows_fin)
    pun = pd.DataFrame(rows_pun)
    ret = pd.DataFrame(rows_ret)

    # 保存到 raw
    txt.to_csv(os.path.join(PATH_RAW, "esg_reports.csv"), index=False, encoding="utf-8-sig")
    esg.to_csv(os.path.join(PATH_RAW, "esg_ratings.csv"), index=False, encoding="utf-8-sig")
    fin.to_csv(os.path.join(PATH_RAW, "financials.csv"), index=False, encoding="utf-8-sig")
    pun.to_csv(os.path.join(PATH_RAW, "punishments.csv"), index=False, encoding="utf-8-sig")
    ret.to_csv(os.path.join(PATH_RAW, "stock_returns.csv"), index=False, encoding="utf-8-sig")

    # 企业静态表
    firms_out = firms[["stkcd", "industry", "list_year", "soe", "heavy_pollution"]].copy()
    firms_out["industry_name"] = firms_out["industry"].map(INDUSTRIES)
    firms_out.to_csv(os.path.join(PATH_RAW, "firms.csv"), index=False, encoding="utf-8-sig")

    return {"txt": txt, "esg": esg, "fin": fin, "pun": pun, "ret": ret}


def main():
    print("=" * 62)
    print("步骤① 语料获取（模拟巨潮资讯网披露文本与 CSMAR 财务数据）")
    print("=" * 62)
    firms = gen_firms()
    data = gen_panel(firms)

    txt, esg, fin, pun = data["txt"], data["esg"], data["fin"], data["pun"]
    n_obs = len(esg)
    print(f"企业数            : {firms.shape[0]}")
    print(f"样本区间          : {YEARS[0]}-{YEARS[-1]}（共 {len(YEARS)} 年）")
    print(f"企业-年度观测     : {n_obs}")
    print(f"句子级语料条数    : {len(txt)}（共 {txt['doc_chars'].sum():,} 字）")
    print(f"  其中 ESG 报告   : {(txt['source']=='ESG报告').sum():,} 条")
    print(f"  其中 年报 MD&A  : {(txt['source']=='年报MD&A').sum():,} 条")
    print(f"ESG 评级记录      : {len(esg)}")
    print(f"财务观测          : {len(fin)}")
    print(f"环境处罚记录      : {len(pun)}")
    print(f"股票月度回报      : {len(data['ret'])}")
    print(f"\n输出目录: {PATH_RAW}")
    for fn in sorted(os.listdir(PATH_RAW)):
        print(f"  - {fn}")


if __name__ == "__main__":
    main()
