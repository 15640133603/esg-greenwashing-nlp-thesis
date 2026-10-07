# 企业ESG话语的文本测度构建与漂绿识别研究

> 2026届本科毕业论文（模拟稿）· 大数据管理与应用专业
> 方法基础：管理研究中的 NLP 测度构建方法（五步法）
> **语料获取 → 文本预处理 → 特征提取 → 指标构建 → 信效度检验**

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![论文](https://img.shields.io/badge/论文-Word%20%2B%20PDF-orange)]()

## 研究简介

以 2019—2023 年沪深 A 股 468 家上市公司为样本，构建了一套**可复现的 ESG 话语中文文本测度体系**，用于识别企业"说得多、做得少"的漂绿（greenwashing）行为。

### 核心测度

| 测度 | 符号 | 含义 |
|---|---|---|
| 话语强度 | `Talk` | 实质+宣传词频 / ESG报告词数 ×100 |
| 宣传漂移度 | `Promo` | 宣传词占绿色话语比重 |
| 话语具体性 | `Spec` | 实质词千句密度（可验证性代理） |
| 模糊承诺度 | `Hedge` | 模糊限定词千词密度 |
| **复合漂绿测度** | `GW` | PCA 第一主成分（Promo, Hedge, −Spec） |

### 关键结果（全部真实运行产出）

- **聚合效度**：Talk ↔ 华证ESG评级 r = **+0.386***；GW ↔ 评级 r = **−0.382***
- **预测效度**：GW → 下一年环境处罚 **OR = 1.182***（Logit）
- **经济后果**：高漂绿组合月度超额回报多数年份显著为负
- **方差分解**：GW 企业间方差占比 **41.9%**，信号充足
- LDA 主题数按连贯性 C_v 选定 **K = 4**（C_v = 0.550）

## 仓库结构

```
esg-greenwashing-nlp/
├── 论文全文.docx / 论文全文.pdf      # 论文成品（Word 可编辑 + PDF 排版稿）
├── 论文全文.html                    # 网页版（浏览器直接阅读）
├── docs/
│   └── 论文全文.md                  # 论文源稿（Markdown）
├── code/                            # 全流程代码（五步对应）
│   ├── config.py                    #   全局配置与路径
│   ├── gen_corpus.py                #   ① 语料获取（模拟）
│   ├── esg_lexicon.py               #   ★ ESG话语领域词典（6维度187词）
│   ├── pipeline_features.py         #   ② 预处理 + ③ 特征提取（词典/TF-IDF/W2V/LDA）
│   ├── build_measures.py            #   ④ 指标构建（含PCA合成GW）
│   ├── validity_tests.py            #   ⑤ 信效度检验 + 回归 + 市场反应
│   ├── heterogeneity.py             #   异质性分析（行业/产权/年度）
│   ├── make_figures.py              #   论文全部插图
│   ├── md2docx.py / md2pdf.py       #   论文格式转换
│   └── count_words.py               #   字数统计
├── data/
│   ├── raw/                         # 原始语料与模拟数据库
│   └── processed/                   # 分词特征、测度面板、模型文件
├── results/                         # 论文全部表格（CSV）
└── figures/                         # 论文全部插图（PNG, 200dpi）
```

## 快速复现

```bash
pip install -r requirements.txt

python code/gen_corpus.py          # ① 语料获取（模拟巨潮披露文本）
python code/pipeline_features.py   # ② 预处理 + ③ 特征提取
python code/build_measures.py      # ④ 指标构建
python code/validity_tests.py      # ⑤ 信效度检验与实证
python code/heterogeneity.py       # 异质性分析
python code/make_figures.py        # 论文插图
python code/md2docx.py             # 生成 Word
python code/md2pdf.py              # 生成 PDF
```

每一步输出均落盘（`data/` → `results/` → `figures/`），保证"流程留痕、可复现"。

## 数据来源说明

本项目为**方法预演论文（模拟稿）**：语料与财务数据依据既定经济结构（潜变量模型）模拟生成，用于完整验证五步法流程。正式研究中，将真实数据放入 `data/raw/` 并保持文件名不变，全流程代码**无需修改**即可运行：

| 文件 | 真实研究中的来源 |
|---|---|
| `esg_reports.csv` | 巨潮资讯网：ESG报告/社会责任报告/年报MD&A 句子级文本 |
| `esg_ratings.csv` | 华证指数：ESG评级得分（年末） |
| `financials.csv` | CSMAR 国泰安：财务与治理变量 |
| `punishments.csv` | 生态环境部门公开信息：环境行政处罚 |
| `stock_returns.csv` | CSMAR/Wind：股票月度回报 |

抓取合规：遵守 robots 协议、控制请求频率、保留原始文件与脚本备查。

## 论文规范自查（对照学院要求）

| 指标 | 要求 | 本稿 |
|---|---|---|
| 正文字数 | ≥ 10,000 | **15,800+** ✓ |
| 参考文献 | ≥ 15 篇 | **20 篇** ✓ |
| 外文文献 | ≥ 5 篇 | **10 篇** ✓ |
| 近三年文献占比 | ≥ 60% | ✓ |
| 信效度检验 | 完整 | 内容/聚合/区分/预测四维全做 ✓ |

## 方法引用

- 胡楠, 薛付婧, 王昊楠. 管理者短视主义影响企业长期投资吗？——基于文本分析和机器学习[J]. 管理世界, 2021, 37(5): 139-156.（词典构建流程范本）
- Loughran T, McDonald B. When is a liability not a liability?[J]. Journal of Finance, 2011.（词典法标准）
- Lyon T P, Montgomery A W. The means and end of greenwash[J]. Organization & Environment, 2015.（漂绿理论）

## License

MIT License。词表与代码可用于学术研究，引用时请注明本仓库及上述方法文献。
