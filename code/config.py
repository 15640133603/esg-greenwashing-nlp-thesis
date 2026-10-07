# -*- coding: utf-8 -*-
"""
config.py —— 全局配置与路径管理

项目：企业 ESG 话语的文本测度构建与漂绿识别研究
作者：大数据管理与应用专业 2026 届毕业论文
说明：本文件统一管理项目路径、随机种子、样本区间与绘图样式，
      保证全流程可复现（课件"流程要留痕"要求）。
"""

import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_RAW = os.path.join(BASE_DIR, "data", "raw")
PATH_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
PATH_RESULTS = os.path.join(BASE_DIR, "results")
PATH_FIGURES = os.path.join(BASE_DIR, "figures")

for _p in (PATH_RAW, PATH_PROCESSED, PATH_RESULTS, PATH_FIGURES):
    os.makedirs(_p, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. 研究设计参数
# ---------------------------------------------------------------------------
RANDOM_SEED = 20260101        # 固定随机种子：保证结果可复现
YEAR_START = 2019
YEAR_END = 2023
N_FIRMS = 468                 # 模拟样本企业数（经过筛选后）
YEARS = list(range(YEAR_START, YEAR_END + 1))

# 行业门类（证监会 2012 版行业分类，剔除金融业 J）
INDUSTRIES = {
    "A": "农、林、牧、渔业",
    "B": "采矿业",
    "C": "制造业",
    "D": "电力、热力、燃气及水生产和供应业",
    "E": "建筑业",
    "F": "批发和零售业",
    "G": "交通运输、仓储和邮政业",
    "I": "信息传输、软件和信息技术服务业",
    "K": "房地产业",
    "M": "科学研究和技术服务业",
    "N": "水利、环境和公共设施管理业",
    "R": "文化、体育和娱乐业",
}
# 重污染行业代码（用于异质性分析）
HEAVY_POLLUTION = {"B", "C", "D"}

# 是否为重污染行业的概率（制造业样本占比需较高，贴近 A 股实际）
INDUSTRY_WEIGHTS = {
    "A": 0.02, "B": 0.03, "C": 0.60, "D": 0.04, "E": 0.03,
    "F": 0.04, "G": 0.04, "I": 0.09, "K": 0.03, "M": 0.04,
    "N": 0.02, "R": 0.02,
}

# ---------------------------------------------------------------------------
# 3. 绘图样式（中文友好，统一配色）
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 尝试注册可用中文字体（优先思源黑体 / 思源宋体）
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
CJK_FONT_NAME = "sans-serif"
for _f in _CJK_CANDIDATES:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        CJK_FONT_NAME = font_manager.FontProperties(fname=_f).get_name()
        plt.rcParams["font.sans-serif"] = [CJK_FONT_NAME] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.edgecolor"] = "#4a4a4a"
plt.rcParams["axes.linewidth"] = 0.8

# 论文统一色板（A 股习惯：涨红跌绿不适用于学术图，此处用学术冷色系）
PALETTE = ["#2E5A88", "#C0504D", "#4F81BD", "#9BBB59",
           "#8064A2", "#F79646", "#4BACC6", "#7F7F7F"]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """固定全部随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
