"""選股策略定義與評估。

每個策略 = 一組條件（AND）+ 排名依據。條件格式：
    {"m": 指標, "op": ">"|">="|"<"|"<="|"between"|"is"|">m", "v": 門檻或另一個指標, "label": 顯示文字,
     "skip_financial": 金融業不適用時為 True}
網站前端用同一份定義即時評估與顯示每個條件的結果。
"""
from __future__ import annotations

NOT_TESTED = "尚未驗證"


def C(m, op, v, label, skip_financial=False):
    return {"m": m, "op": op, "v": v, "label": label, "skip_financial": skip_financial}


STRATEGIES = [
    {
        "id": "quality", "name": "優質企業雷達", "group": "核心策略", "priority": 1,
        "tagline": "獲利穩定、財務健全的公司",
        "concept": "與其找便宜的股票，不如先找經營品質良好的公司：高獲利能力、有現金流支持、財務槓桿不過高。",
        "evidence": "Fama-French 五因子模型的獲利能力因子（RMW）、MSCI Quality 因子、Novy-Marx (2013) 毛利獲利能力研究。",
        "risks": "金融業資產負債結構不同，負債比與毛利條件不適用；高 ROE 可能來自高負債，請一併看權益乘數。",
        "conditions": [C("roe", ">", 15, "ROE > 15%"), C("roa", ">", 8, "ROA > 8%"),
                       C("op_margin", ">", 10, "營業利益率 > 10%", True), C("ocf_ttm", ">", 0, "營業現金流 > 0"),
                       C("ni_ttm", ">", 0, "近四季淨利 > 0")],
        "rank_by": "score_quality", "fundamental": True,
    },
    {
        "id": "growth", "name": "營收爆發雷達", "group": "核心策略", "priority": 2,
        "tagline": "營收與獲利快速成長的公司",
        "concept": "利用台灣公司每月公告營收的特性，找營收年增率持續提高、且本業有賺錢的公司。",
        "evidence": "營收與盈餘成長是常見的成長因子；「營收加速」這組具體條件目前沒有可引用的台股超額報酬證據。",
        "risks": "比較基期偏低、一次性訂單、產業循環高點都可能造成短暫高成長。",
        "conditions": [C("rev_yoy", ">", 20, "最新月營收年增 > 20%"), C("rev_yoy_prev", ">", 10, "前一月營收年增 > 10%"),
                       C("rev_yoy", ">m", "rev_yoy_prev", "最新月年增 > 前一月年增（加速）"),
                       C("eps_ttm", ">", 0, "近四季 EPS > 0"), C("op_income_q", ">", 0, "最新季營業利益 > 0")],
        "rank_by": "score_growth", "fundamental": True,
    },
    {
        "id": "momentum", "name": "強勢動能雷達", "group": "核心策略", "priority": 3,
        "tagline": "過去一年相對市場表現強勢的股票",
        "concept": "過去表現強的股票在接下來一段時間傾向維持相對強勢。學術做法以「過去 12 個月、排除最近 1 個月」的報酬排名。",
        "evidence": "Jegadeesh & Titman (1993) 及後續國際市場動能研究。",
        "risks": "動能策略可能突然且劇烈反轉（momentum crash），一定要搭配風險控管。股價未還原除權息。",
        "conditions": [C("rs_rank", ">=", 80, "12-1 動能排名前 20%"), C("close", ">m", "ma60", "收盤 > 60 日均線"),
                       C("avg_value_20", ">", 3e7, "20 日均成交金額 > 3,000 萬")],
        "rank_by": "mom_12_1", "fundamental": False,
    },
    {
        "id": "value", "name": "低估價值雷達", "group": "核心策略", "priority": 4,
        "tagline": "價格相對便宜、基本面尚可的公司",
        "concept": "找市價相對公司獲利偏低的股票，同時要求仍有一定獲利能力與現金流，避免買到「價值陷阱」。",
        "evidence": "Fama-French 價值因子（HML）、MSCI Value 因子。",
        "risks": "便宜可能有理由：產業衰退、獲利高峰。請搭配營收趨勢一起看。",
        "conditions": [C("per", "between", [0, 15], "0 < 本益比 < 15"), C("roe", ">", 10, "ROE > 10%"),
                       C("ocf_ttm", ">", 0, "營業現金流 > 0"), C("ni_ttm", ">", 0, "近四季淨利 > 0")],
        "rank_by": "score_value", "fundamental": True,
    },
    {
        "id": "trend", "name": "多頭趨勢雷達", "group": "核心策略", "priority": 5,
        "tagline": "股價維持上升趨勢的股票",
        "concept": "用均線判斷單一股票本身的價格方向：短、中、長期均線依序排列且季線向上。",
        "evidence": "AQR 跨市場趨勢跟隨研究（研究對象不是台股個股，不能直接當作台股均線有效的證據）。",
        "risks": "盤整行情中均線會反覆產生錯誤訊號，不建議單獨使用。",
        "conditions": [C("close", ">m", "ma20", "收盤 > MA20"), C("ma20", ">m", "ma60", "MA20 > MA60"),
                       C("ma60", ">m", "ma120", "MA60 > MA120"), C("ma60_slope_20", ">", 0, "MA60 高於 20 日前（季線上揚）")],
        "rank_by": "dist_ma60", "fundamental": False,
    },
    {
        "id": "composite", "name": "綜合選股雷達", "group": "核心策略", "priority": 6,
        "tagline": "品質、成長、估值、動能、趨勢綜合評分高的公司",
        "concept": "單一指標都有缺點：低本益比可能是衰退公司、高成長可能太貴、強動能可能在高點。多因子評分把五個面向放在一起比較。",
        "evidence": "MSCI 多因子投資架構。權重（品質 30%、成長 25%、估值 20%、動能 15%、趨勢 10%）是產品設計建議，未經回測。",
        "risks": "分數是相對排名，79 分不代表 79% 上漲機率。",
        "conditions": [C("composite", ">=", 65, "綜合評分 ≥ 65"), C("coverage", ">=", 4, "至少 4 個面向有資料")],
        "rank_by": "composite", "fundamental": True,
    },
    {
        "id": "breakout", "name": "突破起漲雷達", "group": "進階策略", "priority": 7,
        "tagline": "股價剛突破重要區間的股票",
        "concept": "股價突破過去一段時間的高點並伴隨放量，代表可能脫離整理區間。",
        "evidence": "Donchian 通道突破（海龜交易法）。具體型態在台股的有效性需回測。",
        "risks": "假突破、追高、交易成本。",
        "conditions": [C("brk_60", "is", True, "收盤 > 前 60 日最高價"), C("vol_ratio", ">", 1.5, "成交量 > 20 日均量 1.5 倍"),
                       C("close", ">m", "ma60", "收盤 > MA60")],
        "rank_by": "vol_ratio", "fundamental": False,
    },
    {
        "id": "institutional", "name": "法人資金雷達", "group": "進階策略", "priority": 8,
        "tagline": "法人近期資金流向偏正面的股票",
        "concept": "外資與投信同步買超，且股價在月線之上。",
        "evidence": "目前沒有足夠證據認定「外資連買」在台股有穩定超額報酬，建議當作輔助確認條件。",
        "risks": "法人買賣可能出於避險、指數調整或套利。",
        "conditions": [C("foreign_5d", ">", 0, "外資近 5 日買超"), C("trust_5d", ">", 0, "投信近 5 日買超"),
                       C("close", ">m", "ma20", "收盤 > MA20")],
        "rank_by": "inst_ratio_5d", "fundamental": False,
    },
    {
        "id": "dividend", "name": "穩健配息雷達", "group": "進階策略", "priority": 9,
        "tagline": "股利與獲利能力兼具的公司",
        "concept": "不只找殖利率最高的公司，而是同時看配息紀錄、發放率、現金流與獲利能力。",
        "evidence": "股利收益與股利成長投資。",
        "risks": "殖利率可能因股價大跌而被動升高；股利可能超過可持續獲利能力。",
        "conditions": [C("dy", ">=", 4, "現金殖利率 ≥ 4%"), C("div_years", ">=", 5, "連續配息 ≥ 5 年"),
                       C("payout", "between", [30, 80], "股利發放率 30~80%"), C("fcf_ttm", ">", 0, "自由現金流 > 0", True),
                       C("eps_ttm", ">", 0, "近四季 EPS > 0"), C("roe", ">=", 10, "ROE ≥ 10%")],
        "rank_by": "dy", "fundamental": True,
    },
    {
        "id": "lowvol", "name": "低波防禦雷達", "group": "進階策略", "priority": 10,
        "tagline": "歷史波動較低、仍有獲利的股票",
        "concept": "找價格波動相對小、對大盤敏感度低，同時本業有獲利、有現金流的公司。",
        "evidence": "低波動異常（Low Volatility Anomaly）、MSCI Minimum Volatility。",
        "risks": "重點是歷史波動較小，不保證不會下跌。",
        "conditions": [C("vol_1y_pct", "<=", 30, "年化波動率在全市場最低 30%"), C("beta", "<", 1, "Beta < 1"),
                       C("eps_ttm", ">", 0, "近四季 EPS > 0"), C("ocf_ttm", ">", 0, "營業現金流 > 0")],
        "rank_by": "vol_1y", "rank_asc": True, "fundamental": True,
    },
    {
        "id": "fscore", "name": "財務改善雷達（F-Score）", "group": "進階策略", "priority": 11,
        "tagline": "財務體質正在改善的公司",
        "concept": "用 9 項財務檢查（獲利、現金流、槓桿、流動性、營運效率）判斷公司是否在變好。",
        "evidence": "Piotroski (2000)。原始研究針對高帳面市值比股票，套用到台股需回測。",
        "risks": "金融業部分項目不適用；分數只看方向，不看幅度。",
        "conditions": [C("fscore", ">=", 7, "F-Score ≥ 7"), C("pbr", "between", [0, 2], "股價淨值比 < 2")],
        "rank_by": "fscore", "fundamental": True,
    },
    {
        "id": "magic", "name": "神奇公式雷達", "group": "進階策略", "priority": 12,
        "tagline": "又便宜又會賺錢的公司",
        "concept": "同時用「盈餘殖利率」與「資本報酬率」排名，兩者加總名次越前面越好。",
        "evidence": "Greenblatt《打敗大盤的獲利公式》。此處以 E/P 與 ROA 近似原版的 EBIT/EV 與 ROC。",
        "risks": "原版排除金融與公用事業；本系統排除金融業。",
        "conditions": [C("magic_rank", ">=", 90, "神奇公式排名前 10%"), C("is_financial", "is", False, "非金融業")],
        "rank_by": "magic_rank", "fundamental": True,
    },
]
BY_ID = {s["id"]: s for s in STRATEGIES}


def check(cond: dict, row: dict):
    """True / False / None（資料不足）/ "na"（不適用）"""
    if cond.get("skip_financial") and row.get("is_financial"):
        return "na"
    a = row.get(cond["m"])
    v = cond["v"]
    if a is None:
        return None
    op = cond["op"]
    if op == "is":
        return bool(a) == v
    if op == ">m":
        b = row.get(v)
        return None if b is None else a > b
    if op == "between":
        return v[0] < a < v[1]
    return {">": a > v, ">=": a >= v, "<": a < v, "<=": a <= v}[op]


def passes(strategy: dict, row: dict) -> bool:
    if not row.get("base_ok"):
        return False
    res = [check(c, row) for c in strategy["conditions"]]
    return all(r is True or r == "na" for r in res) and any(r is True for r in res)
