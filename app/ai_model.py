"""AI 智慧選股模型：三種模式（穩健／均衡／積極）的多因子評分、硬性條件與自動解讀。

模型完全透明：
- 硬性條件（hard）：不符合就不進排行榜。可以直接轉成自訂篩選器的條件。
- 因子權重（weights）：五個因子分數（全市場百分位）的加權平均，是連續的相對排名，
  無法完整轉成「大於／小於」條件，所以帶入篩選器時只帶硬性條件與參考條件。
- 參考條件（reference）：讓使用者看懂模型偏好的典型門檻，只顯示符合與否，不影響排名。
解讀文字是依數據規則產生的，不是語言模型生成，也不是預測。
"""
from __future__ import annotations

from .strategies import C, check

MODES = {
    "stable": {
        "name": "穩健型", "tagline": "重視財務品質與風險控制",
        "desc": "先要求公司有獲利、有現金流、負債不過高、股價波動不在全市場最高的三成，再以品質與估值為主排名。",
        "weights": {"quality": 0.40, "value": 0.25, "trend": 0.15, "growth": 0.10, "momentum": 0.10},
        "hard": [C("base_ok", "is", True, "通過基本門檻（流動性、處置、長期虧損）"), C("eps_ttm", ">", 0, "近四季 EPS > 0"),
                 C("ocf_ttm", ">", 0, "營業現金流 > 0"), C("debt_ratio", "<", 60, "負債比率 < 60%", True),
                 C("vol_1y_pct", "<=", 70, "年化波動率不在全市場最高 30%"), C("avg_value_20", ">", 3e7, "20 日均成交金額 > 3,000 萬")],
        "reference": [C("roe", ">", 12, "ROE > 12%"), C("per", "between", [0, 20], "本益比 < 20"), C("dy", ">=", 3, "殖利率 ≥ 3%"),
                      C("beta", "<", 1, "Beta < 1"), C("close", ">m", "ma120", "收盤 > MA120")],
    },
    "balanced": {
        "name": "均衡型", "tagline": "兼顧品質、成長、估值及動能",
        "desc": "五個因子依品質 30%、成長 25%、估值 20%、動能 15%、趨勢 10% 加權，只要求基本的獲利與資料完整度。",
        "weights": {"quality": 0.30, "growth": 0.25, "value": 0.20, "momentum": 0.15, "trend": 0.10},
        "hard": [C("base_ok", "is", True, "通過基本門檻（流動性、處置、長期虧損）"), C("eps_ttm", ">", 0, "近四季 EPS > 0"),
                 C("coverage", ">=", 4, "至少 4 個因子有資料")],
        "reference": [C("roe", ">", 15, "ROE > 15%"), C("rev_yoy", ">", 10, "月營收年增 > 10%"), C("close", ">m", "ma60", "收盤 > MA60"),
                      C("per", "between", [0, 25], "本益比 < 25"), C("ocf_ttm", ">", 0, "營業現金流 > 0")],
    },
    "aggressive": {
        "name": "積極型", "tagline": "更重視成長及價格動能",
        "desc": "要求營收仍在成長、股價站上季線且成交活絡，排名以成長與動能為主，估值權重最低。",
        "weights": {"growth": 0.35, "momentum": 0.30, "trend": 0.15, "quality": 0.15, "value": 0.05},
        "hard": [C("base_ok", "is", True, "通過基本門檻（流動性、處置、長期虧損）"), C("rev_yoy", ">", 0, "最新月營收年增 > 0"),
                 C("close", ">m", "ma60", "收盤 > MA60"), C("avg_value_20", ">", 3e7, "20 日均成交金額 > 3,000 萬")],
        "reference": [C("rev_yoy", ">", 20, "月營收年增 > 20%"), C("eps_q_yoy", ">", 15, "單季 EPS 年增 > 15%"),
                      C("rs_rank", ">=", 80, "相對強度前 20%"), C("ma_bull", "is", True, "均線多頭排列"), C("vol_ratio", ">", 1.2, "量比 > 1.2")],
    },
}
LABEL = {"quality": "財務品質", "growth": "成長能力", "value": "估值", "momentum": "股價動能", "trend": "趨勢"}
TAG = {"quality": "獲利能力穩定", "growth": "成長表現良好", "value": "估值相對合理", "momentum": "股價動能強勢", "trend": "趨勢向上"}
PAIR = {frozenset(("quality", "growth")): "品質與成長兼具", frozenset(("quality", "value")): "優質且估值合理",
        frozenset(("growth", "momentum")): "成長與動能同步", frozenset(("growth", "value")): "成長股估值合理",
        frozenset(("quality", "momentum")): "優質強勢股", frozenset(("momentum", "trend")): "價格趨勢強勁"}


def _n(x, d=1):
    return "—" if x is None else f"{x:,.{d}f}"


def score(row: dict, mode: str):
    if row.get("complete") is False:  # 缺少重要因子：不排名，避免把缺資料當成 0 分或只看部分資料
        return None
    w = MODES[mode]["weights"]
    tot = got = 0.0
    for k, v in w.items():
        s = row.get(f"score_{k}")
        if s is not None:
            tot += v
            got += s * v
    return round(got / tot, 1) if tot else None


def eligible(row: dict, mode: str) -> bool:
    return all(check(c, row) in (True, "na") for c in MODES[mode]["hard"])


def strengths(r: dict) -> list[str]:
    out = []
    pct = lambda s: f"全市場前 {max(1, round(100 - s))}%"
    if (r.get("score_quality") or 0) >= 70:
        out.append(f"財務品質佳：ROE {_n(r.get('roe'))}%、營業利益率 {_n(r.get('op_margin'))}%（{pct(r['score_quality'])}）")
    if (r.get("score_growth") or 0) >= 70:
        out.append(f"成長動能強：月營收年增 {_n(r.get('rev_yoy'))}%、近三月 {_n(r.get('rev_yoy_3m'))}%")
    if (r.get("score_value") or 0) >= 70:
        out.append(f"估值相對便宜：本益比 {_n(r.get('per'))} 倍、股價淨值比 {_n(r.get('pbr'), 2)} 倍")
    if (r.get("score_momentum") or 0) >= 75:
        out.append(f"相對強勢：12-1 個月報酬 {_n(r.get('mom_12_1'))}%（{pct(r['score_momentum'])}）")
    if (r.get("score_trend") or 0) >= 80:
        out.append("趨勢向上：均線多頭排列、季線上揚")
    if (r.get("fscore") or 0) >= 7:
        out.append(f"財務持續改善：F-Score {int(r['fscore'])} 分")
    if (r.get("div_years") or 0) >= 10 and (r.get("dy") or 0) >= 3:
        out.append(f"穩定配息：連續 {int(r['div_years'])} 年、殖利率 {_n(r.get('dy'))}%")
    return out


def risks(r: dict) -> list[str]:
    out = []
    if r.get("base_ok") and r.get("missing_factors"):
        out.append(f"資料不足：缺少{'、'.join(r['missing_factors'])}資料，暫不列入完整綜合排名")
    if r.get("score_value") is not None and r["score_value"] < 30:
        out.append(f"估值偏貴：本益比 {_n(r.get('per'))} 倍，在全市場屬於較高水位")
    if r.get("rev_yoy") is not None and r["rev_yoy"] < 0:
        out.append(f"營收衰退：最新月營收年減 {_n(-r['rev_yoy'])}%")
    if (r.get("vol_1y_pct") or 0) >= 80:
        out.append(f"股價波動大：年化波動率 {_n(r.get('vol_1y'))}%")
    if not r.get("is_financial") and (r.get("debt_ratio") or 0) > 60:
        out.append(f"負債比率偏高：{_n(r.get('debt_ratio'))}%")
    if (r.get("dist_ma60") or 0) > 20:
        out.append(f"短線漲多：股價高於季線 {_n(r.get('dist_ma60'))}%，留意拉回")
    if (r.get("rsi") or 0) > 80:
        out.append(f"技術面過熱：RSI {_n(r.get('rsi'))}")
    if (r.get("margin_chg_20d") or 0) > 15:
        out.append(f"融資快速增加：20 日 +{_n(r.get('margin_chg_20d'))}%")
    if (r.get("foreign_5d") or 0) < 0 and (r.get("trust_5d") or 0) < 0:
        out.append("外資與投信近 5 日同步賣超")
    if (r.get("earnings_cv") or 0) > 1:
        out.append("獲利波動大：近 8 季淨利起伏明顯")
    if r.get("fcf_ttm") is not None and r["fcf_ttm"] < 0:
        out.append("近四季自由現金流為負")
    if (r.get("coverage") or 5) < 4:
        out.append("部分財報資料不足，評分可信度較低")
    if r.get("exclude_reasons"):
        out.append("未通過基本門檻：" + "、".join(r["exclude_reasons"]))
    return out


def tag(r: dict) -> str:
    sc = {k: r.get(f"score_{k}") for k in LABEL if r.get(f"score_{k}") is not None}
    if not sc:
        return "資料不足"
    adj = {k: v * (0.75 if k == "trend" else 1) for k, v in sc.items()}  # 趨勢分數只有 5 個級距，排序時降低比重
    top = sorted(adj, key=adj.get, reverse=True)[:2]
    if len(top) == 2 and sc[top[1]] >= 70 and frozenset(top) in PAIR:
        return PAIR[frozenset(top)]
    return TAG[top[0]] if sc[top[0]] >= 60 else "各項表現普通"


def annotate(rows: list[dict]) -> None:
    """在每檔股票加上 ai 欄位：各模式分數、名次、標籤、優勢、風險。"""
    for r in rows:
        r["ai"] = {"tag": tag(r), "strengths": strengths(r), "risks": risks(r), "modes": {}}
    for mode in MODES:
        ranked = sorted((r for r in rows if r.get("is_common") and eligible(r, mode) and score(r, mode) is not None),
                        key=lambda r: score(r, mode), reverse=True)
        for i, r in enumerate(ranked, 1):
            r["ai"]["modes"][mode] = {"score": score(r, mode), "rank": i}
        for r in rows:
            if mode not in r["ai"]["modes"]:
                r["ai"]["modes"][mode] = {"score": score(r, mode), "rank": None}


# ---------------- 進場時機：好公司 × 好時機 = 到達買點 ----------------
TRIGGERS = {
    "sig_ma_cross": "MA5 黃金交叉 MA20（3 日內）",
    "sig_kd_cross": "KD 低檔黃金交叉（K < 40）",
    "sig_macd_cross": "MACD 翻正（3 日內）",
    "sig_breakout": "帶量突破 20 日高點（量比 ≥ 1.5）",
    "sig_pullback": "多頭回測月線有撐",
}
TIMING_RISKS = [("rsi", 80, "RSI 過熱（> 80）"), ("dist_ma20", 15, "距月線乖離過大（> 15%）"), ("margin_chg_20d", 20, "融資 20 日暴增（> 20%）")]


def buy_rule(cfg: dict) -> dict:
    return {"min_score": cfg.get("min_score", 55), "mode": cfg.get("mode", "balanced"),
            "desc": "體質條件：通過 AI 均衡型硬性條件且分數 ≥ {s}；時機條件：至少出現一個進場訊號；且沒有過熱風險。".format(s=cfg.get("min_score", 55))}


def timing(rows: list[dict], cfg: dict) -> None:
    rule = buy_rule(cfg)
    for r in rows:
        trig = [k for k in TRIGGERS if r.get(k)]
        hot = [label for k, lim, label in TIMING_RISKS if (r.get(k) or 0) > lim]
        sc = r["ai"]["modes"][rule["mode"]]
        good = sc["rank"] is not None and (sc["score"] or 0) >= rule["min_score"]
        if good and trig and not hot:
            status = "buy"
        elif good and (trig or ((r.get("close") or 0) > (r.get("ma60") or 1e18))):
            status = "watch"
        else:
            status = "none"
        r["timing"] = {"status": status, "triggers": trig, "hot": hot, "quality_ok": good}
        r["ai"]["summary"] = summary(r)


def summary(r: dict) -> str:
    """一句話結論。"""
    t, a = r.get("timing", {}), r["ai"]
    if r.get("exclude_reasons"):
        return f"未通過基本門檻（{'、'.join(r['exclude_reasons'])}），不列入選股。"
    body = a["tag"] if a["tag"] not in ("資料不足", "各項表現普通") else "體質表現普通"
    if t.get("status") == "buy":
        when = f"目前出現「{TRIGGERS[t['triggers'][0]]}」進場訊號"
    elif t.get("hot"):
        when = f"但{t['hot'][0]}，建議等待拉回"
    elif t.get("status") == "watch":
        when = "趨勢仍在，但還沒出現明確的進場訊號，可列入觀察"
    elif r.get("missing_factors"):
        return f"資料不足（缺少{'、'.join(r['missing_factors'])}），暫不列入完整綜合排名與買點判斷。"
    elif not t.get("quality_ok"):
        when = "AI 體質分數未達買點門檻"
    else:
        when = "趨勢偏弱，目前不是進場時機"
    risk = f"；需留意{a['risks'][0].split('：')[0]}" if a["risks"] else ""
    return f"{body}，{when}{risk}。"
