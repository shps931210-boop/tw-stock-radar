"""買賣計畫（2026-11-18）：把「這支現在能不能買、價格合不合理、什麼時候進、什麼時候出」整理成一句結論和幾個價位。

使用者的目的：清楚知道哪些股票現在能投資、能進場、什麼時候出場；0050 只是用來檢查這些訊號有沒有用的標準。

原則
- 結論只由三件事組合，三件事分開列出，看得到每一項為什麼：
  1. 好公司：AI 均衡型綜合分數（全市場百分位）。
  2. 好價格：本益比在自己近兩年的位置（歷史估值區間），沒有區間就看同業本益比。
  3. 好時機：波段進場規則（app/strategy.py，和回測同一個函式）。
- 價位全部由規則算出來（估值區間、ATR 停損、季線），不由 AI 產生。
- 結論措辭是「可以考慮」而不是「建議買進」；每一張卡都附這套規則過去的實際表現（贏不贏 0050），沒贏就照實寫。
"""
from __future__ import annotations

GOOD_CO = 60        # 均衡型分數 ≥ 60：全市場前 40%
WEAK_CO = 45        # < 45：體質偏弱
CHEAP, RICH, HOT = 30, 70, 85   # 本益比在自己近兩年的百分位
CHASE = 3.0         # 隔天開盤比今天收盤高 3% 以上就不追

VERDICT = {
    "buy": ("可以考慮進場", "公司體質、價格、時機三項都符合"),
    "wait_price": ("好公司，等價格回落", "體質符合，但價格偏貴"),
    "wait_signal": ("好公司，等進場訊號", "體質與價格符合，還沒出現回檔轉強的進場訊號"),
    "watch": ("觀察", "條件只符合一部分"),
    "avoid": ("先不要買", "體質偏弱、估值過熱或趨勢轉弱"),
    "na": ("資料不足", "缺少評分、估值或股價資料，無法判斷"),
}


def _r(x, d=2):
    return None if x is None else round(float(x), d)


def build(r: dict, st: dict, v: dict) -> dict:
    """r：快照的一列；st：strategy.status()；v：估值區間（_valuation_fill 之後）。回傳精簡的買賣計畫（放進快照，首頁清單也用）。"""
    close = r.get("close")
    score = ((r.get("ai") or {}).get("modes") or {}).get("balanced", {}).get("score") if r.get("complete") else None
    pct = v.get("pct") if v.get("p50") is not None else None
    peer, pe = v.get("peer"), v.get("pe")
    sw = (st.get("swing") or {}) if st.get("ready") else {}
    state = sw.get("state", "na") if st.get("ready") else "na"
    hot = list((r.get("timing") or {}).get("hot") or [])
    dist60 = r.get("dist_ma60")
    ma60 = close / (1 + dist60 / 100) if close and dist60 is not None and dist60 > -100 else None

    # 三件事
    co = None if score is None else "good" if score >= GOOD_CO else "weak" if score < WEAK_CO else "mid"
    if pct is not None:
        price = "cheap" if pct <= CHEAP else "hot" if pct >= HOT else "rich" if pct >= RICH else "fair"
    elif pe is not None and peer:
        price = "fair" if pe <= peer else "rich"
    else:
        price = None
    timing = {"triggered": "go", "setup": "near", "none": "no"}.get(state, None)
    down = dist60 is not None and dist60 < 0 and (st.get("trend") or {}).get("label") == "偏弱"

    if co is None and price is None:
        key = "na"
    elif co == "weak" or price == "hot" or down:
        key = "avoid"
    elif co == "good" and price in ("rich",):
        key = "wait_price"
    elif co == "good" and timing == "go" and not hot:
        key = "buy"
    elif co == "good" and price in ("cheap", "fair", None):
        key = "wait_signal"
    else:
        key = "watch"

    why = []
    if co:
        why.append(f"體質：AI 均衡型 {score:.0f} 分（{'全市場前 40%' if co == 'good' else '偏弱' if co == 'weak' else '中等'}）")
    else:
        why.append("體質：沒有完整 AI 分數")
    if pct is not None:
        why.append(f"價格：本益比 {pe:.1f} 倍，比自己近兩年 {pct:.0f}% 的日子貴" + ("（偏低）" if price == "cheap" else "（過熱）" if price == "hot" else "（偏貴）" if price == "rich" else "（中間）"))
    elif price:
        why.append(f"價格：本益比 {pe:.1f} 倍，{'低於' if price == 'fair' else '高於'}同業中位數 {peer:.1f} 倍（沒有自己的歷史區間）")
    else:
        why.append("價格：沒有可用的估值資料")
    why.append({"go": "時機：今天收盤後波段進場條件成立", "near": "時機：多頭趨勢中，等回檔轉強",
                "no": "時機：" + ((sw.get("miss") or ["波段進場條件未成立"])[0])}.get(timing, "時機：股價資料不足，無法判斷"))
    if down:
        why.append("趨勢偏弱且股價在季線之下")
    why += [f"注意：{h}" for h in hot[:1]]

    plan = {"key": key, "label": VERDICT[key][0], "why": why,
            "fair": {"low": _r(v.get("low")), "base": _r(v.get("base")), "high": _r(v.get("high"))} if v.get("p50") is not None else None,
            "stop": _r(sw.get("stop")), "ma60": _r(ma60), "first_target": _r(sw.get("target")),
            "risk_pct": _r(sw.get("risk_pct"), 1), "max_days": 40}
    if key == "buy":
        plan["entry"] = f"明天開盤買；開盤價高於 {close * (1 + CHASE / 100):.2f}（今天收盤 +{CHASE:.0f}%）就不追"
    elif key == "wait_price" and plan["fair"]:
        plan["entry"] = f"股價回到合理價 {plan['fair']['base']:.2f} 以下、且出現波段進場訊號再考慮"
    elif key == "wait_signal":
        plan["entry"] = "等波段進場訊號（回檔到月線附近後轉強）。目前：" + (sw.get("miss") or ["條件還沒成立"])[0]
    return plan
