"""流動性等級與推薦資格（V27）：原始 AI 分數不動，另外決定「能不能列入預設 AI 精選」。

- 原始五因子分數、排名（ai.modes）與 ML 預測完全不改，研究與回測都看得到原始結果。
- 流動性只決定推薦資格（ai.rec）：E、F、資料不足 → 不列入預設 AI 精選；D → 不能進入前 TOP 名（最高推薦層級）。
- 網站上線與歷史回測呼叫同一個 order()，規則只有一份。
- 門檻是研究用的分級，在看回測結果前寫下（2026-10-17），不是最佳化出來的數字。
"""
from __future__ import annotations

GRADES = (("A", 1e9), ("B", 3e8), ("C", 1e8), ("D", 3e7), ("E", 1e7), ("F", 0.0))
ORDER = "ABCDEF"
RISK = {"A": "低", "B": "低", "C": "中", "D": "偏高", "E": "高", "F": "很高", "?": "未知"}
LEVEL = {"A": "很高", "B": "高", "C": "中", "D": "偏低", "E": "低", "F": "很低", "?": "資料不足"}
MIN_DAYS = 10        # 近 20 日少於 10 天有成交資料 → 等級未知（不當成流動性足夠）
SPIKE = 0.5          # 中位數不到平均的一半：成交集中在少數幾天，改用中位數分級
LOW_DAYS = 5         # 近 20 日有 5 天以上單日成交金額 < 500 萬 → 降一級
ZERO_DAYS = 3        # 近 60 日有 3 天以上零成交 → 降一級
TOP = 10             # 最高推薦層級＝預設 AI 精選前 10 名
VOL_HI, VOL_TOP = 80, 90   # 研究版 D：流動性 D 且波動率前 20% 剔除；流動性 C 且波動率前 10% 降為限制推薦
VARIANTS = {
    "A": "原始模型（不看流動性）",
    "B": "排除流動性 E、F 與資料不足",
    "C": "B＋流動性 D 不進前 10 名（網站目前採用）",
    "D": "C＋流動性與波動率聯合限制",
}
LIVE = "C"


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _grade_of(v: float) -> str:
    return next(g for g, lim in GRADES if v >= lim)


def grade(r: dict) -> dict:
    """單一股票的流動性等級。資料缺漏一律是「?」，不會被當成流動性足夠。"""
    avg, med = _num(r.get("avg_value_20")), _num(r.get("value_med_20"))
    days = r.get("value_days_20") or 0
    out = {"grade": "?", "base_grade": None, "avg_value_20": avg, "value_med_20": med,
           "low_value_days_20": r.get("low_value_days_20"), "zero_vol_days_60": r.get("zero_vol_days_60"),
           "value_cv_20": _num(r.get("value_cv_20")), "notes": []}
    if avg is None or med is None or days < MIN_DAYS:
        out["notes"].append("成交資料不足 10 天，無法判斷流動性")
    else:
        g = out["base_grade"] = _grade_of(avg)
        if med < avg * SPIKE:
            g = max(g, _grade_of(med), key=ORDER.index)
            out["notes"].append("成交金額集中在少數幾天（中位數不到平均的一半），改用中位數分級")
        if (r.get("low_value_days_20") or 0) >= LOW_DAYS or (r.get("zero_vol_days_60") or 0) >= ZERO_DAYS:
            g = ORDER[min(ORDER.index(g) + 1, len(ORDER) - 1)]
            out["notes"].append("近期常有低量或零成交的日子，降一級")
        out["grade"] = g
    out["risk"] = RISK[out["grade"]]
    out["level"] = LEVEL[out["grade"]]
    return out


def status(g: str, vol_pct, variant: str = LIVE) -> tuple[str, str | None]:
    """推薦資格：normal（正常）、capped（不進前 10 名）、excluded（不列入預設精選），以及原因。"""
    if variant == "A":
        return "normal", None
    if g in ("E", "F", "?"):
        return "excluded", ("流動性資料不足，不能確認買賣得到" if g == "?" else f"流動性 {g}（{LEVEL[g]}），成交量太小，不列入預設 AI 精選")
    vol = _num(vol_pct)
    if variant == "D" and g == "D" and vol is not None and vol >= VOL_HI:
        return "excluded", f"流動性 D 且波動率在全市場前 {100 - VOL_HI}%，不列入預設 AI 精選"
    if variant in ("C", "D") and g == "D":
        return "capped", f"流動性 D（偏低），不列入前 {TOP} 名"
    if variant == "D" and g == "C" and vol is not None and vol >= VOL_TOP:
        return "capped", f"流動性 C 且波動率在全市場前 {100 - VOL_TOP}%，不列入前 {TOP} 名"
    return "normal", None


def order(items: list[tuple], variant: str = LIVE, top: int = TOP) -> tuple[list[dict], list[dict]]:
    """items：依原始排名排好的 (股票代號, 原始分數, 流動性等級, 波動率百分位)。
    回傳 (推薦順序, 被排除的)；每筆是 {id, raw, score, eligibility, reason}。原始分數不改，推薦分數只對受限股票封頂。"""
    normal, capped, excluded = [], [], []
    for sid, raw, g, vol in items:
        e, why = status(g, vol, variant)
        x = {"id": sid, "raw": raw, "score": raw, "eligibility": e, "reason": why}
        (normal if e == "normal" else capped if e == "capped" else excluded).append(x)
    head, rest = normal[:top], normal[top:]
    ceiling = head[-1]["raw"] if len(head) == top else None
    for x in capped:   # 推薦分數封頂在第 10 名的分數，所以不會排進最高推薦層級
        if ceiling is not None and x["raw"] is not None and x["raw"] > ceiling:
            x["score"] = ceiling
    merged = sorted(rest + capped, key=lambda x: -(x["score"] if x["score"] is not None else -1e9))  # sorted 是穩定排序：同分時一般股票在前
    if ceiling is None:
        merged = rest + capped
    return head + merged, excluded


SPIKE_VOL = 3.0      # 當日量比 ≥ 3：成交量突然暴增
RUNUP_20D = 30       # 近 20 日漲幅 ≥ 30%：短期漲幅過大
RUNUP_MA20 = 15      # 股價高於 20 日均線 15% 以上
IND_CROWD = 4        # 前 10 名同產業 ≥ 4 檔：組合集中


def alerts(r: dict) -> list[dict]:
    """高分股票異常警示（推薦風險狀態）。只提醒，不改分數也不改預測。"""
    out = []
    add = lambda k, t: out.append({"kind": k, "text": t})
    g = (r.get("liq") or {}).get("grade")
    if g in ("D", "E", "F", "?"):
        add("liquidity", f"流動性{LEVEL[g]}，不列入最高推薦")
    if (_num(r.get("vol_ratio")) or 0) >= SPIKE_VOL:
        add("volume_spike", "成交量突然暴增（量比 ≥ 3），注意短期交易異常")
    if (_num(r.get("ret_20d")) or 0) >= RUNUP_20D or (_num(r.get("dist_ma20")) or 0) >= RUNUP_MA20:
        add("run_up", "近期漲幅過大，注意追高風險")
    if r.get("fin_stale") or r.get("rev_stale"):
        add("stale", "財報或營收資料過期，分數可信度較低")
    if r.get("disposition"):
        add("disposition", "處置股票，交易受限")
    return out


def annotate(rows: list[dict], modes) -> None:
    """網站用：每檔股票加上流動性等級，以及各模式的推薦資格（ai.rec）。不改 ai.modes 的原始分數與名次。"""
    for r in rows:
        L = grade(r)
        r["liq"] = L
        r["liquidity_grade"], r["liquidity_risk"] = L["grade"], L["risk"]
        r.setdefault("ai", {})["rec"] = {}
        r["ai"]["alerts"] = alerts(r)
    for mode in modes:
        ranked = sorted((r for r in rows if r.get("ai", {}).get("modes", {}).get(mode, {}).get("rank")),
                        key=lambda r: r["ai"]["modes"][mode]["rank"])
        rec, exc = order([(r["stock_id"], r["ai"]["modes"][mode]["score"], r["liq"]["grade"], r.get("vol_1y_pct")) for r in ranked if not r.get("disposition")])
        exc += [{"id": r["stock_id"], "reason": "處置股票，交易受限，不列入預設 AI 精選"} for r in ranked if r.get("disposition")]
        by = {r["stock_id"]: r for r in ranked}
        n_ok = sum(x["eligibility"] == "normal" for x in rec)
        gap = max(0, TOP - n_ok)   # 正常股票不足 10 檔時，前 10 名就只有這些；受限股票從第 11 名開始，不拿來補滿
        for i, x in enumerate(rec, 1):
            by[x["id"]]["ai"]["rec"][mode] = {"recommendation_eligibility": x["eligibility"], "recommendation_score": x["score"],
                                              "rank": i if i <= n_ok else i + gap, "recommendation_cap_reason": x["reason"]}
        for x in exc:
            by[x["id"]]["ai"]["rec"][mode] = {"recommendation_eligibility": "excluded", "recommendation_score": None,
                                              "rank": None, "recommendation_cap_reason": x["reason"]}
        top = [by[x["id"]] for x in rec[:TOP]]
        cnt = {}
        for r in top:
            cnt[r.get("industry") or ""] = cnt.get(r.get("industry") or "", 0) + 1
        for r in top:
            k = r.get("industry") or ""
            if k and cnt[k] >= IND_CROWD:
                r["ai"]["rec"][mode]["crowded"] = f"前 {TOP} 名有 {cnt[k]} 檔同屬{k}，組合風險集中"
        for r in rows:
            r["ai"]["rec"].setdefault(mode, {"recommendation_eligibility": "not_ranked", "recommendation_score": None, "rank": None,
                                             "recommendation_cap_reason": "未通過此模式的硬性條件，不在原始排名內"})
