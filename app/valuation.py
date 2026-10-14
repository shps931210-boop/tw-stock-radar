"""估值區間：用自己過去的本益比分布，加上同產業比較，估出保守、基準、樂觀三個參考價。

- 歷史本益比 = 每天的實際收盤價 ÷ 當天「已經公布」的近四季 EPS（季報在法定期限隔天才算公布，沒有用到未來資料）。
  這是用財報 EPS 自己算的，和證交所公布的本益比可能有些微差異。
- 三情境：近四季 EPS × 自己歷史本益比的第 20、50、80 百分位。
- 反推市場預期：目前股價要回到「歷史中位數本益比」，EPS 需要成長多少。
- EPS ≤ 0、金融業（獲利受利差與評價影響大，本益比參考性較低時另外標示）、歷史少於 120 個交易日：不估或標示限制。
這是研究參考，不是目標價。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS = 120


def fin_available(period: str, is_financial: bool = False) -> str:
    """季報最早可以使用的日期（法定期限隔天）；和 pit_backtest.fin_available 相同，放這裡避免循環匯入。"""
    y, q = map(int, period.split("-Q"))
    return {1: f"{y}-05-16", 2: f"{y}-09-01" if is_financial else f"{y}-08-15", 3: f"{y}-11-15"}.get(q, f"{y + 1}-04-01")


def eps_ttm_series(fin: pd.DataFrame, is_financial: bool) -> list[tuple[str, float]]:
    """(可以使用的日期, 近四季 EPS)，依日期排序；只有連續四季都有資料才算。"""
    if not len(fin):
        return []
    f = fin.dropna(subset=["eps"]).sort_values("period")
    per = list(f["period"]); eps = list(f["eps"].astype(float))
    idx = {p: i for i, p in enumerate(per)}
    out = []
    for i, p in enumerate(per):
        y, q = map(int, p.split("-Q"))
        prev = [(y, q - k) if q - k > 0 else (y - 1, q - k + 4) for k in range(4)]
        keys = [f"{a}-Q{b}" for a, b in prev]
        if all(k in idx for k in keys):
            out.append((fin_available(p, is_financial), sum(eps[idx[k]] for k in keys)))
    return sorted(out)


def band(raw: pd.DataFrame, fin: pd.DataFrame, is_financial: bool) -> dict:
    """raw：實際成交價（未還原）。回傳目前本益比、歷史分位、三情境價與反推成長。"""
    s = eps_ttm_series(fin, is_financial)
    if not s or not len(raw):
        return {"why": "沒有連續四季 EPS，無法用本益比估值"}
    d = raw["date"].to_numpy(str)
    c = raw["close"].to_numpy(float)
    when = [x[0] for x in s]
    k = np.searchsorted(when, d, side="right") - 1
    eps = np.array([s[j][1] if j >= 0 else np.nan for j in k])
    now_eps = eps[-1]
    if np.isnan(now_eps):
        return {"why": "近四季 EPS 還沒公布齊"}
    if now_eps <= 0:
        return {"why": f"近四季 EPS {now_eps:.2f} 元（虧損），本益比沒有意義，改看股價淨值比與營收", "eps": float(now_eps)}
    ok = (eps > 0) & ~np.isnan(c)
    pe = c[ok] / eps[ok]
    if len(pe) < MIN_DAYS:
        return {"why": f"可用的歷史本益比只有 {len(pe)} 天，至少要 {MIN_DAYS} 天", "eps": float(now_eps), "pe": float(c[-1] / now_eps)}
    pe = pe[-500:]  # 約近兩年
    p20, p50, p80 = (float(np.percentile(pe, q)) for q in (20, 50, 80))
    cur = float(c[-1] / now_eps)
    return {"eps": round(float(now_eps), 2), "pe": round(cur, 2), "p20": round(p20, 2), "p50": round(p50, 2), "p80": round(p80, 2),
            "pct": round(float((pe < cur).mean() * 100), 0), "days": int(len(pe)), "from": str(d[ok][-len(pe)]),
            "low": round(now_eps * p20, 2), "base": round(now_eps * p50, 2), "high": round(now_eps * p80, 2),
            "implied_growth": round((cur / p50 - 1) * 100, 1),
            "note": "金融業獲利受利差與投資評價影響大，本益比區間只供參考" if is_financial else None}


def add_peers(rows: list[dict]) -> None:
    """同產業本益比中位數（用證交所公布的本益比；至少 5 檔有值）。"""
    by: dict[str, list[float]] = {}
    for r in rows:
        if r.get("is_common") and r.get("industry") and r.get("per") and r["per"] > 0:
            by.setdefault(r["industry"], []).append(r["per"])
    med = {k: float(np.median(v)) for k, v in by.items() if len(v) >= 5}
    for r in rows:
        v = r.get("valuation_band")
        if isinstance(v, dict) and r.get("industry") in med:
            v["peer"] = round(med[r["industry"]], 2)
            v["peer_n"] = len(by[r["industry"]])
