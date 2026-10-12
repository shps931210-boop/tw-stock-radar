"""產業熱力圖（data/industry/latest.json）：用最新收盤資料把普通股依產業彙總。

計算方式（全部用同一個資料日、還原除權息後的收盤價）
- 只納入「最新資料日有收盤價」的普通股；停牌、缺價、資料日落後的股票不計入報酬，但計入覆蓋率（n / n_total）。
- 等權報酬：產業內每檔股票報酬的平均。市值加權：只有產業內 8 成以上股票有市值資料時才計算，否則留空，不估算。
- 相對強度：產業等權報酬 − 同期間加權指數報酬（指數收盤 → 收盤）。
- 成交金額比：今天產業成交金額合計 ÷ 近 20 日平均成交金額合計。成交金額是「買賣都算」的交易量，不是資金淨流入。
- 法人買賣超（估算金額）：近 5 個交易日每天的買賣超股數 × 當天收盤價，加總；只算籌碼資料是最新的股票，覆蓋率另外列出。
- 產業分類用資料庫裡證交所／櫃買中心的產業別；沒有分類的歸「未分類」，不猜測。
"""
from __future__ import annotations

import numpy as np

from . import config, db

PERIODS = {"1d": "chg_pct", "5d": "ret_5d", "20d": "ret_20d"}
CAP_MIN_COVERAGE = 0.8
TOP_N = 3


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def _index_returns(dbname: str, day: str) -> dict:
    with db.connect(dbname) as con:
        rows = con.execute("SELECT date, close FROM index_prices WHERE date <= ? ORDER BY date DESC LIMIT 21", (day,)).fetchall()
    if not rows or rows[0][0] != day:
        return {k: None for k in PERIODS}
    c = [r[1] for r in rows]
    back = {"1d": 1, "5d": 5, "20d": 20}
    return {k: _r((c[0] / c[b] - 1) * 100) if len(c) > b and c[b] else None for k, b in back.items()}


def build(snap: dict, dbname: str = "stocks.db") -> dict:
    day = (snap.get("data_dates") or {}).get("prices")
    common = [s for s in snap.get("stocks", []) if s.get("is_common")]
    warn = []
    taiex = _index_returns(dbname, day) if day else {k: None for k in PERIODS}
    if any(v is None for v in taiex.values()):
        warn.append("加權指數缺少部分日期，相對強度可能無法計算")
    groups: dict[str, list[dict]] = {}
    for s in common:
        groups.setdefault(s.get("industry") or "未分類", []).append(s)
    live = [s for s in common if s.get("date") == day]
    market_eq = {k: _r(np.mean([s[f] for s in live if s.get(f) is not None])) if live else None for k, f in PERIODS.items()}
    out = []
    for ind, members in groups.items():
        priced = [s for s in members if s.get("date") == day]
        row = {"industry": ind, "n_total": len(members), "n": len(priced),
               "coverage": _r(len(priced) / len(members) * 100, 1) if members else None,
               "markets": {m: sum(1 for s in members if s.get("market") == m) for m in ("twse", "tpex")},
               "ret": {}, "rel": {}}
        for k, f in PERIODS.items():
            v = [(s[f], s.get("market_cap")) for s in priced if s.get(f) is not None]
            if not v:
                row["ret"][k] = None
                row["rel"][k] = None
                continue
            r = np.array([x for x, _ in v], float)
            caps = [(x, c) for x, c in v if c and c > 0]
            cap = _r(sum(x * c for x, c in caps) / sum(c for _, c in caps)) if len(caps) >= CAP_MIN_COVERAGE * len(v) else None
            row["ret"][k] = {"eq": _r(r.mean()), "cap": cap, "median": _r(np.median(r)), "n": len(r),
                             "up": int((r > 0).sum()), "down": int((r < 0).sum()),
                             "p25": _r(np.percentile(r, 25)), "p75": _r(np.percentile(r, 75))}
            row["rel"][k] = _r(r.mean() - taiex[k]) if taiex[k] is not None else None
        vals = [(s.get("value_1d"), s.get("avg_value_20")) for s in priced if s.get("value_1d") is not None and s.get("avg_value_20")]
        today, avg = sum(a for a, _ in vals), sum(b for _, b in vals)
        row["value"] = {"today": _r(today, 0), "avg20": _r(avg, 0), "ratio": _r(today / avg, 2) if avg else None, "n": len(vals)}
        fl = [s for s in priced if s.get("foreign_5d_value") is not None]
        row["flows"] = {"foreign_5d": _r(sum(s["foreign_5d_value"] for s in fl), 0) if fl else None,
                        "trust_5d": _r(sum(s.get("trust_5d_value") or 0 for s in fl), 0) if fl else None,
                        "n": len(fl), "coverage": _r(len(fl) / len(priced) * 100, 1) if priced else None}
        ranked = sorted((s for s in members if (s.get("ai") or {}).get("modes", {}).get("balanced", {}).get("rank")),
                        key=lambda s: s["ai"]["modes"]["balanced"]["rank"])
        row["top"] = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"]["balanced"]["rank"],
                       "score": _r(s["ai"]["modes"]["balanced"]["score"], 1)} for s in ranked[:TOP_N]]
        row["members"] = sorted(s["stock_id"] for s in members)
        out.append(row)
    out.sort(key=lambda x: -x["n_total"])
    if "未分類" in groups:
        warn.append(f"有 {len(groups['未分類'])} 檔股票沒有產業分類，歸在「未分類」")
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "data_as_of": day,
            "source": "證交所、櫃買中心盤後行情與三大法人（還原除權息）", "periods": list(PERIODS),
            "market": {"taiex": taiex, "equal": market_eq},
            "coverage": {"stocks": len(common), "priced": len(live)}, "warnings": warn, "industries": out}
