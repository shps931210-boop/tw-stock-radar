"""個股研究資料覆蓋率報告（data/coverage.json）。

每次重算快照時，統計所有普通股在各研究模組（行情、基本面、估值、短線、波段、長期、AI 評分）的資料狀態，
並把「資料不足」的原因整理成排行榜（依影響股票數排序），讓管理員知道該先補哪一種資料。
另外挑出八類代表股票（大型權值股、中小型上市、上櫃、新上市、金融、虧損、缺財報、歷史估值不足）當驗收樣本。
"""
from __future__ import annotations

import re
from collections import Counter

MODULES = {"quote": "行情", "fundamental": "基本面", "valuation": "估值", "short": "短線策略", "swing": "波段策略",
           "long": "長期判斷", "ai": "AI 評分"}
ORDER = ["ready", "partial", "insufficient_history", "stale", "not_applicable", "missing_source", "calculation_error"]


def _norm(why: str) -> str:
    """把原因裡的數字、日期、股票各自的值拿掉，相同原因才能合併計數。"""
    why = re.sub(r"（財報補齊中：.*?）", "（財報補齊中）", why)
    s = re.sub(r"\d{4}-\d{2}(-\d{2})?(-Q\d)?", "某期", why)
    s = re.sub(r"-?\d+(\.\d+)?", "N", s)
    return s.split("；")[0][:80]


def _pick(rows: list[dict], cond, key=None) -> dict | None:
    xs = [r for r in rows if cond(r)]
    if key:
        xs.sort(key=key)
    return xs[0] if xs else None


def build(rows: list[dict], decision: dict) -> dict:
    common = [r for r in rows if r.get("is_common") and r["stock_id"] in decision]
    n = len(common)
    mods = {}
    reasons = Counter()
    for k, name in MODULES.items():
        c = Counter((decision[r["stock_id"]].get("modules") or {}).get(k, {}).get("status", "missing_source") for r in common)
        mods[k] = {"name": name, "counts": {s: c.get(s, 0) for s in ORDER if c.get(s)},
                   "usable": round((c.get("ready", 0) + c.get("partial", 0)) / n * 100, 1) if n else None}
        for r in common:
            m = (decision[r["stock_id"]].get("modules") or {}).get(k) or {}
            if m.get("status") not in ("ready", None) and m.get("why"):
                reasons[(name, _norm(m["why"]))] += 1
    usable_any = sum(1 for r in common if all((decision[r["stock_id"]].get("modules") or {}).get(k, {}).get("status") != "missing_source"
                                              for k in ("quote",)))
    D = lambda r: decision[r["stock_id"]]
    st = lambda r, k: ((D(r).get("modules") or {}).get(k) or {}).get("status")
    by_cap = lambda r: -(r.get("market_cap") or 0)
    samples = {
        "大型權值股": _pick(common, lambda r: True, by_cap),
        "中小型上市股票": _pick(common, lambda r: r.get("market") == "twse" and (r.get("market_cap") or 0) < 1e10 and r.get("base_ok"), by_cap),
        "上櫃股票": _pick(common, lambda r: r.get("market") == "tpex" and r.get("base_ok"), by_cap),
        "新上市股票": _pick(common, lambda r: (r.get("history_days") or 0) < 120, lambda r: -(r.get("history_days") or 0)),
        "金融股": _pick(common, lambda r: r.get("is_financial"), by_cap),
        "EPS 為負": _pick(common, lambda r: (r.get("eps_ttm") or 0) < 0, by_cap),
        "財報缺漏": _pick(common, lambda r: not r.get("fin_period"), by_cap),
        "價格完整但歷史估值不足": _pick(common, lambda r: st(r, "valuation") == "insufficient_history", by_cap),
    }
    return {
        "schema_version": 1, "stocks": n, "usable_quote": usable_any,
        "modules": mods,
        "reasons": [{"module": m, "why": w, "stocks": c} for (m, w), c in reasons.most_common(25)],
        "samples": {k: None if r is None else {"stock_id": r["stock_id"], "name": r.get("name"),
                                               "status": {m: st(r, m) for m in MODULES}} for k, r in samples.items()},
    }
