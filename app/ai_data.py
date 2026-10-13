"""給 AI 研究助理用的精簡資料（data/ai/universe.json）。

AI 後端（worker/）在 Cloudflare 上執行，CPU 時間很短，不能每次讀整份 snapshot.json（數 MB）。
這裡把全市場股票整理成「欄位 → 一整列數值」的格式（比一檔一個物件小很多、解析也快），
另外附上大盤、產業、排名變化與指標說明，讓 AI 可以查單股、比較多檔、依條件篩選、看產業與大盤。
"""
from __future__ import annotations

FIELDS = [
    "stock_id", "name", "industry", "market", "date", "close", "chg_pct", "ret_5d", "ret_20d", "ret_3m", "ret_12m",
    "market_cap", "avg_value_20", "per", "pbr", "dy", "per_pctile_1y", "eps_ttm", "eps_ttm_yoy", "roe", "roa",
    "gross_margin", "op_margin", "debt_ratio", "fscore", "rev_ym", "rev_yoy", "rev_yoy_3m", "rev_ytd_yoy", "fin_period", "val_date",
    "vol_1y", "mdd_1y", "beta", "rsi", "dist_ma60", "inst_date", "foreign_5d", "trust_5d", "foreign_streak", "trust_streak",
    "div_years", "payout", "score_quality", "score_growth", "score_value", "score_momentum", "score_trend",
    "industry_rank", "industry_n", "data_completeness",
]
EXTRA = ["score", "rank", "tag", "signal", "is_financial"]


def _r(v):
    return round(v, 2) if isinstance(v, float) else v


def _board(b):
    if not b or not b.get("markets"):
        return None
    slim = lambda xs: [{k: x.get(k) for k in ("stock_id", "name", "market", "industry", "chg_pct", "value")} for x in xs]
    return {"date": b.get("date"), "prev_date": b.get("prev_date"), "markets": b["markets"], "method": b.get("method"),
            "limit_up": slim(b.get("limit_up") or []), "limit_down": slim(b.get("limit_down") or []),
            "touched_up": b.get("touched_up"), "touched_down": b.get("touched_down"), "warnings": b.get("warnings")}


def build(snap: dict) -> dict:
    rows = [s for s in snap.get("stocks", []) if s.get("is_common")]
    cols: dict[str, list] = {k: [] for k in FIELDS + EXTRA}
    for s in rows:
        for k in FIELDS:
            cols[k].append(_r(s.get(k)))
        m = (((s.get("ai") or {}).get("modes") or {}).get("balanced") or {})
        cols["score"].append(_r(m.get("score")))
        cols["rank"].append(m.get("rank"))
        cols["tag"].append((s.get("ai") or {}).get("tag"))
        cols["signal"].append(1 if (s.get("timing") or {}).get("status") == "buy" else 0)
        cols["is_financial"].append(1 if s.get("is_financial") else 0)
    cat = [{"key": c["key"], "name": c["name"], "unit": c.get("unit"), "formula": c.get("formula"), "limits": c.get("limits")}
           for c in snap.get("catalog", []) if c["key"] in FIELDS or c["key"].startswith("score_")]
    ch = snap.get("changes") or None
    return {
        "schema_version": 1, "generated_at": snap.get("generated_at"), "data_dates": snap.get("data_dates"),
        "model": (snap.get("model") or {}).get("version"), "source": snap.get("source"),
        "note": "AI 量化評分是固定規則計算的全市場百分位（0～100，50 是中位數），score/rank 是均衡型；不是上漲機率。",
        "market": snap.get("market"), "breadth": snap.get("breadth"),
        # 市場看板的家數與漲跌停名單（名單只留代號、名稱、漲跌幅，控制大小）
        "market_board": _board(snap.get("market_board")),
        "industries": [{k: _r(v) for k, v in x.items()} for x in snap.get("industries", [])],
        "changes": {"since": ch.get("since"), "new": [{"stock_id": x["stock_id"], "rank": x.get("rank")} for x in ch.get("new", [])],
                    "dropped": [{"stock_id": x["stock_id"], "why": x.get("why")} for x in ch.get("dropped", [])],
                    "moved": [{"stock_id": x["stock_id"], "was": x.get("was"), "rank": x.get("rank")} for x in ch.get("moved", [])]} if ch else None,
        "catalog": cat, "fields": FIELDS + EXTRA, "n": len(rows), "cols": cols,
    }
