"""每日 AI 選股紀錄與績效追蹤。

- record()：每個資料日第一次完整更新後，把當天各模式的前 20 名與技術訊號股票寫進 history/picks/日期.json。
  檔案已存在就不覆寫，GitHub Actions 會把它 commit 進 repository，所以不會因為快取被清掉而遺失，也不能事後修改。
- performance()：用資料庫裡最新的（還原除權息）股價，計算每一天的入選股票到現在的報酬，並和加權指數比較。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, db, metrics

HISTORY = config.ROOT / "history" / "picks"
TOP_N = 20


def record(snap: dict) -> str | None:
    """寫入當天的選股紀錄；已經有紀錄、或資料不能用時回傳 None。"""
    if snap.get("source") != "live" or snap.get("rankings_blocked"):
        return None
    day = (snap.get("data_dates") or {}).get("prices")
    if not day:
        return None
    path = HISTORY / f"{day}.json"
    if path.exists():
        return None  # 不覆寫：紀錄的是當時網站實際顯示的結果
    from .ai_model import MODEL_VERSION
    st = [s for s in snap["stocks"] if s.get("is_common")]
    picks = {}
    for mode in snap.get("ai_modes", {}):
        ranked = sorted((s for s in st if s["ai"]["modes"][mode]["rank"]), key=lambda s: s["ai"]["modes"][mode]["rank"])[:TOP_N]
        picks[mode] = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"][mode]["rank"],
                        "score": s["ai"]["modes"][mode]["score"], "close": s.get("close")} for s in ranked]
    sigs = [{"stock_id": s["stock_id"], "name": s["name"], "close": s.get("close"), "triggers": s["timing"]["triggers"],
             "score": s["ai"]["modes"]["balanced"]["score"]} for s in st if s.get("timing", {}).get("status") == "buy"]
    out = {"date": day, "model_version": MODEL_VERSION, "generated_at": snap.get("generated_at"),
           "taiex": (snap.get("market") or {}).get("taiex"), "universe": snap.get("universe"), "picks": picks, "signals": sigs}
    HISTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return str(path)


def load() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(p.read_text("utf-8")) for p in sorted(HISTORY.glob("*.json"))]


def performance(dbname: str = "stocks.db") -> dict:
    logs = load()
    if not logs:
        return {"days": 0, "logs": []}
    ids = sorted({p["stock_id"] for L in logs for g in list(L["picks"].values()) + [L["signals"]] for p in g})
    first = logs[0]["date"]
    with db.connect(dbname) as con:
        q = ",".join("?" * len(ids))
        px = db.read(con, f"SELECT * FROM prices WHERE stock_id IN ({q}) AND date >= ? ORDER BY date", tuple(ids) + (first,)) if ids else pd.DataFrame()
        ex = db.read(con, f"SELECT * FROM exrights WHERE stock_id IN ({q})", tuple(ids)) if ids else pd.DataFrame()
        idx = db.read(con, "SELECT * FROM index_prices WHERE date >= ? ORDER BY date", (first,))
    idx_s = idx.set_index("date")["close"] if len(idx) else pd.Series(dtype=float)
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    series = {}
    for sid, p in px.groupby("stock_id") if len(px) else []:
        adj, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        series[sid] = adj.set_index("date")["close"]
    latest = idx_s.index[-1] if len(idx_s) else None

    def ret(sid, d):
        s = series.get(sid)
        if s is None or d not in s.index or not len(s):
            return None
        return float(s.iloc[-1] / s[d] - 1) * 100

    out = []
    for L in logs:
        d = L["date"]
        ir = float(idx_s.iloc[-1] / idx_s[d] - 1) * 100 if d in idx_s.index else None
        held = int((idx_s.index > d).sum()) if len(idx_s) else 0
        row = {"date": d, "model_version": L.get("model_version"), "held_days": held, "index_ret": _r(ir), "groups": {}}
        for name, group in list(L["picks"].items()) + [("signals", L["signals"])]:
            rs = [r for r in (ret(p["stock_id"], d) for p in group) if r is not None]
            if not rs:
                row["groups"][name] = {"n": len(group)}
                continue
            row["groups"][name] = {"n": len(group), "priced": len(rs), "avg": _r(np.mean(rs)),
                                   "excess": _r(np.mean(rs) - ir) if ir is not None else None,
                                   "beat": _r(np.mean([r > ir for r in rs]) * 100) if ir is not None else None}
        out.append(row)
    return {"days": len(logs), "first": first, "latest": latest, "logs": out[::-1]}


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)
