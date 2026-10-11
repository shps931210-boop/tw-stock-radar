"""每日 AI 選股紀錄、入選後股價追蹤、AI 選股成績單。

紀錄（history/picks/）
- 日期.json：每個資料日第一次完整更新後的選股結果（首次發布版）。
- 日期.v2.json、v3…：同一天稍後的更新讓名單改變時另存「補正版」（例如 21:00 補上法人資料後）。
  所有檔案都不覆寫，GitHub Actions 會 commit 進 repository。成績單一律用首次發布版，避免事後挑選。

成績單的計算方式（固定規則，所有紀錄一致）
- 進場：選股日的「下一個交易日開盤價」（名單在收盤後才產生，當天收盤價買不到）。
- 出場：進場後第 5、20、60 個交易日的收盤價；還沒到期的期間不計入。
- 扣交易成本 0.585%（手續費 0.1425% × 2 ＋ 證交稅 0.3%），用還原除權息股價。
- 基準：同期加權指數（選股日收盤 → 出場日收盤；加權指數是價格指數、沒有開盤價，會有一晚的時間差）。
- 所有股票用同一個「觀察日」的價格；觀察日沒有成交（停牌等）的股票不計入，並列出檔數。
- 等權重組合的最大回撤：持有期間每天的組合淨值相對前高的最大跌幅。
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config, db, metrics

HISTORY = config.ROOT / "history" / "picks"
TOP_N = 20
COST = 0.001425 * 2 + 0.003
HORIZONS = (5, 20, 60)
FACTORS = ("quality", "growth", "value", "momentum", "trend")


def _snapshot_picks(snap: dict) -> tuple[dict, list]:
    st = [s for s in snap["stocks"] if s.get("is_common")]
    picks = {}
    for mode in snap.get("ai_modes", {}):
        ranked = sorted((s for s in st if s["ai"]["modes"][mode]["rank"]), key=lambda s: s["ai"]["modes"][mode]["rank"])[:TOP_N]
        picks[mode] = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"][mode]["rank"],
                        "score": s["ai"]["modes"][mode]["score"], "close": s.get("close"), "industry": s.get("industry"),
                        "factors": {f: s.get(f"score_{f}") for f in FACTORS}} for s in ranked]
    sigs = [{"stock_id": s["stock_id"], "name": s["name"], "close": s.get("close"), "triggers": s["timing"]["triggers"],
             "score": s["ai"]["modes"]["balanced"]["score"]} for s in st if s.get("timing", {}).get("status") == "buy"]
    return picks, sigs


def _versions(day: str) -> list:
    return sorted(HISTORY.glob(f"{day}*.json"), key=lambda p: int((re.search(r"\.v(\d+)\.json$", p.name) or [0, 1])[1]))


def record(snap: dict) -> str | None:
    """寫入當天的選股紀錄。第一次寫首次發布版；之後名單有變才另存補正版。不能用的資料回傳 None。"""
    if snap.get("source") != "live" or snap.get("rankings_blocked"):
        return None
    day = (snap.get("data_dates") or {}).get("prices")
    if not day:
        return None
    from .ai_model import MODEL_VERSION
    picks, sigs = _snapshot_picks(snap)
    old = _versions(day)
    if old:
        last = json.loads(old[-1].read_text("utf-8"))
        same = all([p["stock_id"] for p in last["picks"].get(m, [])] == [p["stock_id"] for p in picks[m]] for m in picks)
        if same and [x["stock_id"] for x in last["signals"]] == [x["stock_id"] for x in sigs]:
            return None  # 名單沒變，不用另存
        path = HISTORY / f"{day}.v{len(old) + 1}.json"
    else:
        path = HISTORY / f"{day}.json"
    out = {"date": day, "version": len(old) + 1, "kind": "首次發布" if not old else "補正版",
           "model_version": MODEL_VERSION, "generated_at": snap.get("generated_at"),
           "taiex": (snap.get("market") or {}).get("taiex"), "universe": snap.get("universe"), "picks": picks, "signals": sigs}
    HISTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return str(path)


def load(first_only: bool = True) -> list[dict]:
    if not HISTORY.exists():
        return []
    files = sorted(HISTORY.glob("*.json"))
    if first_only:
        files = [p for p in files if not re.search(r"\.v\d+\.json$", p.name)]
    return [json.loads(p.read_text("utf-8")) for p in files]


def _prices(ids: list[str], first: str, dbname: str):
    with db.connect(dbname) as con:
        q = ",".join("?" * len(ids))
        px = db.read(con, f"SELECT * FROM prices WHERE stock_id IN ({q}) AND date >= ? ORDER BY date", tuple(ids) + (first,)) if ids else pd.DataFrame()
        ex = db.read(con, f"SELECT * FROM exrights WHERE stock_id IN ({q})", tuple(ids)) if ids else pd.DataFrame()
        idx = db.read(con, "SELECT * FROM index_prices WHERE date >= ? ORDER BY date", (first,))
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    series = {}
    for sid, p in (px.groupby("stock_id") if len(px) else []):
        adj, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        series[sid] = adj.set_index("date")[["open", "close"]]
    return series, (idx.set_index("date")["close"] if len(idx) else pd.Series(dtype=float))


def _cohort(group: list[dict], day: str, series: dict, idx: pd.Series, h: int | None, obs: str):
    """一組股票從 day 隔日開盤進場、持有 h 天（None = 持有到觀察日）的結果。"""
    dates = idx.index[idx.index > day]
    if not len(dates):
        return None
    entry = dates[0]
    if h is None:
        exit_ = obs
    elif len(dates) >= h:
        exit_ = dates[h - 1]
    else:
        return None  # 還沒到期
    if exit_ < entry or day not in idx.index:
        return None
    path_dates = dates[(dates >= entry) & (dates <= exit_)]
    rets, paths, missing = [], [], 0
    for p in group:
        s = series.get(p["stock_id"])
        if s is None or entry not in s.index or exit_ not in s.index:
            missing += 1  # 進場日或觀察日沒有成交
            continue
        o = s.at[entry, "open"]
        if pd.isna(o) or o <= 0:
            o = s.at[entry, "close"]
        rets.append(s.at[exit_, "close"] / o - 1 - COST)
        paths.append(s["close"].reindex(path_dates).ffill() / o)
    if not rets:
        return {"n": len(group), "priced": 0, "missing": missing, "exit": exit_}
    bench = float(idx[exit_] / idx[day] - 1)
    nav = pd.concat(paths, axis=1).mean(axis=1)
    mdd = float((nav / nav.cummax() - 1).min()) if len(nav) else 0.0
    r = np.array(rets)
    return {"n": len(group), "priced": len(r), "missing": missing, "entry": entry, "exit": exit_, "held": len(path_dates),
            "avg": _p(r.mean()), "median": _p(np.median(r)), "bench": _p(bench), "excess": _p(r.mean() - bench),
            "win": _p((r > 0).mean()), "beat": _p((r > bench).mean()), "mdd": _p(mdd)}


def performance(dbname: str = "stocks.db") -> dict:
    logs = load()
    if not logs:
        return {"days": 0, "logs": [], "scorecard": {}}
    groups = lambda L: list(L["picks"].items()) + [("signals", L["signals"])]
    ids = sorted({p["stock_id"] for L in logs for _, g in groups(L) for p in g})
    series, idx = _prices(ids, logs[0]["date"], dbname)
    obs = idx.index[-1] if len(idx) else None
    rows = []
    for L in logs:
        d = L["date"]
        row = {"date": d, "model_version": L.get("model_version"), "track": {}, "fixed": {}}
        for name, g in groups(L):
            row["track"][name] = _cohort(g, d, series, idx, None, obs) if obs else None
            row["fixed"][name] = {h: _cohort(g, d, series, idx, h, obs) for h in HORIZONS}
        rows.append(row)
    # 成績單：所有已到期的選股日，各期間的平均
    card = {}
    for name in [k for k, _ in groups(logs[-1])]:
        card[name] = {}
        for h in HORIZONS:
            cs = [r["fixed"][name][h] for r in rows if r["fixed"].get(name, {}).get(h) and r["fixed"][name][h].get("priced")]
            if cs:
                card[name][h] = {"cohorts": len(cs), "avg": _p(np.mean([c["avg"] for c in cs]) / 100),
                                 "excess": _p(np.mean([c["excess"] for c in cs]) / 100),
                                 "beat_cohorts": _p(np.mean([c["excess"] > 0 for c in cs])),
                                 "win": _p(np.mean([c["win"] for c in cs]) / 100), "worst_mdd": min(c["mdd"] for c in cs)}
    return {"days": len(logs), "first": logs[0]["date"], "observed": obs, "horizons": list(HORIZONS),
            "cost_pct": round(COST * 100, 3), "logs": rows[::-1], "scorecard": card}


def changes(snap: dict) -> dict | None:
    """和上一個選股日相比：均衡型前 20 名的新進榜、掉出榜、排名變化，以及分數變化最大的因子。"""
    day = (snap.get("data_dates") or {}).get("prices")
    prev = [L for L in load(first_only=False) if L["date"] < (day or "")]
    if not prev:
        return None
    last = max(prev, key=lambda L: (L["date"], L.get("version", 1)))
    before = {p["stock_id"]: p for p in last["picks"].get("balanced", [])}
    by = {s["stock_id"]: s for s in snap["stocks"]}
    now = sorted((s for s in snap["stocks"] if s.get("is_common") and s["ai"]["modes"]["balanced"]["rank"]),
                 key=lambda s: s["ai"]["modes"]["balanced"]["rank"])[:TOP_N]
    now_ids = [s["stock_id"] for s in now]

    def driver(sid):
        old = (before.get(sid) or {}).get("factors") or {}
        s = by.get(sid) or {}
        diffs = {f: (s.get(f"score_{f}") or 0) - old[f] for f in FACTORS if old.get(f) is not None and s.get(f"score_{f}") is not None}
        if not diffs:
            return None
        f = max(diffs, key=lambda k: abs(diffs[k]))
        return {"factor": f, "change": round(diffs[f], 1)} if abs(diffs[f]) >= 3 else None

    new = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"]["balanced"]["rank"], "driver": driver(s["stock_id"])}
           for s in now if s["stock_id"] not in before]
    dropped = []
    for sid, p in before.items():
        if sid in now_ids:
            continue
        s = by.get(sid) or {}
        r = s.get("ai", {}).get("modes", {}).get("balanced", {})
        why = ("資料不足，暫不排名" if s and not s.get("complete") else "未通過基本門檻" if s and not s.get("base_ok")
               else f"排名降到第 {r['rank']} 名" if r.get("rank") else "不在排行榜")
        dropped.append({"stock_id": sid, "name": p["name"], "was": p["rank"], "why": why, "driver": driver(sid)})
    moved = []
    for s in now:
        p = before.get(s["stock_id"])
        if p and p["rank"] != s["ai"]["modes"]["balanced"]["rank"]:
            moved.append({"stock_id": s["stock_id"], "name": s["name"], "was": p["rank"], "rank": s["ai"]["modes"]["balanced"]["rank"],
                          "driver": driver(s["stock_id"])})
    moved.sort(key=lambda x: abs(x["was"] - x["rank"]), reverse=True)
    return {"since": last["date"], "new": new, "dropped": dropped, "moved": moved[:8]}


def _p(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x) * 100, 2)
