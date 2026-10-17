"""每週固定名單績效 → data/weekly.json（網站「績效與回測」頁的「每週成績單」）。

規則（每週都一樣，不挑對自己有利的口徑）：
- 名單：每週第一個交易日首次發布、封存的 AI 量化選股名單（不用之後的補正版）。
- 進場：名單發布後的下一個交易日開盤價（名單在收盤後才產生，當天收盤價買不到）。
- 出場：同一週最後一個交易日收盤價；這週還沒結束就算到最新收盤。
- 組合：前 5／10／20 名等權重、中途不換股；扣一次來回交易成本（手續費＋證交稅）。
  進場日買不到的股票（沒成交、漲停鎖死、成交太少）那一份留現金，並列出原因。
- 對照：0050 同一段期間（開盤進場、收盤出場，還原除權息）；加權指數只有收盤價，用名單日收盤到出場日收盤。
- 另外附上「名單日收盤價」口徑，方便和外部報告對帳，但那個價格實際上買不到。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from . import config, db, track

TOPS = (5, 10, 20)
WEEKS = 8
OUT = "weekly.json"


def _week(d: str) -> tuple[int, int]:
    return date.fromisoformat(d).isocalendar()[:2]


def build(dbname: str = "stocks.db") -> dict:
    logs = track.load()
    if not logs:
        return {"weeks": []}
    first = {}
    for L in logs:  # 每週第一份首次發布名單
        first.setdefault(_week(L["date"]), L)
    lists = [first[k] for k in sorted(first)][-WEEKS:]
    ids = sorted({p["stock_id"] for L in lists for g in L["picks"].values() for p in g})
    series, idx, bench = track._prices(ids, lists[0]["date"], dbname)
    b = bench[0]
    cal = list(idx.index)
    weeks = []
    for L in lists:
        d = L["date"]
        after = [x for x in cal if x > d]
        if not after:
            weeks.append({"list_date": d, "generated_at": L.get("generated_at"), "status": "waiting"})
            continue
        entry = after[0]
        wk = _week(d)
        days = [x for x in cal if entry <= x and _week(x) == wk]
        if not days:  # 名單在週五發布：進場已經是下一週，這份名單不列入（每週只看當週）
            continue
        exit_ = days[-1]
        today = config.today()
        done = exit_ != cal[-1] or _week(today.isoformat()) != wk or (today.weekday() >= 5 and today.isoformat() > exit_)
        bench_ret = float(b.at[exit_, "close"] / b.at[entry, "open"] - 1) if entry in b.index and exit_ in b.index else None
        taiex = float(idx[exit_] / idx[d] - 1) if d in idx.index and exit_ in idx.index and idx[d] and idx[exit_] else None
        modes = {}
        for m, picks in L["picks"].items():
            rows = []
            for p in picks[:max(TOPS)]:
                s = series.get(p["stock_id"])
                why = track.untradable(s, entry)
                r = r_close = None
                path = []
                if s is not None and exit_ in s.index and d in s.index:
                    r_close = float(s.at[exit_, "close"] / s.at[d, "close"] - 1)
                if not why and s is not None and exit_ in s.index:
                    o = float(s.at[entry, "open"])
                    r = float(s.at[exit_, "close"] / o - 1)
                    path = [round(float(s.at[x, "close"] / o - 1) * 100, 2) if x in s.index else None for x in days]
                rows.append({"stock_id": p["stock_id"], "name": p.get("name"), "rank": p.get("rank"), "industry": p.get("industry"),
                             "entry_open": None if why or s is None else round(float(s.at[entry, "open"]), 2),
                             "exit_close": round(float(s.at[exit_, "close"]), 2) if s is not None and exit_ in s.index else None,
                             "ret": None if r is None else round(r * 100, 2), "ret_close_basis": None if r_close is None else round(r_close * 100, 2),
                             "untradable": why, "path": path})
            tops = {}
            for n in TOPS:
                sub = rows[:n]
                if not sub:
                    continue
                w = 1 / n
                got = [x["ret"] / 100 for x in sub if x["ret"] is not None]
                net = sum(x * w for x in got) - track.COST * w * len(got)
                cb = [x["ret_close_basis"] / 100 for x in sub if x["ret_close_basis"] is not None]
                path = []
                for i in range(len(days)):
                    v = [x["path"][i] / 100 for x in sub if x["path"] and x["path"][i] is not None]
                    path.append(round((sum(v) * w - track.COST * w * len(v)) * 100, 2) if v else None)
                tops[str(n)] = {"net": round(net * 100, 2), "excess": None if bench_ret is None else round((net - bench_ret) * 100, 2),
                                "vs_taiex": None if taiex is None else round((net - taiex) * 100, 2),
                                "up": sum(1 for x in sub if (x["ret"] or 0) > 0), "tradable": len(got), "n": len(sub),
                                "close_basis": round(float(np.mean(cb)) * 100, 2) if cb else None, "path": path}
            modes[m] = {"top": tops, "stocks": rows}
        weeks.append({"list_date": d, "generated_at": L.get("generated_at"), "model_version": L.get("model_version"),
                      "on_time": track.on_time(L, entry), "entry": entry, "exit": exit_, "days": days, "final": done,
                      "bench": None if bench_ret is None else round(bench_ret * 100, 2), "taiex": None if taiex is None else round(taiex * 100, 2),
                      "modes": modes})
    # 累計：已經結束的週，每週成本後超額報酬
    tally = {}
    for w in weeks:
        if not w.get("final") or "modes" not in w:
            continue
        for m, M in w["modes"].items():
            for n, t in M["top"].items():
                if t["excess"] is not None:
                    tally.setdefault(m, {}).setdefault(n, []).append(t["excess"])
    summary = {m: {n: {"weeks": len(v), "beat": sum(x > 0 for x in v), "avg_excess": round(float(np.mean(v)), 2)} for n, v in T.items()}
               for m, T in tally.items()}
    return {"generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "cost_pct": round(track.COST * 100, 3),
            "bench": track.BENCH_ID, "tops": list(TOPS), "weeks": weeks[::-1], "summary": summary}


def write(dbname: str = "stocks.db") -> dict:
    res = build(dbname)
    (config.DATA_DIR / OUT).write_text(json.dumps(res, ensure_ascii=False), "utf-8")
    return res


if __name__ == "__main__":
    r = write()
    for w in r["weeks"][:3]:
        print(w["list_date"], w.get("entry"), w.get("exit"), "0050", w.get("bench"), "加權", w.get("taiex"))
        for m, M in (w.get("modes") or {}).items():
            print("  ", m, {n: (t["net"], t["excess"], t["close_basis"]) for n, t in M["top"].items()})
