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


def _eval(picks, series, entry, exit_, d, days, bench_ret, taiex) -> dict:
    rows = []
    for p in picks[:max(TOPS)]:
        s = series.get(p["stock_id"])
        why = track.untradable(s, entry) if s is not None else "沒有股價資料"
        r = r_close = None
        path = []
        if s is not None and exit_ in s.index and d in s.index:
            r_close = float(s.at[exit_, "close"] / s.at[d, "close"] - 1)
        if not why and s is not None and exit_ in s.index:
            o = float(s.at[entry, "open"])
            r = float(s.at[exit_, "close"] / o - 1)
            path = [round(float(s.at[x, "close"] / o - 1) * 100, 2) if x in s.index else None for x in days]
        rows.append({"stock_id": p["stock_id"], "name": p.get("name"), "rank": p.get("rank"), "industry": p.get("industry"),
                     "liq": p.get("liq"), "rec": p.get("rec"), "factors": p.get("factors"),
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
                        "up": sum(1 for x in sub if (x["ret"] or 0) > 0),
                        "beat_bench": None if bench_ret is None else sum(1 for x in sub if x["ret"] is not None and x["ret"] / 100 > bench_ret),
                        "tradable": len(got), "n": len(sub), "missing": sum(1 for x in sub if x["exit_close"] is None),
                        "close_basis": round(float(np.mean(cb)) * 100, 2) if cb else None, "path": path}
    return {"top": tops, "stocks": rows}


def _ml_picks(d: str) -> dict:
    """名單日封存的影子預測（20 日）前幾名。沒有封存就不比較（不事後補算）。"""
    p = track.HISTORY.parent / "predictions" / f"{d}.json"
    if not p.exists():
        return {}
    rec = json.loads(p.read_text("utf-8"))
    out = {}
    for k in ("lgbm", "ens"):
        rows = [(s["id"], (s.get("20") or {}).get(k)) for s in rec.get("stocks", [])]
        rows = sorted([x for x in rows if x[1] is not None], key=lambda x: -x[1])[:max(TOPS)]
        if rows:
            out[k] = [{"stock_id": sid, "rank": i} for i, (sid, _) in enumerate(rows, 1)]
    return out


def diagnose(rows: list[dict], raw_top: dict | None, rec_top: dict | None) -> dict:
    """這一週前 10 名：拖累最多的股票、產業，流動性限制的差異，因子在贏家與輸家的差別。"""
    ok = [x for x in rows if x["ret"] is not None]
    if not ok:
        return {}
    worst = sorted(ok, key=lambda x: x["ret"])[:3]
    best = sorted(ok, key=lambda x: -x["ret"])[:3]
    ind = {}
    for x in ok:
        ind[x["industry"] or "（未分類）"] = ind.get(x["industry"] or "（未分類）", 0) + x["ret"] / len(rows)
    fac = {}
    win = [x for x in ok if x["ret"] > np.median([y["ret"] for y in ok])]
    lose = [x for x in ok if x not in win]
    for f in ("quality", "growth", "value", "momentum", "trend"):
        a = [x["factors"][f] for x in win if (x.get("factors") or {}).get(f) is not None]
        b = [x["factors"][f] for x in lose if (x.get("factors") or {}).get(f) is not None]
        if a and b:
            fac[f] = round(float(np.mean(a) - np.mean(b)), 1)
    return {"worst": [{k: x[k] for k in ("stock_id", "name", "industry", "ret")} for x in worst],
            "best": [{k: x[k] for k in ("stock_id", "name", "industry", "ret")} for x in best],
            "industry": sorted([{"industry": k, "contrib": round(v, 2)} for k, v in ind.items()], key=lambda r: r["contrib"]),
            "liquidity_effect": None if not raw_top or not rec_top else round(rec_top["net"] - raw_top["net"], 2),
            "factor_gap": fac, "note": "只有 10 檔、一週：因子差異只是描述，不代表因子有效或失效"}


def _summary(weeks: list[dict], key: str) -> dict:
    tally = {}
    for w in weeks:
        if not w.get("final"):
            continue
        for m, M in (w.get(key) or {}).items():
            for n, t in M["top"].items():
                if t["excess"] is not None:
                    tally.setdefault(m, {}).setdefault(n, []).append(t["excess"])
    return {m: {n: {"weeks": len(v), "beat": sum(x > 0 for x in v), "avg_excess": round(float(np.mean(v)), 2)} for n, v in T.items()}
            for m, T in tally.items()}


def build(dbname: str = "stocks.db") -> dict:
    logs = track.load()
    if not logs:
        return {"weeks": []}
    first = {}
    for L in logs:  # 每週第一份首次發布名單
        first.setdefault(_week(L["date"]), L)
    lists = [first[k] for k in sorted(first)][-WEEKS:]
    ids = sorted({p["stock_id"] for L in lists for key in ("picks", "rec_picks") for g in (L.get(key) or {}).values() for p in g}
                 | {p["stock_id"] for L in lists for g in _ml_picks(L["date"]).values() for p in g})
    series, idx, bench = track._prices(ids, lists[0]["date"], dbname)
    with db.connect(dbname) as con:
        info = {sid: (n, ind) for sid, n, ind in con.execute("SELECT stock_id, name, industry FROM stocks")}
    for L in lists:   # 舊紀錄的精選名單沒有存名稱與產業：從資料庫補（不影響報酬計算）
        for g in (L.get("rec_picks") or {}).values():
            for p in g:
                n, ind = info.get(p["stock_id"], (None, None))
                p.setdefault("name", n)
                p.setdefault("industry", ind)
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
        ev = lambda picks: _eval(picks, series, entry, exit_, d, days, bench_ret, taiex)
        modes = {m: ev(picks) for m, picks in L["picks"].items()}
        rec = {m: ev(picks) for m, picks in (L.get("rec_picks") or {}).items()}
        ml = {}
        P = _ml_picks(d)
        for k, picks in P.items():
            for x in picks:
                x["name"], x["industry"] = info.get(x["stock_id"], (None, None))
            ml[k] = ev(picks)
        for m in rec:   # 每週選股品質診斷（只描述這一週發生什麼事，不拿來調整模型）
            rec[m]["diagnosis"] = diagnose(rec[m]["stocks"][:10], modes.get(m, {}).get("top", {}).get("10"), rec[m]["top"].get("10"))
        weeks.append({"list_date": d, "generated_at": L.get("generated_at"), "model_version": L.get("model_version"),
                      "on_time": track.on_time(L, entry), "entry": entry, "exit": exit_, "days": days, "final": done,
                      "bench": None if bench_ret is None else round(bench_ret * 100, 2), "taiex": None if taiex is None else round(taiex * 100, 2),
                      "modes": modes, "rec_modes": rec, "ml": ml, "ml_note": "LightGBM／V21＋LightGBM 是影子模型：用名單日開盤前封存的 20 日預測，取前幾名，和固定名單同一套進出場與成本；沒有套用流動性限制"})
    # 累計：已經結束的週，每週成本後超額報酬
    summary = _summary(weeks, "modes")
    summary_rec = _summary(weeks, "rec_modes")
    summary_ml = _summary(weeks, "ml")
    return {"generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "cost_pct": round(track.COST * 100, 3),
            "bench": track.BENCH_ID, "tops": list(TOPS), "weeks": weeks[::-1], "summary": summary,
            "summary_rec": summary_rec, "summary_ml": summary_ml,
            "lists": {"modes": "原始模型排名（研究用，包含流動性不合格、不列入正式精選的股票）",
                      "rec_modes": "預設 AI 精選（正式推薦：排除流動性 E／F 級，D 級不進前 10 名）",
                      "ml": "影子模型（LightGBM 等，尚未通過驗收，只做同步比較）"}}


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
