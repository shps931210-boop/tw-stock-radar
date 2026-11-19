"""進出場規則贏 0050 的研究（事前登記：docs/prereg_trade_rules_1118.md）→ history/backtest/trade_rules.json。

波段進場訊號、成交限制、成本、基準都和 app/strategy.py 的回測同一套；候選只多加一個進場過濾或換一種出場。
AI 分數用每 20 個交易日、當時資料重算的均衡型分數（和 selection_study 同一套期間），訊號日用最近一期。

用法：python -m app.trade_rule_study
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db, metrics, strategy, track
from . import pit_backtest as pb
from .selection_study import HOLDOUT_FROM, signal_dates

STUDY_VERSION = "2026-11-18"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "trade_rules.json"
CACHE = config.DATA_DIR / "score_periods.pkl"   # 不進 git
START = "2019-08-01"
T_NEED = 2.6
MIN_TRADES = 300
GOOD = 60
HOLD = 20
COST = strategy.COST
NAMES = {"BASE": "現行波段規則（對照）", "F1": "加上：AI 均衡型分數 ≥ 60", "F2": "加上：加權指數在 120 日均線之上",
         "F3": "加上：近 60 日報酬強過 0050", "F4": "加上：AI 分數 ≥ 60 且大盤多頭", "X20": "出場改成固定持有 20 個交易日"}

_D = _CFG = None


def _scores(sig: str) -> tuple[str, dict]:
    rows = pb.scored_rows(_D, sig, _CFG)
    return sig, {r["stock_id"]: r["ai"]["modes"]["balanced"]["score"] for r in rows if r["ai"]["modes"]["balanced"]["score"] is not None}


def score_periods(dbname: str, log=print) -> dict[str, dict]:
    global _D, _CFG
    if CACHE.exists() and os.environ.get("SCORE_CACHE") == "1":
        return pickle.loads(CACHE.read_bytes())
    t0 = time.time()
    _D, _CFG = pb.load(dbname), config.load()
    sigs = [s for s, _, _ in signal_dates(_D["dates"], START)]
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(sigs)} 期分數")
    import multiprocessing as mp
    with ProcessPoolExecutor(max(1, min(4, os.cpu_count() or 1)), mp_context=mp.get_context("fork")) as ex:
        out = dict(ex.map(_scores, sigs))
    CACHE.write_bytes(pickle.dumps(out))
    return out


def trades(dbname: str) -> pd.DataFrame:
    """全部波段訊號的交易（現行出場）＋同一筆訊號固定持有 20 日的結果，和訊號日的過濾欄位。"""
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT * FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        idx = db.read(con, "SELECT date, close FROM index_prices ORDER BY date")
        bench, kind = track.benchmark(con)
    ix = idx.set_index("date")["close"].astype(float)
    bull = (ix >= ix.rolling(120).mean()).to_dict()
    b60 = (bench["close"] / bench["close"].shift(60) - 1).to_dict()
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    rows = []
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid) or len(p) < 100:
            continue
        p, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        f = strategy.frame(p)
        e = strategy.entries(f)
        o, c, h, lo = (f[k].to_numpy(float) for k in ("open", "close", "high", "low"))
        val = p["value"].to_numpy(float)
        dates = f["date"].to_numpy()
        b = bench.reindex(f["date"])
        bo, bc = b["open"].to_numpy(float), b["close"].to_numpy(float)
        nxt = nxt20 = 0
        for i in np.flatnonzero(e["swing"].to_numpy(bool) & f["liquid"].to_numpy(bool)):
            if dates[i] < START or i + 1 >= len(f):
                continue
            if o[i + 1] >= c[i] * (1 + track.LIMIT_UP) and h[i + 1] == lo[i + 1]:
                continue
            if val[i + 1] < track.THIN_VALUE:
                continue
            r = {"sid": sid, "date": dates[i], "bull": bool(bull.get(dates[i], False)),
                 "rs60": (c[i] / c[i - 60] - 1) - b60.get(dates[i], np.nan) if i >= 60 else np.nan}
            if i >= nxt:
                t = strategy.simulate(f, i, "swing")
                if t is not None:
                    br = bc[t["exit_i"]] / bo[i + 1] - 1
                    rows.append({**r, "rule": "swing", "ret": t["ret"], "excess": t["ret"] - br, "days": t["days"], "mae": t["mae"]})
                    nxt = t["exit_i"] + 1
            if i >= nxt20 and i + HOLD < len(f):
                j = i + HOLD
                ret = c[j] / o[i + 1] - 1 - COST
                br = bc[j] / bo[i + 1] - 1
                rows.append({**r, "rule": "x20", "ret": ret, "excess": ret - br, "days": HOLD,
                             "mae": float(lo[i + 1:j + 1].min() / o[i + 1] - 1)})
                nxt20 = j + 1
    return pd.DataFrame(rows)


def attach_scores(T: pd.DataFrame, S: dict[str, dict]) -> pd.DataFrame:
    sigs = sorted(S)
    k = np.searchsorted(np.array(sigs), T["date"].to_numpy(), side="right") - 1
    T = T.copy()
    T["score"] = [S[sigs[j]].get(sid) if j >= 0 else None for j, sid in zip(k, T["sid"])]
    T["score"] = pd.to_numeric(T["score"], errors="coerce")
    return T[k >= 0]


def select(T: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "X20":
        return T[T["rule"] == "x20"]
    X = T[T["rule"] == "swing"]
    if rule == "F1":
        return X[X["score"] >= GOOD]
    if rule == "F2":
        return X[X["bull"]]
    if rule == "F3":
        return X[X["rs60"] > 0]
    if rule == "F4":
        return X[(X["score"] >= GOOD) & X["bull"]]
    return X


def monthly(X: pd.DataFrame) -> pd.Series:
    return X.dropna(subset=["excess"]).groupby(X["date"].str[:7])["excess"].mean() * 100


def summarize(X: pd.DataFrame) -> dict:
    if not len(X):
        return {"n": 0}
    m = monthly(X)
    t = float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) >= 3 and m.std(ddof=1) > 0 else None
    pct = lambda v: round(float(v) * 100, 2)
    return {"n": int(len(X)), "months": int(len(m)), "month_excess": round(float(m.mean()), 2), "t": None if t is None else round(t, 2),
            "trade_excess": pct(X["excess"].mean()), "win": round(float((X["ret"] > 0).mean()) * 100, 1), "avg": pct(X["ret"].mean()),
            "beat": round(float((X["excess"] > 0).mean()) * 100, 1), "days": round(float(X["days"].mean()), 1), "mae": pct(X["mae"].mean()),
            "by_year": {y: round(float(g["excess"].mean() * 100), 2) for y, g in X.groupby(X["date"].str[:4])}}


def analyse(T: pd.DataFrame) -> dict:
    out = {"study_version": STUDY_VERSION, "prereg": "docs/prereg_trade_rules_1118.md", "generated_at": config.now().isoformat(timespec="seconds"),
           "names": NAMES, "gate": {"t": T_NEED, "min_trades": MIN_TRADES}, "rules": {}}
    for rule in NAMES:
        X = select(T, rule)
        dev, hold = X[X["date"] < HOLDOUT_FROM], X[X["date"] >= HOLDOUT_FROM]
        months = sorted(dev["date"].str[:7].unique())
        half = months[len(months) // 2] if months else ""
        d1, d2 = dev[dev["date"].str[:7] < half], dev[dev["date"].str[:7] >= half]
        sd, sh, s1, s2 = (summarize(x) for x in (dev, hold, d1, d2))
        res = {"name": NAMES[rule], "dev": sd, "dev_first": s1, "dev_second": s2, "holdout": sh}
        if rule != "BASE":
            ck = {"開發段贏 0050 且 t ≥ 2.6": sd.get("n", 0) > 0 and sd["month_excess"] > 0 and (sd["t"] or 0) >= T_NEED,
                  "開發段前後兩半都贏": s1.get("n", 0) > 0 and s2.get("n", 0) > 0 and s1["month_excess"] > 0 and s2["month_excess"] > 0,
                  "保留段贏 0050": sh.get("n", 0) > 0 and sh["month_excess"] > 0,
                  f"開發段交易數 ≥ {MIN_TRADES}": sd.get("n", 0) >= MIN_TRADES}
            res["gate"] = {"passed": all(ck.values()), "checks": ck}
        out["rules"][rule] = res
    return out


def run(dbname: str = "stocks.db", log=print) -> dict:
    S = score_periods(dbname, log)
    T = attach_scores(trades(dbname), S)
    r = analyse(T)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    return r


def main(argv: list[str]) -> int:
    r = run(argv[1] if len(argv) > 1 else "stocks.db")
    for k, v in r["rules"].items():
        g = v.get("gate")
        print(k, v["name"], "開發", v["dev"].get("month_excess"), "t", v["dev"].get("t"), "n", v["dev"].get("n"),
              "前後", v["dev_first"].get("month_excess"), v["dev_second"].get("month_excess"), "保留", v["holdout"].get("month_excess"),
              "" if g is None else ("通過" if g["passed"] else "未過"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
