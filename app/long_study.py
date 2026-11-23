"""長期買進條件重新設計（事前登記：docs/prereg_long_redesign_1123.md）→ history/backtest/long_redesign.json。

11/21 發現「本益比比自己過去便宜」的好公司長期反而輸（價值陷阱）。這裡把它換成「獲利和營收現在還在成長」，測 G1～G3。
分數、本益比、交易、成本、0050 基準都和 app/plan_audit.py 相同。

用法：python -m app.long_study [資料庫檔名]（PLAN_CACHE=1 沿用 data/plan_periods.pkl）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db, metrics, track, valuation
from . import pit_backtest as pb
from . import plan_audit as pa
from . import trade_rule_study as tr

STUDY_VERSION = "2026-11-23"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "long_redesign.json"
NAMES = {"G1": "品質、成長、分數 ≥ 60＋獲利成長＋營收成長（不看估值）",
         "G2": "G1＋本益比不高於自己兩年第 80 百分位",
         "G3": "G1＋收盤在年線之上",
         "OLD": "現行長期結論（本益比 ≤ 自己兩年中位數，11/21 已知沒過，對照）",
         "ALL": "所有有完整分數的股票（對照）"}


class Growth:
    """當時已公布的近四季 EPS 成長、近 3 個月營收年增。"""

    def __init__(self, fin: pd.DataFrame, rev: pd.DataFrame):
        self.eps = {}
        for sid, g in fin.groupby("stock_id"):
            s = valuation.eps_ttm_series(g, False)
            if len(s) >= 5:
                self.eps[sid] = ([x[0] for x in s], [x[1] for x in s])
        self.rev = {}
        for sid, g in rev.groupby("stock_id"):
            g = g.dropna(subset=["revenue"]).sort_values("ym")
            self.rev[sid] = (list(g["ym"]), g["revenue"].astype(float).to_numpy(), [pb.rev_available(y) for y in g["ym"]])

    def eps_up(self, sid: str, day: str) -> bool:
        x = self.eps.get(sid)
        if not x:
            return False
        when, v = x
        i = int(np.searchsorted(when, day, side="right")) - 1
        return i >= 4 and v[i] > 0 and v[i] > v[i - 4]

    def rev_up(self, sid: str, day: str) -> bool:
        x = self.rev.get(sid)
        if not x:
            return False
        yms, v, av = x
        k = int(np.searchsorted(av, day, side="right")) - 1
        if k < 0:
            return False
        idx = {y: j for j, y in enumerate(yms)}
        last = pd.Period(yms[k], "M")
        cur = [idx.get(str(last - m)) for m in range(3)]
        old = [idx.get(str(last - m - 12)) for m in range(3)]
        if None in cur or None in old:
            return False
        a, b = v[cur].sum(), v[old].sum()
        return b > 0 and a > b


def events(dbname: str, S: dict, pe: pa.PE, gr: Growth) -> list[dict]:
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT stock_id, date, open, high, low, close FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        bench, _ = track.benchmark(con)
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    O, C = {}, {}
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid):
            continue
        a, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        a = a.set_index("date")
        o = a["open"].astype(float)
        O[sid], C[sid] = o.where(o > 0, a["close"].astype(float)), a["close"].astype(float)
    O, C = pd.DataFrame(O).sort_index(), pd.DataFrame(C).sort_index()
    MA = C.rolling(240, min_periods=200).mean()
    dates = list(C.index)
    bo = bench["open"].reindex(dates).astype(float)
    out = []
    for sig in sorted(S):
        i = dates.index(sig) if sig in dates else None
        if i is None or i + 1 + pa.LONG_HOLD >= len(dates):
            continue
        a, b = i + 1, i + 1 + pa.LONG_HOLD
        br = bo.iloc[b] / bo.iloc[a] - 1
        if not np.isfinite(br):
            continue
        R = {k: [] for k in NAMES}
        for sid, (sc, q, g) in S[sig].items():
            if sid not in O:
                continue
            o_in = O[sid].iloc[a]
            if not np.isfinite(o_in) or o_in <= 0:
                continue
            o_out = O[sid].iloc[b]
            if not np.isfinite(o_out):
                last = C[sid].iloc[a:b + 1].dropna()
                o_out = float(last.iloc[-1]) if len(last) else np.nan
            if not np.isfinite(o_out):
                continue
            r = o_out / o_in - 1 - tr.COST
            R["ALL"].append(r)
            if sc is None or q is None or g is None or sc < 60 or q < 60 or g < 60:
                continue
            ma, cl = MA[sid].iloc[i], C[sid].iloc[i]
            cur, pct, p50 = pe.at(sid, sig)
            if cur is not None and p50 is not None and cur <= p50 and np.isfinite(ma) and cl >= ma * 0.9:
                R["OLD"].append(r)
            if not (gr.eps_up(sid, sig) and gr.rev_up(sid, sig)):
                continue
            R["G1"].append(r)
            if pct is None or pct < 80:
                R["G2"].append(r)
            if np.isfinite(ma) and cl >= ma:
                R["G3"].append(r)
        out.append({"date": sig, "bench": float(br), **{f"n_{k}": len(v) for k, v in R.items()},
                    **{k: (float(np.mean(v)) if v else None) for k, v in R.items()}})
    return out


def judge(E: list[dict], key: str) -> dict:
    v = pa.judge_long([{**e, "buy": e[key], "n_buy": e[f"n_{key}"]} for e in E], "buy")
    ok = (v["holdout_beat"] or 0) > 50
    v["checks"]["保留段贏 0050 的期間 > 50%"] = bool(ok)
    v["passed"] = bool(v["passed"] and ok)
    return {"name": NAMES[key], **v}


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    S = pa.score_periods(dbname, log)
    with db.connect(dbname) as con:
        raw = db.read(con, "SELECT stock_id, date, close FROM prices ORDER BY stock_id, date")
        fin = db.read(con, "SELECT stock_id, period, eps FROM financials")
        rev = db.read(con, "SELECT stock_id, ym, revenue FROM revenue")
    pe, gr = pa.PE(raw, fin), Growth(fin, rev)
    log(f"本益比與成長資料 {time.time() - t0:.0f} 秒")
    E = events(dbname, S, pe, gr)
    res = {k: judge(E, k) for k in NAMES}
    for k in ("OLD", "ALL"):
        res[k]["reference_only"] = True
    return {"schema_version": 1, "study_version": STUDY_VERSION, "prereg": "docs/prereg_long_redesign_1123.md",
            "generated_at": config.now().isoformat(timespec="seconds"), "data_end": raw["date"].max(),
            "holdout_from": pa.HOLDOUT_FROM, "hold": pa.LONG_HOLD, "cost_pct": round(tr.COST * 100, 3),
            "results": res, "passed": [k for k in ("G1", "G2", "G3") if res[k]["passed"]], "periods": E,
            "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "stocks.db")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    for k, v in r["results"].items():
        print(k, json.dumps({x: v[x] for x in v if x not in ("name", "by_year")}, ensure_ascii=False))
