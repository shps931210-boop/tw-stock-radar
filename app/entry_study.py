"""V21.2 進場方法研究（事前登記：docs/prereg_v21_2.md 第三節）→ history/backtest/entry.json。

同一份名單（均衡型前 10 名）、同一段日曆期間（A 的進場日開盤 → A 的出場日開盤）比較三種進場方法：
A 隔日開盤直接買；B 等收盤站上 20 日均線且放量；C 等回檔 3% 後重新轉強。
沒買到的部位放現金（報酬 0），錯失的股票另外列出它在 A 期間漲跌多少，避免只看成交的交易。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import pit_backtest as pb
from . import academic_study as ac
from . import selection_study as ss

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "entry.json"
STUDY_VERSION = "2026-10-29"
WAIT = 10
PULLBACK = 0.03
T_NEED = 2.24
HOLDOUT_FROM = ac.HOLDOUT_FROM


def trigger(a: pd.DataFrame, dates: list[str], sig: str, method: str) -> int | None:
    """回傳「觸發日」在 dates 的位置（隔天開盤買）；A 的觸發日就是訊號日。WAIT 個交易日內沒觸發回傳 None。"""
    i = dates.index(sig)
    if method == "A":
        return i
    c, v, h = a["close"], a["volume"], a["high"]
    low_seen = False
    ref = c.get(sig)
    for k in range(i + 1, min(i + 1 + WAIT, len(dates) - 1)):
        d = dates[k]
        if d not in a.index or not a.at[d, "close"]:
            continue
        hist = [x for x in dates[max(0, k - 20):k] if x in a.index]
        if len(hist) < 15:
            continue
        if method == "B":
            ma = c.reindex(dates[k - 19:k + 1]).mean()
            if a.at[d, "close"] > ma and a.at[d, "volume"] > v.reindex(hist).mean():
                return k
        else:
            if ref and a.at[d, "close"] <= ref * (1 - PULLBACK):
                low_seen = True
                continue
            prev = dates[k - 1]
            if low_seen and prev in a.index and a.at[d, "close"] > a.at[prev, "high"]:
                return k
    return None


def period(D: dict, P: dict, ids: list[str], method: str) -> dict:
    dates = D["dates"]
    entry, exit_ = P["entry"], P["exit"]
    ie = dates.index(exit_)
    rets, waits, filled, missed = [], [], 0, []
    for sid in ids:
        a = D["adj"].get(sid)
        full = pb.hold_return(D, sid, entry, exit_)[0]
        if a is None or full is None:
            rets.append(0.0)       # A 也買不到：同樣放現金
            continue
        k = trigger(a, dates, P["signal"], method)
        buy = None if k is None else dates[k + 1]
        if buy is None or dates.index(buy) >= ie:
            missed.append(full)
            rets.append(0.0)
            continue
        r, _ = pb.hold_return(D, sid, buy, exit_)
        if r is None:
            missed.append(full)
            rets.append(0.0)
            continue
        filled += 1
        waits.append(dates.index(buy) - dates.index(entry))
        rets.append(r - pb.COST)
    return {"net": float(np.mean(rets)), "filled": filled, "n": len(ids), "wait": waits, "missed": missed}


def run(D: dict | None = None, log=print) -> dict:
    from .factor_audit import load_fs
    D = D or pb.load("stocks.db")
    FS = load_fs()
    periods = [P for P, _ in FS]
    bench = [P["bench"] for P in periods]
    dv = np.array([P["signal"] < HOLDOUT_FROM for P in periods])
    ids = [list(P["df"][P["df"]["rk_balanced"].notna()].sort_values("rk_balanced").index[:10]) for P in periods]
    R, S = {}, {}
    for m in "ABC":
        rows = [period(D, P, x, m) for P, x in zip(periods, ids)]
        R[m] = [r["net"] for r in rows]
        allw = [w for r in rows for w in r["wait"]]
        miss = [x for r in rows for x in r["missed"]]
        S[m] = {"rule": {"A": "隔日開盤直接買", "B": "收盤站上 20 日均線且放量後隔日開盤", "C": "回檔 3% 後收盤突破前一日高點，隔日開盤"}[m],
                **ss.summary(R[m], bench), "fill_pct": round(sum(r["filled"] for r in rows) / sum(r["n"] for r in rows) * 100, 1),
                "avg_wait_days": round(float(np.mean(allw)), 2) if allw else None,
                "missed_n": len(miss), "missed_avg_ret_pct": round(float(np.mean(miss)) * 100, 2) if miss else None,
                "dev": ss.summary(list(np.array(R[m])[dv]), list(np.array(bench)[dv])),
                "holdout": ss.summary(list(np.array(R[m])[~dv]), list(np.array(bench)[~dv]))}
    cmp = {}
    for m in "BC":
        d = np.array(R[m]) - np.array(R["A"])
        dd = d[dv]
        h = len(dd) // 2
        v = {"dev_diff": ac._r(dd.mean() * 100), "dev_t": ac._r(ac._t(dd)), "first_half": ac._r(dd[:h].mean() * 100),
             "second_half": ac._r(dd[h:].mean() * 100), "holdout_diff": ac._r(d[~dv].mean() * 100)}
        v["passed"] = bool(v["dev_t"] is not None and v["dev_t"] >= T_NEED and dd[:h].mean() > 0 and dd[h:].mean() > 0 and d[~dv].mean() > 0)
        cmp[f"{m}_vs_A"] = v
    return {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
            "periods": len(periods), "wait_days": WAIT, "pullback": PULLBACK, "t_need": T_NEED, "holdout_from": HOLDOUT_FROM,
            "methods": S, "compare": cmp, "passed": [k for k, v in cmp.items() if v["passed"]],
            "note": "同一段日曆期間比較；沒買到放現金，錯失的股票另列 A 期間報酬。歷史回測（研究），不改正式推薦。"}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps({"methods": {k: {x: v[x] for x in ("fill_pct", "avg_wait_days", "missed_avg_ret_pct")} for k, v in res["methods"].items()},
                      "compare": res["compare"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
