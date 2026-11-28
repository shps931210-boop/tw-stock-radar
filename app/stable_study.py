"""穩健型抗跌研究（事前登記：docs/prereg_stable_downside_1128.md）→ history/backtest/stable_downside.json。

11/17、11/18 兩批 0050 下跌（−4.2%、−2.6%），穩健型卻跌 6.9%、7.3%：前 10 名有 5 檔航運、加上台塑化與鋼鐵。
穩健型只用「股價自己的波動」排除高風險股，沒有看「跟大盤一起跌的程度」（Beta）。這裡測：
LB 加上對 0050 的 Beta ≤ 0.8、LDB 加上 0050 下跌日的 Beta ≤ 0.8、RW 權重改成趨勢與成長為主、LBRW＝LB＋RW。
期數、成交、成本、基準都和 app/selection_study.py 相同（20 個交易日一期、不重疊、隔天開盤買、扣成本、0050）。

用法：python -m app.stable_study [資料庫檔名]（STABLE_CACHE=1 沿用 data/stable_periods.pkl）
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

from . import ai_model, config
from . import pit_backtest as pb
from . import selection_study as ss

STUDY_VERSION = "2026-11-28"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "history" / "backtest" / "stable_downside.json"
CACHE = ROOT / "data" / "stable_periods.pkl"
HOLDOUT_FROM = ss.HOLDOUT_FROM
N = 10
BETA_MAX = 0.8
BETA_DAYS = 250
BETA_MIN_DAYS = 120
T_NEED = 2.3
RW_WEIGHTS = {"trend": 0.30, "growth": 0.25, "quality": 0.20, "value": 0.15, "momentum": 0.10}
NAMES = {"BASE": "現行穩健型前 10 名（對照）",
         "LB": f"穩健型＋對 0050 的一年 Beta ≤ {BETA_MAX}",
         "LDB": f"穩健型＋0050 下跌日的 Beta ≤ {BETA_MAX}",
         "RW": "穩健型條件不變，權重改成趨勢 30%、成長 25%、品質 20%、估值 15%、動能 10%",
         "LBRW": "LB＋RW"}
FACT = list(ss.FACTORS)

_D = _CFG = None


def betas(D: dict, sig: str) -> dict:
    """訊號日（含）之前 250 個交易日，每檔股票對 0050 的 Beta 與 0050 下跌日的 Beta（還原除權息收盤）。"""
    b = D["adj"].get("0050")
    if b is None:
        return {}
    br = np.log(b["close"][b.index <= sig].astype(float)).diff().iloc[-BETA_DAYS:]
    out = {}
    for sid, p in D["adj"].items():
        if sid == "0050":
            continue
        s = p["close"][p.index <= sig]
        if len(s) < BETA_MIN_DAYS:
            continue
        sr = np.log(s.iloc[-BETA_DAYS - 1:].astype(float)).diff()
        j = pd.concat([sr, br], axis=1, join="inner").dropna()
        if len(j) < BETA_MIN_DAYS or j.iloc[:, 1].var() <= 0:
            continue
        x, y = j.iloc[:, 1].to_numpy(), j.iloc[:, 0].to_numpy()
        beta = float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))
        dn = x < 0
        db = float(np.cov(y[dn], x[dn])[0, 1] / np.var(x[dn], ddof=1)) if dn.sum() >= 40 and np.var(x[dn]) > 0 else None
        out[sid] = (beta, db)
    return out


def _period(p: tuple[str, str, str]) -> dict:
    sig, entry, exit_ = p
    D = _D
    rows = pb.scored_rows(D, sig, _CFG)
    B = betas(D, sig)
    recs = []
    for r in rows:
        if not r.get("is_common") or not ai_model.eligible(r, "stable"):
            continue
        sid = r["stock_id"]
        ret, st = pb.hold_return(D, sid, entry, exit_)
        be, db = B.get(sid, (None, None))
        recs.append({"id": sid, "industry": r.get("industry"), "elig": True, "complete": r.get("complete"),
                     "sc": r["ai"]["modes"]["stable"]["score"], "rk": r["ai"]["modes"]["stable"]["rank"],
                     **{f: r.get(f"score_{f}") for f in FACT}, "beta": be, "dbeta": db, "ret": ret, "status": st})
    return {"signal": sig, "entry": entry, "exit": exit_, "bench": pb._bench(D, entry, exit_),
            "df": pd.DataFrame(recs).set_index("id") if recs else pd.DataFrame()}


def collect(dbname: str, log=print) -> list[dict]:
    global _D, _CFG
    if os.environ.get("STABLE_CACHE") == "1" and CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    t0 = time.time()
    _D, _CFG = pb.load(dbname), config.load()
    ps = ss.signal_dates(_D["dates"])
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(ps)} 期")
    w = max(1, min(4, os.cpu_count() or 1))
    if w > 1 and sys.platform != "win32":
        import multiprocessing as mp
        with ProcessPoolExecutor(w, mp_context=mp.get_context("fork")) as ex:
            P = list(ex.map(_period, ps))
    else:
        P = [_period(p) for p in ps]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(P))
    log(f"評分完成 {time.time() - t0:.0f} 秒")
    return P


def _rw(df: pd.DataFrame) -> pd.Series:
    num = sum(df[f].fillna(0) * w for f, w in RW_WEIGHTS.items())
    den = sum(df[f].notna() * w for f, w in RW_WEIGHTS.items())
    s = num / den.replace(0, np.nan)
    return s.where(df["complete"] != False)  # noqa: E712  和 ai_model.score 一樣：資料不完整不排名


def pick(df: pd.DataFrame, rule: str) -> list[str]:
    if not len(df):
        return []
    ok = df[df["rk"].notna()]
    if rule in ("LB", "LBRW"):
        ok = ok[ok["beta"].notna() & (ok["beta"] <= BETA_MAX)]
    if rule == "LDB":
        ok = ok[ok["dbeta"].notna() & (ok["dbeta"] <= BETA_MAX)]
    if rule in ("RW", "LBRW"):
        s = _rw(ok).dropna()
        return list(s.sort_values(ascending=False, kind="mergesort").index[:N])
    return list(ok.sort_values("rk").index[:N])


def _mean(x) -> float | None:
    x = np.array(x, float)
    return round(float(x.mean()) * 100, 2) if len(x) else None


def judge(P: list[dict], R: dict, key: str) -> dict:
    b = np.array([p["bench"] for p in P], float)
    r, base = np.array(R[key], float), np.array(R["BASE"], float)
    dev = np.array([p["signal"] < HOLDOUT_FROM for p in P])
    hold, down = ~dev, b < 0
    d = r - base
    ex = r - b
    half = np.flatnonzero(dev)
    h1, h2 = half[: len(half) // 2], half[len(half) // 2:]
    t = ss._t(d[dev])
    v = {"name": NAMES[key],
         "dev": ss.summary(list(r[dev]), list(b[dev])), "holdout": ss.summary(list(r[hold]), list(b[hold])),
         "dev_down_excess": _mean(ex[dev & down]), "dev_up_excess": _mean(ex[dev & ~down]),
         "holdout_down_excess": _mean(ex[hold & down]), "holdout_up_excess": _mean(ex[hold & ~down]),
         "dev_down_periods": int((dev & down).sum()), "holdout_down_periods": int((hold & down).sum())}
    if key == "BASE":
        return v
    base_down = _mean((base - b)[dev & down])
    checks = {f"開發段比現行穩健型好 t ≥ {T_NEED}": t is not None and t >= T_NEED,
              "開發段前後兩半都比現行好": bool(d[h1].mean() > 0 and d[h2].mean() > 0),
              "保留段比現行好": bool(d[hold].mean() > 0),
              "開發段 0050 下跌期比現行少輸": bool(v["dev_down_excess"] is not None and base_down is not None and v["dev_down_excess"] > base_down)}
    beat = {"開發段平均贏 0050": bool(ex[dev].mean() > 0), "保留段平均贏 0050": bool(ex[hold].mean() > 0)}
    v |= {"vs_base_dev": _mean(d[dev]), "vs_base_t": None if t is None else round(t, 2),
          "vs_base_halves": [_mean(d[h1]), _mean(d[h2])], "vs_base_holdout": _mean(d[hold]),
          "checks": checks, "beat_0050": beat, "improves": all(checks.values()), "passed": all(checks.values()) and all(beat.values())}
    return v


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    P = collect(dbname, log)
    P = [p for p in P if p["bench"] is not None and np.isfinite(p["bench"])]
    R, picks = {}, {}
    for k in NAMES:
        picks[k] = [pick(p["df"], k) for p in P]
        R[k] = ss._net(P, picks[k])
    res = {k: judge(P, R, k) for k in NAMES}
    last = {k: picks[k][-1] for k in NAMES}
    beta_of = lambda k: [round(float(P[i]["df"].loc[picks[k][i], "beta"].mean()), 2) if picks[k][i] else None for i in range(len(P))]
    return {"schema_version": 1, "study_version": STUDY_VERSION, "prereg": "docs/prereg_stable_downside_1128.md",
            "generated_at": config.now().isoformat(timespec="seconds"), "holdout_from": HOLDOUT_FROM, "hold_days": ss.STEP,
            "cost_pct": round(pb.COST * 100, 3), "periods": len(P), "results": res,
            "passed": [k for k in NAMES if k != "BASE" and res[k]["passed"]],
            "improves": [k for k in NAMES if k != "BASE" and res[k]["improves"]],
            "avg_pick_beta": {k: float(np.nanmean([x for x in beta_of(k) if x is not None])) for k in NAMES},
            "last_picks": last, "rows": [{"signal": p["signal"], "bench": round(p["bench"] * 100, 2),
                                          **{k: round(R[k][i] * 100, 2) for k in NAMES}} for i, p in enumerate(P)],
            "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "stocks.db")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)), "utf-8")
    for k, v in r["results"].items():
        print(k, json.dumps({x: v[x] for x in v if x not in ("name",)}, ensure_ascii=False))
