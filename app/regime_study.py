"""上漲時跟得上 0050 的做法（事前登記：docs/prereg_regime_1128.md）→ history/backtest/regime.json。

11/28 穩健型研究發現：穩健型在 0050 下跌的期數平均贏 0050，輸在上漲期。這裡測：
SW1／SW3 大盤強（加權指數在 60／120 日均線之上）就整期持有 0050，弱才持有穩健型前 10 名；
SW2 同 SW1 但用均衡型；M50 市值前 50 大裡，一年動能（12-1 個月）最強的 10 檔等權。
期數、成交、成本、基準和 app/selection_study.py 相同。

用法：python -m app.regime_study [資料庫檔名]（REGIME_CACHE=1 沿用 data/regime_periods.pkl）
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

from . import config
from . import pit_backtest as pb
from . import selection_study as ss

STUDY_VERSION = "2026-11-28"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "history" / "backtest" / "regime.json"
CACHE = ROOT / "data" / "regime_periods.pkl"
HOLDOUT_FROM = ss.HOLDOUT_FROM
N = 10
BIG = 50
COST_0050 = 0.00385   # 0050 來回：手續費 0.1425%×2＋ETF 證交稅 0.1%（和 prereg_v22 相同）
T_NEED = 2.3
NAMES = {"SW1": "加權指數 ≥ 60 日均線持有 0050，否則穩健型前 10 名",
         "SW2": "加權指數 ≥ 60 日均線持有 0050，否則均衡型前 10 名",
         "SW3": "加權指數 ≥ 120 日均線持有 0050，否則穩健型前 10 名",
         "M50": f"市值前 {BIG} 大裡一年動能（12-1 個月）最強的 {N} 檔等權",
         "STABLE": "穩健型前 10 名（對照）", "BALANCED": "均衡型前 10 名（對照）"}

_D = _CFG = None


def _above(D: dict, sig: str, n: int) -> bool | None:
    s = D["idx_s"]
    if s is None:
        return None
    s = s[s.index <= sig].astype(float)
    return None if len(s) < n else bool(s.iloc[-1] >= s.iloc[-n:].mean())


def _period(p: tuple[str, str, str]) -> dict:
    sig, entry, exit_ = p
    D = _D
    rows = pb.scored_rows(D, sig, _CFG)
    recs = []
    caps = sorted(((r.get("market_cap") or 0, r["stock_id"]) for r in rows if r.get("is_common") and r.get("base_ok")), reverse=True)
    big = {sid for _, sid in caps[:BIG]}
    for r in rows:
        rk_s, rk_b = r["ai"]["modes"]["stable"]["rank"], r["ai"]["modes"]["balanced"]["rank"]
        sid = r["stock_id"]
        if not (rk_s and rk_s <= 30) and not (rk_b and rk_b <= 30) and sid not in big:
            continue
        ret, st = pb.hold_return(D, sid, entry, exit_)
        recs.append({"id": sid, "rk_stable": rk_s, "rk_balanced": rk_b, "big": sid in big,
                     "market_cap": r.get("market_cap"), "mom": r.get("mom_12_1"), "ret": ret, "status": st})
    return {"signal": sig, "entry": entry, "exit": exit_, "bench": pb._bench(D, entry, exit_),
            "up60": _above(D, sig, 60), "up120": _above(D, sig, 120),
            "df": pd.DataFrame(recs).set_index("id") if recs else pd.DataFrame()}


def collect(dbname: str, log=print) -> list[dict]:
    global _D, _CFG
    if os.environ.get("REGIME_CACHE") == "1" and CACHE.exists():
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
    return P


def top(df: pd.DataFrame, mode: str) -> list[str]:
    if not len(df):
        return []
    c = f"rk_{mode}"
    return list(df[df[c].notna()].sort_values(c).index[:N])


def m50(df: pd.DataFrame) -> list[str]:
    if not len(df):
        return []
    b = df[df["big"] & df["mom"].notna()]
    return list(b.sort_values("mom", ascending=False, kind="mergesort").index[:N])


def returns(P: list[dict], key: str) -> list[float]:
    """每期成本後報酬。持有 0050 的期數報酬＝0050，從股票換進 0050 扣 0050 來回成本；股票只對新買進的部位扣成本。"""
    out, prev = [], None  # prev：None、"0050" 或上一期的股票清單
    for p in P:
        if key in ("SW1", "SW2", "SW3"):
            up = p["up120"] if key == "SW3" else p["up60"]
            if up:
                out.append(p["bench"] - (COST_0050 if prev != "0050" else 0.0))
                prev = "0050"
                continue
            ids = top(p["df"], "balanced" if key == "SW2" else "stable")
        elif key == "M50":
            ids = m50(p["df"])
        else:
            ids = top(p["df"], key.lower())
        if not ids:
            out.append(0.0)
            prev = None
            continue
        r = p["df"]["ret"].reindex(ids)
        held = set(prev) if isinstance(prev, list) else set()
        new = [s for s in ids if s not in held and pd.notna(r.get(s))]
        out.append(float(r.fillna(0).sum() / len(ids) - pb.COST * len(new) / len(ids)))
        prev = ids
    return out


def judge(P: list[dict], r: list[float], key: str) -> dict:
    b = np.array([p["bench"] for p in P], float)
    r = np.array(r, float)
    ex = r - b
    dev = np.array([p["signal"] < HOLDOUT_FROM for p in P])
    hold, down = ~dev, b < 0
    idx = np.flatnonzero(dev)
    h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    t = ss._t(ex[dev])
    pct = lambda x: round(float(np.mean(x)) * 100, 2) if len(x) else None
    checks = {f"開發段平均贏 0050 且 t ≥ {T_NEED}": bool(ex[dev].mean() > 0 and t is not None and t >= T_NEED),
              "開發段前後兩半都贏 0050": bool(ex[h1].mean() > 0 and ex[h2].mean() > 0),
              "保留段平均贏 0050": bool(ex[hold].mean() > 0)}
    v = {"name": NAMES[key], "dev": ss.summary(list(r[dev]), list(b[dev])), "holdout": ss.summary(list(r[hold]), list(b[hold])),
         "halves": [pct(ex[h1]), pct(ex[h2])], "dev_down_excess": pct(ex[dev & down]), "dev_up_excess": pct(ex[dev & ~down]),
         "holdout_down_excess": pct(ex[hold & down]), "holdout_up_excess": pct(ex[hold & ~down])}
    if key in ("SW1", "SW2", "SW3"):
        flag = "up120" if key == "SW3" else "up60"
        v["share_in_0050"] = round(float(np.mean([bool(p[flag]) for p in P])) * 100, 1)
    if key in ("STABLE", "BALANCED"):
        v["reference_only"] = True
        return v
    v |= {"checks": checks, "passed": all(checks.values())}
    return v


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    P = [p for p in collect(dbname, log) if p["bench"] is not None and np.isfinite(p["bench"])]
    R = {k: returns(P, k) for k in NAMES}
    res = {k: judge(P, R[k], k) for k in NAMES}
    return {"schema_version": 1, "study_version": STUDY_VERSION, "prereg": "docs/prereg_regime_1128.md",
            "generated_at": config.now().isoformat(timespec="seconds"), "holdout_from": HOLDOUT_FROM, "hold_days": ss.STEP,
            "cost_pct": round(pb.COST * 100, 3), "cost_0050_pct": COST_0050 * 100, "periods": len(P), "results": res,
            "passed": [k for k in res if res[k].get("passed")],
            "last": {"signal": P[-1]["signal"], "up60": P[-1]["up60"], "up120": P[-1]["up120"], "M50": m50(P[-1]["df"])},
            "rows": [{"signal": p["signal"], "bench": round(p["bench"] * 100, 2), **{k: round(R[k][i] * 100, 2) for k in NAMES}}
                     for i, p in enumerate(P)], "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "stocks.db")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    for k, v in r["results"].items():
        print(k, json.dumps({x: v[x] for x in v if x != "name"}, ensure_ascii=False))
